import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from xgboost import XGBRegressor


BURST_B_POP_VECTOR_PATH = Path(
    "./data/Model_features_predicting_neural_data/"
    "Predicting_burst_B/pop_vector_mean_burst_rate_epochB.h5"
)
BURST_B_TRIAL_PATH = Path(
    "./data/Model_features_predicting_neural_data/"
    "Predicting_burst_B/hit_go_imgs_initial_changed.h5"
)
VIT_EMBEDDING_CACHE_PATH = Path("./data/embeddings/vit_b16_embeddings.npz")
SANITY_CHECK_RESULTS_DIR = Path("./results/sanity_check")
DEFAULT_PER_DIM_METRIC_CSV = Path(
    "./results/shap_control/burst_B/fam-none_dev-gaussian_trials_seed42/"
    "per_dim_burst_B_full_nosup_fam-none_dev-gaussian_trials_seed42.csv"
)
SESSION_IDS = [
    "session_1048196054",
    "session_1053925378",
    "session_1063010385",
    "session_1064639378",
    "session_1065908084",
    "session_1067781390",
    "session_1081431006",
    "session_1086410738",
    "session_1091039376",
    "session_1092466205",
    "session_1096935816",
    "session_1104297538",
    "session_1108528422",
    "session_1108531612",
    "session_1112515874",
    "session_1115356973",
    "session_1120251466",
    "session_1121607504",
    "session_1122903357",
    "session_1130349290",
]


def print_burst_b_session_stats() -> None:
    with h5py.File(BURST_B_POP_VECTOR_PATH, "r") as f:
        session_ids = sorted(f.keys())

        print(f"burst_B session count: {len(session_ids)}")
        print(f"source file: {BURST_B_POP_VECTOR_PATH}")
        print("-" * 80)

        for session_id in session_ids:
            pop_vector = f[session_id]["pop_vector"]
            shape = pop_vector.shape

            if len(shape) == 2:
                n_samples, feature_dim = shape
            elif len(shape) == 1:
                n_samples = shape[0]
                feature_dim = 1
            else:
                n_samples = shape[0] if shape else 0
                feature_dim = shape[1:] if len(shape) > 1 else 1

            print(
                f"{session_id}: "
                f"n_samples={n_samples}, "
                f"dim={feature_dim}, "
                f"full_shape={shape}"
            )


def _default_xgb_params(random_state: int = 42) -> dict:
    return dict(
        objective="reg:squarederror",
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        tree_method="hist",
        random_state=random_state,
    )


def load_pop_vec(session_id: str, predict: str = "burst_B") -> np.ndarray:
    if predict != "burst_B":
        raise ValueError(f"Unsupported predict={predict}")

    with h5py.File(BURST_B_POP_VECTOR_PATH, "r") as f:
        if session_id not in f:
            raise KeyError(f"Session {session_id} not found in {BURST_B_POP_VECTOR_PATH}")
        pop_vec = np.asarray(f[session_id]["pop_vector"])

    if pop_vec.ndim != 2:
        raise ValueError(f"Expected 2D pop_vector, got shape {pop_vec.shape}")
    return pop_vec


def load_image_trial(session_id: str) -> np.ndarray:
    with h5py.File(BURST_B_TRIAL_PATH, "r") as f:
        if session_id not in f:
            raise KeyError(f"Session {session_id} not found in {BURST_B_TRIAL_PATH}")
        imgs_id = np.asarray(f[session_id]["hit_go_imgs"]).T

    if imgs_id.ndim != 2:
        raise ValueError(f"Unexpected hit_go_imgs shape: {imgs_id.shape}")

    return np.asarray(
        [
            [
                x.decode("utf-8") if isinstance(x, (bytes, np.bytes_)) else str(x)
                for x in row
            ]
            for row in imgs_id
        ],
        dtype=str,
    )


def load_cached_vit_embeddings(cache_path: Path = VIT_EMBEDDING_CACHE_PATH) -> dict[str, np.ndarray]:
    npz = np.load(cache_path)
    return {key: np.asarray(npz[key], dtype=np.float32) for key in npz.files}


def prepare_data(session_id: str, pred: str = "burst_B") -> tuple[np.ndarray, np.ndarray, dict]:
    imgs_id = load_image_trial(session_id=session_id)
    ntrials = imgs_id.shape[0]

    unique_pairs, inverse_indices = np.unique(imgs_id, axis=0, return_inverse=True)
    duplicate_groups = [np.where(inverse_indices == k)[0] for k in range(unique_pairs.shape[0])]

    pop_raw = load_pop_vec(session_id=session_id, predict=pred)
    if pop_raw.shape[0] != ntrials:
        raise ValueError(f"pop_vec ntrials {pop_raw.shape[0]} != image trials {ntrials}")

    y = np.zeros((unique_pairs.shape[0], pop_raw.shape[1]), dtype=np.float32)
    for k, idx in enumerate(duplicate_groups):
        y[k] = pop_raw[idx].mean(axis=0)

    label_to_emb = load_cached_vit_embeddings()
    example_emb = next(iter(label_to_emb.values()))
    d_img = example_emb.shape[0]
    X = np.zeros((unique_pairs.shape[0], d_img * 2), dtype=np.float32)

    for k, (img_init, img_changed) in enumerate(unique_pairs):
        if img_init not in label_to_emb:
            raise KeyError(f"Image label '{img_init}' not found in cached embeddings.")
        if img_changed not in label_to_emb:
            raise KeyError(f"Image label '{img_changed}' not found in cached embeddings.")
        X[k] = np.concatenate([label_to_emb[img_init], label_to_emb[img_changed]], axis=0)

    meta = {
        "pred": pred,
        "ntrials_raw": int(ntrials),
        "ntrials_unique": int(unique_pairs.shape[0]),
        "unique_pairs": unique_pairs,
        "duplicate_groups": duplicate_groups,
    }
    return X, y, meta


def analyze_single_session_prediction(
    session_id: str,
    pred: str = "burst_B",
    n_splits: int = 5,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    X, y, meta = prepare_data(
        session_id=session_id,
        pred=pred,
    )
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    n_samples = X.shape[0]
    n_splits_eff = min(n_splits, n_samples)
    if n_splits_eff < 2:
        raise ValueError(f"Session {session_id} has too few samples: {n_samples}")

    kf = KFold(n_splits=n_splits_eff, shuffle=True, random_state=random_state)
    y_pred_oof = np.zeros_like(y)
    xgb_params = _default_xgb_params(random_state=random_state)

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]

        for d in range(y.shape[1]):
            model = XGBRegressor(**xgb_params)
            model.fit(X_train, y_train[:, d])
            y_pred_oof[test_idx, d] = model.predict(X_test)

    y_true_flat = y.ravel()
    y_pred_flat = y_pred_oof.ravel()
    pearson = np.corrcoef(y_true_flat, y_pred_flat)[0, 1]
    r2 = r2_score(y_true_flat, y_pred_flat)

    print("-" * 80)
    print(f"Single-session prediction check: {session_id}")
    print(f"raw_trials={meta['ntrials_raw']}, unique_trials={meta['ntrials_unique']}")
    print(f"X shape={X.shape}, y shape={y.shape}")
    print(f"OOF Pearson={pearson:.4f}, OOF R2={r2:.4f}")

    SANITY_CHECK_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = SANITY_CHECK_RESULTS_DIR / f"{session_id}_{pred}_oof_prediction_vs_gt.png"

    vmin = float(min(y.min(), y_pred_oof.min()))
    vmax = float(max(y.max(), y_pred_oof.max()))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=160, constrained_layout=True)

    im0 = axes[0].imshow(y, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0].set_title(f"Ground Truth\n{session_id}")
    axes[0].set_xlabel("Neuron dim")
    axes[0].set_ylabel("Trial")

    im1 = axes[1].imshow(y_pred_oof, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    axes[1].set_title(f"OOF Prediction\nPearson={pearson:.4f}, R2={r2:.4f}")
    axes[1].set_xlabel("Neuron dim")
    axes[1].set_ylabel("Trial")

    axes[2].scatter(y_true_flat, y_pred_flat, s=12, alpha=0.5, edgecolors="none")
    axes[2].plot([vmin, vmax], [vmin, vmax], linestyle="--", color="red", linewidth=1.2)
    axes[2].set_title("Prediction vs Ground Truth")
    axes[2].set_xlabel("Ground Truth")
    axes[2].set_ylabel("Prediction")
    axes[2].grid(alpha=0.25, linestyle="--")

    cbar = fig.colorbar(im1, ax=axes, shrink=0.95)
    cbar.set_label("Response value")

    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {save_path}")

    return y, y_pred_oof


def summarize_per_dim_metrics_by_session(
    csv_path: Path,
    metrics: tuple[str, ...] = ("r2", "pearson"),
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    df = pd.read_csv(csv_path)

    required_columns = {"session_id", *metrics}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        missing_str = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns in {csv_path}: {missing_str}")

    per_session = (
        df.groupby("session_id", dropna=False)[list(metrics)]
        .mean(numeric_only=True)
        .sort_index()
    )
    metric_mean = per_session.mean(numeric_only=True)
    metric_std = per_session.std(numeric_only=True, ddof=1)

    print("-" * 80)
    print(f"Per-session metric summary from: {csv_path}")
    print(f"session_count={len(per_session)}")
    print("Per-session means:")
    print(per_session.to_string(float_format=lambda x: f"{x:.6f}"))
    print("\nAcross-session mean:")
    print(metric_mean.to_string(float_format=lambda x: f"{x:.6f}"))
    print("\nAcross-session std (ddof=1):")
    print(metric_std.to_string(float_format=lambda x: f"{x:.6f}"))

    return per_session, metric_mean, metric_std


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("burst_b_stats", "single_session_pred", "per_dim_metric_summary"),
        default="per_dim_metric_summary",
        help="Select which sanity check to run.",
    )
    parser.add_argument(
        "--session-id",
        default=SESSION_IDS[6],
        help="Session used by --mode single_session_pred.",
    )
    parser.add_argument(
        "--pred",
        default="burst_B",
        help="Prediction target used by --mode single_session_pred.",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=DEFAULT_PER_DIM_METRIC_CSV,
        help="Per-dimension metric CSV used by --mode per_dim_metric_summary.",
    )
    args = parser.parse_args()

    if args.mode == "burst_b_stats":
        print_burst_b_session_stats()
        return

    if args.mode == "single_session_pred":
        analyze_single_session_prediction(session_id=args.session_id, pred=args.pred)
        return

    summarize_per_dim_metrics_by_session(csv_path=args.csv_path)


if __name__ == "__main__":
    main()
