import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from importance_control import ControlConfig, prepare_data_with_control

try:
    import torch
except ModuleNotFoundError:
    torch = None


class ViTEmbeddingFeatureEngineer:
    def __init__(self, variance_threshold=0.95, max_components=30, fixed_components=None):
        self.variance_threshold = variance_threshold
        self.max_components = max_components
        self.fixed_components = fixed_components
        self.pca_diff = None
        self.pca_had = None
        self.scaler = StandardScaler()

    def _split_and_compute(self, X):
        if torch is not None and isinstance(X, torch.Tensor):
            if X.is_cuda:
                X = X.cpu()
            X_np = X.detach().numpy().astype(np.float32)
        else:
            X_np = np.asarray(X, dtype=np.float32)

        A = X_np[:, :768]
        B = X_np[:, 768:]

        diff = A
        hadamard = B

        return diff, hadamard

    def _n_components(self, X_train):
        max_valid = min(X_train.shape[0] - 1, X_train.shape[1], self.max_components)
        if max_valid <= 0:
            raise ValueError(
                f"Not enough training samples for PCA: got shape {X_train.shape}"
            )

        if self.fixed_components is not None:
            fixed_components = max(int(self.fixed_components), 1)
            if max_valid < fixed_components:
                raise ValueError(
                    f"Cannot keep PCA fixed at {fixed_components} components with training shape "
                    f"{X_train.shape}; maximum valid components is {max_valid}."
                )
            return fixed_components

        pca_tmp = PCA(n_components=max_valid).fit(X_train)
        cumvar = np.cumsum(pca_tmp.explained_variance_ratio_)
        n = int(np.argmax(cumvar >= self.variance_threshold) + 1)
        return min(max(n, 1), max_valid)

    def fit_transform(self, X) -> np.ndarray:
        diff, hadamard = self._split_and_compute(X)

        n_diff = self._n_components(diff)
        n_had = self._n_components(hadamard)

        self.pca_diff = PCA(n_components=n_diff)
        self.pca_had = PCA(n_components=n_had)

        diff_r = self.pca_diff.fit_transform(diff)
        had_r = self.pca_had.fit_transform(hadamard)

        features = np.concatenate([diff_r, had_r], axis=1)
        features = self.scaler.fit_transform(features)

        print(
            f"  [FE] diff PCA: {n_diff} dims, "
            f"hadamard PCA: {n_had} dims -> total {features.shape[1]} dims"
        )
        return features.astype(np.float32)

    def transform(self, X) -> np.ndarray:
        assert self.pca_diff is not None, "Call fit_transform first."
        diff, hadamard = self._split_and_compute(X)

        diff_r = self.pca_diff.transform(diff)
        had_r = self.pca_had.transform(hadamard)

        features = np.concatenate([diff_r, had_r], axis=1)
        features = self.scaler.transform(features)
        return features.astype(np.float32)


@dataclass
class FENeuronConfig:
    session_id: str = "session_1091039376"
    pred: str = "burst_B"
    pred_mean: bool = False
    need_sup_feat: bool = False
    n_splits: int = 5
    random_state: int = 42
    results_root: str = "./results/fe_neuron"
    variance_threshold: float = 0.95
    max_components: int = 30
    fixed_embedding_components: int = 6
    familiar_mode: str = "gaussian_trials"
    deviant_mode: str = "gaussian_trials"
    neuron_keep_fraction: float = 0.2
    min_neurons_to_keep: int = 3
    figure_dpi: int = 180


def _build_control_config(config: FENeuronConfig) -> ControlConfig:
    control_config = ControlConfig(
        familiar_mode=config.familiar_mode,
        deviant_mode=config.deviant_mode,
        random_state=config.random_state,
    )
    control_config.validate()

    allowed_modes = {"none", "gaussian_trials", "scramble_pixels"}
    if control_config.familiar_mode not in allowed_modes:
        raise ValueError(
            f"Unsupported familiar_mode={control_config.familiar_mode!r} for single_session_neuron_plots. "
            f"Allowed modes: {sorted(allowed_modes)}"
        )
    if control_config.deviant_mode not in allowed_modes:
        raise ValueError(
            f"Unsupported deviant_mode={control_config.deviant_mode!r} for single_session_neuron_plots. "
            f"Allowed modes: {sorted(allowed_modes)}"
        )
    return control_config


def default_xgb_params(random_state: int = 42) -> Dict:
    return {
        "objective": "reg:squarederror",
        "max_depth": 3,
        "min_child_weight": 8,
        "reg_alpha": 0.1,
        "reg_lambda": 3.0,
        "gamma": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "colsample_bylevel": 0.8,
        "learning_rate": 0.02,
        "n_estimators": 2000,
        "early_stopping_rounds": 40,
        "tree_method": "hist",
        "n_jobs": 1,
        "random_state": random_state,
        "verbosity": 0,
    }


def _safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    if x.size < 2 or y.size < 2:
        return float("nan")
    if np.allclose(x.std(), 0.0) or np.allclose(y.std(), 0.0):
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _nanmean_or_nan(values: np.ndarray) -> float:
    finite = np.isfinite(values)
    if not finite.any():
        return float("nan")
    return float(values[finite].mean())


def _nanstd_or_nan(values: np.ndarray) -> float:
    finite = np.isfinite(values)
    if not finite.any():
        return float("nan")
    return float(values[finite].std())


def _nanmin_or_nan(values: np.ndarray) -> float:
    finite = np.isfinite(values)
    if not finite.any():
        return float("nan")
    return float(np.min(values[finite]))


def _nanmax_or_nan(values: np.ndarray) -> float:
    finite = np.isfinite(values)
    if not finite.any():
        return float("nan")
    return float(np.max(values[finite]))


def _select_neurons_by_pearson(
    pearson_per_neuron: np.ndarray,
    keep_fraction: float,
    min_neurons_to_keep: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_neurons = len(pearson_per_neuron)
    if n_neurons == 0:
        raise ValueError("No neurons found for filtering.")

    keep_fraction = float(np.clip(keep_fraction, 0.0, 1.0))
    n_keep = int(math.ceil(n_neurons * keep_fraction))
    n_keep = max(n_keep, int(min_neurons_to_keep))
    n_keep = min(max(n_keep, 1), n_neurons)

    sort_key = np.where(np.isfinite(pearson_per_neuron), pearson_per_neuron, -np.inf)
    keep_indices = np.argsort(sort_key, kind="stable")[-n_keep:]
    keep_mask = np.zeros(n_neurons, dtype=bool)
    keep_mask[keep_indices] = True
    drop_indices = np.flatnonzero(~keep_mask)
    return keep_mask, drop_indices


def _summarize_filtered_session(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    keep_mask: np.ndarray,
    r2_per_neuron: np.ndarray,
    pearson_per_neuron: np.ndarray,
) -> dict[str, float]:
    y_true_kept = y_true[:, keep_mask]
    y_pred_kept = y_pred[:, keep_mask]
    r2_trial_kept, pearson_trial_kept = _compute_prediction_metrics_per_trial(
        y_true_kept,
        y_pred_kept,
    )
    return {
        "n_neurons_kept": int(keep_mask.sum()),
        "n_neurons_dropped": int((~keep_mask).sum()),
        "y_true_std_kept": float(np.std(y_true_kept)),
        "y_pred_std_kept": float(np.std(y_pred_kept)),
        "pearson_mean_kept": _nanmean_or_nan(pearson_per_neuron[keep_mask]),
        "pearson_std_kept": _nanstd_or_nan(pearson_per_neuron[keep_mask]),
        "pearson_min_kept": _nanmin_or_nan(pearson_per_neuron[keep_mask]),
        "pearson_max_kept": _nanmax_or_nan(pearson_per_neuron[keep_mask]),
        "r2_mean_kept": _nanmean_or_nan(r2_per_neuron[keep_mask]),
        "r2_std_kept": _nanstd_or_nan(r2_per_neuron[keep_mask]),
        "r2_min_kept": _nanmin_or_nan(r2_per_neuron[keep_mask]),
        "r2_max_kept": _nanmax_or_nan(r2_per_neuron[keep_mask]),
        "pearson_mean_per_trial_kept": _nanmean_or_nan(pearson_trial_kept),
        "pearson_std_per_trial_kept": _nanstd_or_nan(pearson_trial_kept),
        "r2_mean_per_trial_kept": _nanmean_or_nan(r2_trial_kept),
        "r2_std_per_trial_kept": _nanstd_or_nan(r2_trial_kept),
    }


def _compute_prediction_metrics_per_dim(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n_targets = y_true.shape[1]
    r2_vals = np.zeros(n_targets, dtype=np.float32)
    pearson_vals = np.full(n_targets, np.nan, dtype=np.float32)

    for d in range(n_targets):
        r2_vals[d] = r2_score(y_true[:, d], y_pred[:, d])
        pearson_vals[d] = _safe_pearson(y_true[:, d], y_pred[:, d])

    return r2_vals, pearson_vals


def _compute_prediction_metrics_per_trial(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n_trials = y_true.shape[0]
    r2_vals = np.zeros(n_trials, dtype=np.float32)
    pearson_vals = np.full(n_trials, np.nan, dtype=np.float32)

    for i in range(n_trials):
        r2_vals[i] = r2_score(y_true[i, :], y_pred[i, :])
        pearson_vals[i] = _safe_pearson(y_true[i, :], y_pred[i, :])

    return r2_vals, pearson_vals


def _summarize_session_evaluation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    r2_per_neuron: np.ndarray,
    pearson_per_neuron: np.ndarray,
    r2_per_trial: np.ndarray,
    pearson_per_trial: np.ndarray,
) -> dict[str, float]:
    flat_true = np.asarray(y_true).reshape(-1)
    flat_pred = np.asarray(y_pred).reshape(-1)
    finite_pearson = np.isfinite(pearson_per_neuron)
    finite_r2 = np.isfinite(r2_per_neuron)

    return {
        "y_true_std": float(np.std(flat_true)),
        "y_pred_std": float(np.std(flat_pred)),
        "r2_mean_per_neuron": _nanmean_or_nan(r2_per_neuron),
        "r2_std_per_neuron": _nanstd_or_nan(r2_per_neuron),
        "r2_min_per_neuron": _nanmin_or_nan(r2_per_neuron),
        "r2_max_per_neuron": _nanmax_or_nan(r2_per_neuron),
        "r2_valid_neuron_count": int(finite_r2.sum()),
        "pearson_mean_per_neuron": _nanmean_or_nan(pearson_per_neuron),
        "pearson_std_per_neuron": _nanstd_or_nan(pearson_per_neuron),
        "pearson_min_per_neuron": _nanmin_or_nan(pearson_per_neuron),
        "pearson_max_per_neuron": _nanmax_or_nan(pearson_per_neuron),
        "pearson_valid_neuron_count": int(finite_pearson.sum()),
        "r2_mean_per_trial": _nanmean_or_nan(r2_per_trial),
        "r2_std_per_trial": _nanstd_or_nan(r2_per_trial),
        "pearson_mean_per_trial": _nanmean_or_nan(pearson_per_trial),
        "pearson_std_per_trial": _nanstd_or_nan(pearson_per_trial),
    }


def _fit_xgb_regressor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    xgb_params: Dict,
) -> XGBRegressor:
    model = XGBRegressor(**xgb_params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    return model


def compute_oof_predictions_with_fe(
    X_raw: np.ndarray,
    y: np.ndarray,
    xgb_params: Dict,
    variance_threshold: float = 0.95,
    max_components: int = 30,
    fixed_components: Optional[int] = None,
    n_splits: int = 5,
    random_state: int = 42,
) -> np.ndarray:
    n_samples = X_raw.shape[0]
    n_splits_eff = min(n_splits, n_samples)
    if n_splits_eff < 2:
        raise ValueError(f"Too few samples for CV: {n_samples}")

    X_tensor = (
        torch.as_tensor(np.asarray(X_raw, dtype=np.float32))
        if torch is not None
        else np.asarray(X_raw, dtype=np.float32)
    )
    y = np.asarray(y, dtype=np.float32)
    y_pred_oof = np.zeros_like(y)
    kf = KFold(n_splits=n_splits_eff, shuffle=True, random_state=random_state)

    for fold_idx, (tr_idx, val_idx) in enumerate(kf.split(X_raw), start=1):
        print(f"[Fold {fold_idx}/{n_splits_eff}] train={len(tr_idx)}, val={len(val_idx)}")

        fe = ViTEmbeddingFeatureEngineer(
            variance_threshold=variance_threshold,
            max_components=max_components,
            fixed_components=fixed_components,
        )
        X_tr = fe.fit_transform(X_tensor[tr_idx])
        X_val = fe.transform(X_tensor[val_idx])
        y_tr = y[tr_idx]
        y_val = y[val_idx]

        for d in range(y.shape[1]):
            model = _fit_xgb_regressor(
                X_train=X_tr,
                y_train=y_tr[:, d],
                X_val=X_val,
                y_val=y_val[:, d],
                xgb_params=xgb_params,
            )
            y_pred_oof[val_idx, d] = model.predict(X_val)

    return y_pred_oof


def _format_stats_text(
    metric_name: str,
    values: np.ndarray,
) -> str:
    return "\n".join(
        [
            f"{metric_name}",
            f"mean = {_nanmean_or_nan(values):.4f}",
            f"std = {_nanstd_or_nan(values):.4f}",
            f"min = {_nanmin_or_nan(values):.4f}",
            f"max = {_nanmax_or_nan(values):.4f}",
        ]
    )


def _plot_metric_bars(
    r2_per_neuron: np.ndarray,
    pearson_per_neuron: np.ndarray,
    neuron_indices: np.ndarray,
    session_id: str,
    familiar_mode: str,
    deviant_mode: str,
    output_path: Path,
    figure_dpi: int = 180,
) -> None:
    plot_positions = np.arange(len(neuron_indices))
    r2_colors = np.where(r2_per_neuron >= 0, "#55A868", "#C44E52")
    pearson_colors = np.where(pearson_per_neuron >= 0, "#4C72B0", "#C44E52")

    width = max(12, min(36, len(neuron_indices) * 0.08))
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(width, 10),
        sharex=True,
        gridspec_kw={"hspace": 0.12},
    )

    ax_r2, ax_pearson = axes

    ax_r2.bar(plot_positions, r2_per_neuron, color=r2_colors, width=0.9)
    ax_r2.axhline(0.0, color="black", linewidth=0.8)
    ax_r2.set_title(
        "Neuron-wise Prediction Metrics: "
        f"{session_id}\n"
        f"familiar_mode={familiar_mode}, deviant_mode={deviant_mode}"
    )
    ax_r2.set_ylabel("R2")
    ax_r2.set_xlim(-0.5, len(plot_positions) - 0.5)
    ax_r2.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
    ax_r2.text(
        0.985,
        0.965,
        _format_stats_text("R2", r2_per_neuron),
        transform=ax_r2.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9, "edgecolor": "#999999"},
    )

    ax_pearson.bar(plot_positions, pearson_per_neuron, color=pearson_colors, width=0.9)
    ax_pearson.axhline(0.0, color="black", linewidth=0.8)
    ax_pearson.set_xlabel("Neuron Index")
    ax_pearson.set_ylabel("Pearson")
    ax_pearson.set_xlim(-0.5, len(plot_positions) - 0.5)
    ax_pearson.set_ylim(-1.0, 1.0)
    ax_pearson.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
    ax_pearson.set_xticks(plot_positions)
    ax_pearson.set_xticklabels(neuron_indices, rotation=90 if len(neuron_indices) > 20 else 0)
    ax_pearson.text(
        0.985,
        0.965,
        _format_stats_text("Pearson", pearson_per_neuron),
        transform=ax_pearson.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9, "edgecolor": "#999999"},
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=figure_dpi, bbox_inches="tight", format="pdf")
    plt.close(fig)


def _plot_prediction_vs_true_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    session_id: str,
    familiar_mode: str,
    deviant_mode: str,
    r2_per_neuron: np.ndarray,
    pearson_per_neuron: np.ndarray,
    output_path: Path,
    figure_dpi: int = 180,
) -> None:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if not valid.any():
        raise ValueError("No valid prediction/target pairs available for scatter plot.")

    true_valid = y_true[valid]
    pred_valid = y_pred[valid]

    min_val = float(np.nanmin([true_valid.min(), pred_valid.min()]))
    max_val = float(np.nanmax([true_valid.max(), pred_valid.max()]))
    pearson_mean = _nanmean_or_nan(pearson_per_neuron)
    pearson_std = _nanstd_or_nan(pearson_per_neuron)
    r2_mean = _nanmean_or_nan(r2_per_neuron)
    r2_std = _nanstd_or_nan(r2_per_neuron)

    plt.figure(figsize=(6.8, 6.8), dpi=figure_dpi)
    plt.scatter(
        true_valid,
        pred_valid,
        color=plt.cm.Blues(0.65),
        alpha=0.8,
        s=28,
        edgecolors="none",
    )

    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="#666666", linewidth=1)
    plt.xlabel("True response")
    plt.ylabel("Predicted response")
    plt.title(
        "Prediction vs True for Kept Neurons x Trials: "
        f"{session_id}\n"
        f"familiar_mode={familiar_mode}, deviant_mode={deviant_mode}"
    )
    plt.grid(alpha=0.25, linestyle="--")
    plt.text(
        0.03,
        0.97,
        f"Pearson = {pearson_mean:.4f} +- {pearson_std:.4f}\n"
        f"R2 = {r2_mean:.4f} +- {r2_std:.4f}",
        transform=plt.gca().transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9, "edgecolor": "#999999"},
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", format="pdf")
    plt.close()


def run_single_session_fe_neuron(
    config: Optional[FENeuronConfig] = None,
    xgb_params: Optional[Dict] = None,
    control_config: Optional[ControlConfig] = None,
):
    if config is None:
        config = FENeuronConfig()
    if xgb_params is None:
        xgb_params = default_xgb_params(random_state=config.random_state)

    if control_config is None:
        control_config = _build_control_config(config)
    else:
        control_config.validate()

        allowed_modes = {"none", "gaussian_trials", "scramble_pixels"}
        if control_config.familiar_mode not in allowed_modes:
            raise ValueError(
                f"Unsupported familiar_mode={control_config.familiar_mode!r} for single_session_neuron_plots. "
                f"Allowed modes: {sorted(allowed_modes)}"
            )
        if control_config.deviant_mode not in allowed_modes:
            raise ValueError(
                f"Unsupported deviant_mode={control_config.deviant_mode!r} for single_session_neuron_plots. "
                f"Allowed modes: {sorted(allowed_modes)}"
            )

    results_dir = Path(config.results_root) / config.pred / control_config.tag / config.session_id
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Session: {config.session_id} ===")
    X_raw, y, meta = prepare_data_with_control(
        session_id=config.session_id,
        pred=config.pred,
        pred_mean=config.pred_mean,
        need_sup_feat=config.need_sup_feat,
        control_config=control_config,
    )

    y_pred_oof = compute_oof_predictions_with_fe(
        X_raw=X_raw,
        y=y,
        xgb_params=xgb_params,
        variance_threshold=config.variance_threshold,
        max_components=config.max_components,
        fixed_components=config.fixed_embedding_components,
        n_splits=config.n_splits,
        random_state=config.random_state,
    )

    y_true_path = results_dir / f"{config.session_id}_y_true.npy"
    y_pred_path = results_dir / f"{config.session_id}_y_pred_oof.npy"
    np.save(y_true_path, y)
    np.save(y_pred_path, y_pred_oof)

    r2_per_neuron, pearson_per_neuron = _compute_prediction_metrics_per_dim(y, y_pred_oof)
    keep_mask, drop_indices = _select_neurons_by_pearson(
        pearson_per_neuron=pearson_per_neuron,
        keep_fraction=config.neuron_keep_fraction,
        min_neurons_to_keep=config.min_neurons_to_keep,
    )
    keep_indices = np.flatnonzero(keep_mask)
    session_eval = _summarize_filtered_session(
        y_true=y,
        y_pred=y_pred_oof,
        keep_mask=keep_mask,
        r2_per_neuron=r2_per_neuron,
        pearson_per_neuron=pearson_per_neuron,
    )

    neuron_indices = np.arange(y.shape[1], dtype=int)
    df_per_neuron = pd.DataFrame(
        {
            "session_id": config.session_id,
            "neuron_index": neuron_indices,
            "is_kept": keep_mask,
            "is_dropped": ~keep_mask,
            "r2_per_neuron": r2_per_neuron.astype(np.float64),
            "pearson_per_neuron": pearson_per_neuron.astype(np.float64),
        }
    )
    df_per_neuron["pearson_rank_desc"] = (
        df_per_neuron["pearson_per_neuron"]
        .rank(method="min", ascending=False, na_option="bottom")
        .astype(int)
    )

    evaluation_row = {
        "session_id": config.session_id,
        "ntrials_raw": int(meta["ntrials_raw"]),
        "ntrials_unique": int(meta["ntrials_unique"]),
        "input_feature_dim_raw": int(meta["input_feature_dim"]),
        "input_feature_dim_fe": int(config.fixed_embedding_components * 2),
        "target_dim": int(meta["target_dim"]),
        "kept_neuron_indices": json.dumps(keep_indices.tolist()),
        "dropped_neuron_indices": json.dumps(drop_indices.tolist()),
        **session_eval,
    }
    df_eval = pd.DataFrame([evaluation_row])

    per_neuron_path = results_dir / f"{config.session_id}_pearson_by_neuron.csv"
    evaluation_path = results_dir / f"{config.session_id}_evaluation.csv"
    barplot_path = results_dir / f"{config.session_id}_pearson_by_neuron_bar.pdf"
    scatterplot_path = results_dir / f"{config.session_id}_prediction_vs_true_scatter.pdf"
    metadata_path = results_dir / f"{config.session_id}_metadata.json"

    df_per_neuron.to_csv(per_neuron_path, index=False)
    df_eval.to_csv(evaluation_path, index=False)
    _plot_metric_bars(
        r2_per_neuron=r2_per_neuron[keep_mask],
        pearson_per_neuron=pearson_per_neuron[keep_mask],
        neuron_indices=keep_indices,
        session_id=config.session_id,
        familiar_mode=control_config.familiar_mode,
        deviant_mode=control_config.deviant_mode,
        output_path=barplot_path,
        figure_dpi=config.figure_dpi,
    )
    _plot_prediction_vs_true_scatter(
        y_true=y[:, keep_mask],
        y_pred=y_pred_oof[:, keep_mask],
        session_id=config.session_id,
        familiar_mode=control_config.familiar_mode,
        deviant_mode=control_config.deviant_mode,
        r2_per_neuron=r2_per_neuron[keep_mask],
        pearson_per_neuron=pearson_per_neuron[keep_mask],
        output_path=scatterplot_path,
        figure_dpi=config.figure_dpi,
    )

    top_k = min(10, len(df_per_neuron))
    top_neurons = (
        df_per_neuron.sort_values("pearson_per_neuron", ascending=False)
        .head(top_k)[["neuron_index", "pearson_per_neuron", "r2_per_neuron"]]
        .to_dict(orient="records")
    )
    bottom_neurons = (
        df_per_neuron.sort_values("pearson_per_neuron", ascending=True)
        .head(top_k)[["neuron_index", "pearson_per_neuron", "r2_per_neuron"]]
        .to_dict(orient="records")
    )

    metadata = {
        "config": asdict(config),
        "xgb_params": xgb_params,
        "control_config": asdict(control_config),
        "session_summary": evaluation_row,
        "kept_neuron_indices": keep_indices.tolist(),
        "dropped_neuron_indices": drop_indices.tolist(),
        "top_10_neurons_by_pearson": top_neurons,
        "bottom_10_neurons_by_pearson": bottom_neurons,
        "files": {
            "evaluation_csv": str(evaluation_path),
            "per_neuron_csv": str(per_neuron_path),
            "pearson_bar_plot_pdf": str(barplot_path),
            "prediction_vs_true_scatter_pdf": str(scatterplot_path),
            "y_true_npy": str(y_true_path),
            "y_pred_oof_npy": str(y_pred_path),
        },
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(
        f"  [Filter] dropped={len(drop_indices)}, kept={len(keep_indices)}, "
        f"r2_mean_kept={session_eval['r2_mean_kept']:.4f}, "
        f"r2_std_kept={session_eval['r2_std_kept']:.4f}, "
        f"pearson_mean_kept={session_eval['pearson_mean_kept']:.4f}, "
        f"pearson_std_kept={session_eval['pearson_std_kept']:.4f}"
    )
    print("\nSaved neuron_filtering outputs:")
    print(f"  {evaluation_path}")
    print(f"  {per_neuron_path}")
    print(f"  {barplot_path}")
    print(f"  {scatterplot_path}")
    print(f"  {metadata_path}")

    return {
        "df_evaluation": df_eval,
        "df_per_neuron": df_per_neuron,
        "metadata": metadata,
        "paths": {
            "evaluation_csv": evaluation_path,
            "per_neuron_csv": per_neuron_path,
            "pearson_bar_plot_pdf": barplot_path,
            "prediction_vs_true_scatter_pdf": scatterplot_path,
            "metadata_json": metadata_path,
            "y_true_npy": y_true_path,
            "y_pred_oof_npy": y_pred_path,
        },
    }


if __name__ == "__main__":
    config = FENeuronConfig(
        session_id="session_1053925378",
        pred="burst_B",
        pred_mean=False,
        need_sup_feat=False,
        n_splits=5,
        random_state=42,
        results_root="./results/fe_neuron_plot",
        variance_threshold=0.95,
        max_components=30,
        fixed_embedding_components=6,
        familiar_mode="none",  # "none", "gaussian_trials", "scramble_pixels"
        deviant_mode="scramble_pixels",  # "none", "gaussian_trials", "scramble_pixels"
        neuron_keep_fraction=0.2,
        min_neurons_to_keep=3,
        figure_dpi=180,
    )

    control_config = _build_control_config(config)

    run_single_session_fe_neuron(
        config=config,
        xgb_params=default_xgb_params(random_state=config.random_state),
        control_config=control_config,
    )
