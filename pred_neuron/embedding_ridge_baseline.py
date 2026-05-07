import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from importance_control import SESSION_IDS, ControlConfig, prepare_data_with_control

try:
    import torch
except ModuleNotFoundError:
    torch = None


class ViTEmbeddingFeatureEngineer:
    """
    Input: N x 1536 torch tensor with concatenated A/B embeddings.
    Output: N x D NumPy array.
    Usage: fit_transform on train folds, transform on validation folds.
    """

    def __init__(self, variance_threshold=0.95, max_components=30):
        self.variance_threshold = variance_threshold
        self.max_components = max_components
        self.pca_diff = None
        self.pca_had = None
        self.scaler = StandardScaler()

    def _split_and_compute(self, X):
        """Split 1536-dim inputs into A/B blocks and compute interaction features."""
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

        norm_a = np.linalg.norm(A, axis=1, keepdims=True)
        norm_b = np.linalg.norm(B, axis=1, keepdims=True)
        norm_diff = norm_a - norm_b

        cosine = (A * B).sum(axis=1, keepdims=True) / (norm_a * norm_b + 1e-8)

        var_a = A.var(axis=1, keepdims=True)
        var_b = B.var(axis=1, keepdims=True)

        scalars = np.concatenate(
            [norm_a, norm_b, norm_diff, cosine, var_a, var_b],
            axis=1,
        )

        return diff, hadamard, scalars

    def _n_components(self, X_train):
        """Choose the PCA dimension from the training data without exceeding max_components."""
        max_valid = min(X_train.shape[0] - 1, X_train.shape[1], self.max_components)
        if max_valid <= 0:
            raise ValueError(
                f"Not enough training samples for PCA: got shape {X_train.shape}"
            )

        pca_tmp = PCA(n_components=max_valid).fit(X_train)
        cumvar = np.cumsum(pca_tmp.explained_variance_ratio_)
        n = int(np.argmax(cumvar >= self.variance_threshold) + 1)
        return min(max(n, 1), max_valid)

    def fit_transform(self, X) -> np.ndarray:
        """Fit on a training fold and transform it."""
        diff, hadamard, _ = self._split_and_compute(X)

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
            f"hadamard PCA: {n_had} dims, "
            f"scalars: 0 dims -> total {features.shape[1]} dims"
        )
        return features.astype(np.float32)

    def transform(self, X) -> np.ndarray:
        """Transform validation/test folds using PCA fitted on the training fold."""
        assert self.pca_diff is not None, "Call fit_transform first."
        diff, hadamard, _ = self._split_and_compute(X)

        diff_r = self.pca_diff.transform(diff)
        had_r = self.pca_had.transform(hadamard)

        features = np.concatenate([diff_r, had_r], axis=1)
        features = self.scaler.transform(features)
        return features.astype(np.float32)


@dataclass
class FERunConfig:
    pred: str = "burst_B"
    pred_mean: bool = False
    need_sup_feat: bool = False
    n_splits: int = 5
    random_state: int = 42
    results_root: str = "./results/fe_ridge"
    variance_threshold: float = 0.95
    max_components: int = 30
    familiar_mode: str = "none"
    deviant_mode: str = "none"


def default_ridge_params() -> Dict:
    return {
        "alphas": np.logspace(0, 4, 50),
        "cv": 5,
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
    return {
        "y_true_std": float(np.std(flat_true)),
        "y_pred_std": float(np.std(flat_pred)),
        "r2_mean_per_neuron": _nanmean_or_nan(r2_per_neuron),
        "r2_std_per_neuron": _nanstd_or_nan(r2_per_neuron),
        "pearson_mean_per_neuron": _nanmean_or_nan(pearson_per_neuron),
        "pearson_std_per_neuron": _nanstd_or_nan(pearson_per_neuron),
        "r2_mean_per_trial": _nanmean_or_nan(r2_per_trial),
        "r2_std_per_trial": _nanstd_or_nan(r2_per_trial),
        "pearson_mean_per_trial": _nanmean_or_nan(pearson_per_trial),
        "pearson_std_per_trial": _nanstd_or_nan(pearson_per_trial),
    }


def _fit_ridge_regressor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    ridge_params: Dict,
) -> RidgeCV:
    model_params = dict(ridge_params)
    requested_cv = int(model_params.get("cv", 5))
    model_params["cv"] = min(requested_cv, X_train.shape[0])
    model = RidgeCV(**model_params)
    model.fit(X_train, y_train)
    return model


def compute_oof_predictions_with_fe(
    X_raw: np.ndarray,
    y: np.ndarray,
    ridge_params: Dict,
    variance_threshold: float = 0.95,
    max_components: int = 30,
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
        )
        X_tr = fe.fit_transform(X_tensor[tr_idx])
        X_val = fe.transform(X_tensor[val_idx])
        y_tr = y[tr_idx]

        for d in range(y.shape[1]):
            model = _fit_ridge_regressor(
                X_train=X_tr,
                y_train=y_tr[:, d],
                ridge_params=ridge_params,
            )
            y_pred_oof[val_idx, d] = model.predict(X_val)

    return y_pred_oof


def run_fe_ridge_pipeline(
    session_ids,
    config: Optional[FERunConfig] = None,
    ridge_params: Optional[Dict] = None,
    control_config: Optional[ControlConfig] = None,
):
    if config is None:
        config = FERunConfig()
    if ridge_params is None:
        ridge_params = default_ridge_params()

    if control_config is None:
        control_config = ControlConfig(
            familiar_mode=config.familiar_mode,
            deviant_mode=config.deviant_mode,
            random_state=config.random_state,
        )
    control_config.validate()

    allowed_modes = {"none", "gaussian_trials"}
    if control_config.familiar_mode not in allowed_modes:
        raise ValueError(
            f"Unsupported familiar_mode={control_config.familiar_mode!r} for embedding_ridge_baseline. "
            f"Allowed modes: {sorted(allowed_modes)}"
        )
    if control_config.deviant_mode not in allowed_modes:
        raise ValueError(
            f"Unsupported deviant_mode={control_config.deviant_mode!r} for embedding_ridge_baseline. "
            f"Allowed modes: {sorted(allowed_modes)}"
        )

    results_dir = Path(config.results_root) / config.pred / control_config.tag
    results_dir.mkdir(parents=True, exist_ok=True)

    evaluation_rows = []
    per_neuron_rows = []
    prediction_dir = results_dir / "oof_predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)

    for session_id in session_ids:
        print(f"\n=== Session: {session_id} ===")
        X_raw, y, meta = prepare_data_with_control(
            session_id=session_id,
            pred=config.pred,
            pred_mean=config.pred_mean,
            need_sup_feat=config.need_sup_feat,
            control_config=control_config,
        )

        y_pred_oof = compute_oof_predictions_with_fe(
            X_raw=X_raw,
            y=y,
            ridge_params=ridge_params,
            variance_threshold=config.variance_threshold,
            max_components=config.max_components,
            n_splits=config.n_splits,
            random_state=config.random_state,
        )

        np.save(prediction_dir / f"{session_id}_y_true.npy", y)
        np.save(prediction_dir / f"{session_id}_y_pred_oof.npy", y_pred_oof)

        r2_per_neuron, pearson_per_neuron = _compute_prediction_metrics_per_dim(y, y_pred_oof)
        r2_per_trial, pearson_per_trial = _compute_prediction_metrics_per_trial(y, y_pred_oof)
        session_eval = _summarize_session_evaluation(
            y_true=y,
            y_pred=y_pred_oof,
            r2_per_neuron=r2_per_neuron,
            pearson_per_neuron=pearson_per_neuron,
            r2_per_trial=r2_per_trial,
            pearson_per_trial=pearson_per_trial,
        )
        evaluation_rows.append(
            {
                "session_id": session_id,
                "ntrials_raw": int(meta["ntrials_raw"]),
                "ntrials_unique": int(meta["ntrials_unique"]),
                "input_feature_dim_raw": int(meta["input_feature_dim"]),
                "target_dim": int(meta["target_dim"]),
                **session_eval,
            }
        )

        for d in range(y.shape[1]):
            per_neuron_rows.append(
                {
                    "session_id": session_id,
                    "neuron": d,
                    "r2_per_neuron": float(r2_per_neuron[d]),
                    "pearson_per_neuron": float(pearson_per_neuron[d]),
                }
            )

        print(
            f"  [Eval] y_true_std={session_eval['y_true_std']:.4f}, "
            f"y_pred_std={session_eval['y_pred_std']:.4f}, "
            f"r2_mean_per_neuron={session_eval['r2_mean_per_neuron']:.4f}, "
            f"r2_std_per_neuron={session_eval['r2_std_per_neuron']:.4f}, "
            f"pearson_mean_per_neuron={session_eval['pearson_mean_per_neuron']:.4f}, "
            f"pearson_std_per_neuron={session_eval['pearson_std_per_neuron']:.4f}, "
            f"r2_mean_per_trial={session_eval['r2_mean_per_trial']:.4f}, "
            f"r2_std_per_trial={session_eval['r2_std_per_trial']:.4f}, "
            f"pearson_mean_per_trial={session_eval['pearson_mean_per_trial']:.4f}, "
            f"pearson_std_per_trial={session_eval['pearson_std_per_trial']:.4f}"
        )

    df_eval = pd.DataFrame(evaluation_rows)
    df_per_dim = pd.DataFrame(per_neuron_rows)

    tag = f"{config.pred}_{'mean' if config.pred_mean else 'full'}_fe_ridge"
    eval_path = results_dir / f"evaluation_{tag}.csv"
    per_dim_path = results_dir / f"per_dim_{tag}.csv"
    metadata_path = results_dir / f"metadata_{tag}.json"

    df_eval.to_csv(eval_path, index=False)
    df_per_dim.to_csv(per_dim_path, index=False)

    metadata = {
        "config": asdict(config),
        "session_ids": list(session_ids),
        "ridge_params": {
            **ridge_params,
            "alphas": np.asarray(ridge_params["alphas"]).tolist(),
        },
        "control_config": asdict(control_config),
        "files": {
            "evaluation_csv": str(eval_path),
            "per_dim_csv": str(per_dim_path),
            "oof_prediction_dir": str(prediction_dir),
        },
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("\nSaved embedding_ridge_baseline outputs:")
    print(f"  {eval_path}")
    print(f"  {per_dim_path}")
    print(f"  {metadata_path}")

    return {
        "df_evaluation": df_eval,
        "df_per_dim": df_per_dim,
        "metadata": metadata,
        "paths": {
            "evaluation_csv": eval_path,
            "per_dim_csv": per_dim_path,
            "metadata_json": metadata_path,
            "oof_prediction_dir": prediction_dir,
        },
    }


if __name__ == "__main__":
    config = FERunConfig(
        pred="burst_B",
        pred_mean=False,
        need_sup_feat=False,
        n_splits=5,
        random_state=42,
        results_root="./results/fe_ridge",
        variance_threshold=0.95,
        max_components=30,
        familiar_mode="gaussian_trials",  # none | gaussian_trials
        deviant_mode="gaussian_trials",  # none | gaussian_trials
    )

    control_config = ControlConfig(
        familiar_mode=config.familiar_mode,
        deviant_mode=config.deviant_mode,
        random_state=config.random_state,
    )

    run_fe_ridge_pipeline(
        session_ids=SESSION_IDS[:2],
        config=config,
        ridge_params=default_ridge_params(),
        control_config=control_config,
    )
