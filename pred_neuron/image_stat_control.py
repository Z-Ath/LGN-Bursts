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
import shap
from xgboost import XGBRegressor

from embedding_xgboost_baseline import (
    _compute_prediction_metrics_per_dim,
    _compute_prediction_metrics_per_trial,
    _nanmean_or_nan,
    _nanstd_or_nan,
    compute_oof_predictions_with_fe,
    default_xgb_params,
)
from importance_control import (
    SESSION_IDS,
    _aggregate_support_features,
    _aggregate_trial_targets,
    _build_original_embeddings,
    _default_color_limits_for_metric,
    _load_trial_structure,
    _stable_seed,
    _validate_plot_metric,
    build_vit_embeddings_from_X,
    load_image,
    plot_familiar_vs_deviant,
)


VALID_STAT_MODES = {"none", "brightness", "contrast"}
BRIGHTNESS_FACTOR = 2.0
CONTRAST_FACTOR = 0.2
DEFAULT_EMB_CACHE_ROOT = "./data/embeddings/fe_stat_control"
_STAT_EMBED_CACHE: dict[str, Dict[str, np.ndarray]] = {}


@dataclass(frozen=True)
class StatControlConfig:
    familiar_mode: str = "none"
    deviant_mode: str = "none"
    random_state: int = 42

    def validate(self) -> None:
        for role, mode in (
            ("familiar", self.familiar_mode),
            ("deviant", self.deviant_mode),
        ):
            if mode not in VALID_STAT_MODES:
                raise ValueError(
                    f"Unsupported {role}_mode={mode}. Expected one of {sorted(VALID_STAT_MODES)}."
                )

    @property
    def tag(self) -> str:
        self.validate()
        return (
            f"fam-{self.familiar_mode}_"
            f"dev-{self.deviant_mode}_"
            f"seed{self.random_state}"
        )

    def describe(self) -> str:
        self.validate()
        return (
            f"familiar={self.familiar_mode}, "
            f"deviant={self.deviant_mode}, "
            f"seed={self.random_state}"
        )


@dataclass
class FENeuronStatConfig:
    pred: str = "burst_B"
    pred_mean: bool = False
    need_sup_feat: bool = False
    n_splits: int = 5
    random_state: int = 42
    results_root: str = "./results/fe_neuron_stat"
    variance_threshold: float = 0.95
    max_components: int = 30
    fixed_embedding_components: int = 6
    familiar_mode: str = "brightness"
    deviant_mode: str = "brightness"
    top_k_sessions: int = 10
    neuron_keep_fraction: float = 0.2
    min_neurons_to_keep: int = 3
    figure_dpi: int = 180
    plot_metric: str = "pearson"
    save_shap_values: bool = False


def _build_control_config(config: FENeuronStatConfig) -> StatControlConfig:
    control_config = StatControlConfig(
        familiar_mode=config.familiar_mode,
        deviant_mode=config.deviant_mode,
        random_state=config.random_state,
    )
    control_config.validate()
    return control_config


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
        "r2_mean_kept": _nanmean_or_nan(r2_per_neuron[keep_mask]),
        "r2_std_kept": _nanstd_or_nan(r2_per_neuron[keep_mask]),
        "pearson_mean_per_trial_kept": _nanmean_or_nan(pearson_trial_kept),
        "pearson_std_per_trial_kept": _nanstd_or_nan(pearson_trial_kept),
        "r2_mean_per_trial_kept": _nanmean_or_nan(r2_trial_kept),
        "r2_std_per_trial_kept": _nanstd_or_nan(r2_trial_kept),
    }


def _plot_top_sessions(
    df_eval: pd.DataFrame,
    output_path: Path,
    title: str,
    figure_dpi: int = 180,
) -> None:
    df_plot = df_eval.sort_values("pearson_mean_kept", ascending=False).reset_index(drop=True)
    r2_session_mean = _nanmean_or_nan(df_plot["r2_mean_kept"].to_numpy(dtype=np.float64))
    r2_session_std = _nanstd_or_nan(df_plot["r2_mean_kept"].to_numpy(dtype=np.float64))
    pearson_session_mean = _nanmean_or_nan(df_plot["pearson_mean_kept"].to_numpy(dtype=np.float64))
    pearson_session_std = _nanstd_or_nan(df_plot["pearson_mean_kept"].to_numpy(dtype=np.float64))
    x = np.arange(len(df_plot))
    labels = [
        f"{sid}\n(orig n={n_orig}, kept n={n_kept})"
        for sid, n_orig, n_kept in zip(
            df_plot["session_id"],
            df_plot["target_dim"],
            df_plot["n_neurons_kept"],
        )
    ]

    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True, gridspec_kw={"hspace": 0.18})

    axes[0].bar(
        x,
        df_plot["r2_mean_kept"],
        yerr=df_plot["r2_std_kept"],
        color="#d62728",
        edgecolor="black",
        linewidth=0.8,
        capsize=4,
        alpha=0.9,
    )
    axes[0].set_ylabel("R2 mean +/- std")
    axes[0].set_title(title)
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.4)
    axes[0].text(
        0.97,
        0.97,
        f"R2 across sessions\nmean = {r2_session_mean:.4f}\nstd = {r2_session_std:.4f}",
        transform=axes[0].transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9, "edgecolor": "#999999"},
    )

    axes[1].bar(
        x,
        df_plot["pearson_mean_kept"],
        yerr=df_plot["pearson_std_kept"],
        color="#1f77b4",
        edgecolor="black",
        linewidth=0.8,
        capsize=4,
        alpha=0.9,
    )
    axes[1].set_ylabel("Pearson mean +/- std")
    axes[1].set_xlabel("Session")
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.4)
    axes[1].text(
        0.97,
        0.97,
        f"Pearson across sessions\nmean = {pearson_session_mean:.4f}\nstd = {pearson_session_std:.4f}",
        transform=axes[1].transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9, "edgecolor": "#999999"},
    )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=30, ha="right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=figure_dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_filtered_shap_scatter(
    df_shap: pd.DataFrame,
    output_path: Path,
    metric: str,
    title: str,
    figure_dpi: int = 180,
) -> None:
    metric = _validate_plot_metric(metric)
    if df_shap.empty:
        raise ValueError("No SHAP rows available for scatter plot.")
    if metric not in df_shap.columns:
        raise KeyError(f"Column '{metric}' not found in SHAP dataframe.")

    x = df_shap["familiar_importance"].to_numpy(dtype=np.float64)
    y = df_shap["deviant_importance"].to_numpy(dtype=np.float64)
    colors = df_shap[metric].to_numpy(dtype=np.float64)
    valid = np.isfinite(colors)
    limits = _default_color_limits_for_metric(metric)

    plt.figure(figsize=(6.8, 6.8), dpi=figure_dpi)
    if valid.any():
        sc = plt.scatter(
            x[valid],
            y[valid],
            c=colors[valid],
            cmap="Blues",
            alpha=0.8,
            s=28,
            edgecolors="none",
            vmin=float(limits[0]),
            vmax=float(limits[1]),
        )
        cbar = plt.colorbar(sc)
        cbar.set_label("R2 per dim" if metric == "r2" else "Pearson per dim")
    if (~valid).any():
        plt.scatter(x[~valid], y[~valid], alpha=0.8, s=28, edgecolors="none", color="#bdbdbd")

    max_val = float(np.nanmax([x.max(), y.max()]))
    plt.plot([0, max_val], [0, max_val], linestyle="--", color="#666666", linewidth=1)
    plt.xlabel("Familiar importance (sum |SHAP|)")
    plt.ylabel("Deviant importance (sum |SHAP|)")
    plt.title(title)
    plt.grid(alpha=0.25, linestyle="--")

    metric_mean = _nanmean_or_nan(colors)
    metric_std = _nanstd_or_nan(colors)
    metric_label = "R2" if metric == "r2" else "Pearson"
    plt.text(
        0.03,
        0.97,
        f"{metric_label}\nmean = {metric_mean:.4f}\nstd = {metric_std:.4f}",
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


def _normalize_image_range(img: np.ndarray) -> tuple[np.ndarray, float]:
    arr = np.asarray(img, dtype=np.float32)
    if np.issubdtype(np.asarray(img).dtype, np.integer):
        max_value = float(np.iinfo(np.asarray(img).dtype).max)
    else:
        max_value = float(np.nanmax(arr))
        if not np.isfinite(max_value) or max_value <= 0:
            max_value = 1.0
        max_value = max(max_value, 1.0)
    return arr, max_value


def _increase_brightness(img: np.ndarray, factor: float = BRIGHTNESS_FACTOR) -> np.ndarray:
    arr, max_value = _normalize_image_range(img)
    adjusted = np.clip(arr * factor, 0.0, max_value)
    return adjusted.astype(np.asarray(img).dtype, copy=False)


def _decrease_contrast(img: np.ndarray, factor: float = CONTRAST_FACTOR) -> np.ndarray:
    arr, max_value = _normalize_image_range(img)
    mean_val = float(arr.mean())
    adjusted = np.clip((arr - mean_val) * factor + mean_val, 0.0, max_value)
    return adjusted.astype(np.asarray(img).dtype, copy=False)


def _transform_image_for_mode(img: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return np.asarray(img).copy()
    if mode == "brightness":
        return _increase_brightness(img)
    if mode == "contrast":
        return _decrease_contrast(img)
    raise ValueError(f"Unsupported image transform mode: {mode}")


def _build_adjusted_embeddings_for_role(
    role: str,
    mode: str,
    control_config: StatControlConfig,
    cache_root: str = DEFAULT_EMB_CACHE_ROOT,
) -> Dict[str, np.ndarray]:
    cache_root_path = Path(cache_root)
    cache_root_path.mkdir(parents=True, exist_ok=True)
    cache_path = cache_root_path / f"vit_b16_{role}_{mode}_{control_config.tag}.npz"
    cache_key = f"{role}::{mode}::{cache_path}"

    if cache_key in _STAT_EMBED_CACHE:
        return _STAT_EMBED_CACHE[cache_key]

    if cache_path.exists():
        with np.load(cache_path) as npz:
            emb = {k: npz[k] for k in npz.files}
        _STAT_EMBED_CACHE[cache_key] = emb
        return emb

    X_imgs, y_labels = load_image()
    X_adjusted = np.empty_like(X_imgs)
    for idx, _label in enumerate(y_labels):
        X_adjusted[idx] = _transform_image_for_mode(X_imgs[idx], mode)

    emb = build_vit_embeddings_from_X(
        X_adjusted,
        y_labels,
        cache_path=str(cache_path),
    )
    _STAT_EMBED_CACHE[cache_key] = emb
    return emb


def _build_role_to_embedding(
    control_config: StatControlConfig,
    cache_root: str = DEFAULT_EMB_CACHE_ROOT,
) -> dict[str, Dict[str, np.ndarray]]:
    control_config.validate()
    original = _build_original_embeddings()

    role_to_emb = {
        "familiar": original,
        "deviant": original,
    }

    if control_config.familiar_mode in {"brightness", "contrast"}:
        role_to_emb["familiar"] = _build_adjusted_embeddings_for_role(
            role="familiar",
            mode=control_config.familiar_mode,
            control_config=control_config,
            cache_root=cache_root,
        )

    if control_config.deviant_mode in {"brightness", "contrast"}:
        role_to_emb["deviant"] = _build_adjusted_embeddings_for_role(
            role="deviant",
            mode=control_config.deviant_mode,
            control_config=control_config,
            cache_root=cache_root,
        )

    return role_to_emb


def prepare_data_with_stat_control(
    session_id: str,
    pred: str = "tonic_A",
    pred_mean: bool = False,
    need_sup_feat: bool = False,
    control_config: Optional[StatControlConfig] = None,
    emb_cache_root: str = DEFAULT_EMB_CACHE_ROOT,
) -> tuple[np.ndarray, np.ndarray, Dict]:
    if control_config is None:
        control_config = StatControlConfig()
    control_config.validate()

    unique_pairs, duplicate_groups, ntrials_raw = _load_trial_structure(session_id=session_id)
    y = _aggregate_trial_targets(
        session_id=session_id,
        pred=pred,
        pred_mean=pred_mean,
        duplicate_groups=duplicate_groups,
        expected_ntrials=ntrials_raw,
    )

    sup_unique = None
    sup_feat_names = None
    if need_sup_feat:
        sup_unique, sup_feat_names = _aggregate_support_features(
            session_id=session_id,
            pred=pred,
            duplicate_groups=duplicate_groups,
            expected_ntrials=ntrials_raw,
        )

    role_to_emb = _build_role_to_embedding(
        control_config=control_config,
        cache_root=emb_cache_root,
    )
    example_emb = next(iter(role_to_emb["familiar"].values()))
    d_img = int(example_emb.shape[0])

    img_emb_unique = np.zeros((unique_pairs.shape[0], d_img * 2), dtype=example_emb.dtype)

    for k, (img_familiar, img_deviant) in enumerate(unique_pairs):
        if img_familiar not in role_to_emb["familiar"]:
            raise KeyError(f"Image label '{img_familiar}' not found in familiar embedding dict.")
        if img_deviant not in role_to_emb["deviant"]:
            raise KeyError(f"Image label '{img_deviant}' not found in deviant embedding dict.")

        familiar_emb = role_to_emb["familiar"][img_familiar]
        deviant_emb = role_to_emb["deviant"][img_deviant]
        img_emb_unique[k] = np.concatenate([familiar_emb, deviant_emb], axis=0)

    if need_sup_feat and sup_unique is not None:
        X = np.concatenate([img_emb_unique, sup_unique], axis=1)
    else:
        X = img_emb_unique

    meta = {
        "session_id": session_id,
        "sup_feat_names": sup_feat_names,
        "pred": pred,
        "pred_mean": pred_mean,
        "need_sup_feat": need_sup_feat,
        "ntrials_raw": int(ntrials_raw),
        "ntrials_unique": int(unique_pairs.shape[0]),
        "img_feature_dim": d_img,
        "input_feature_dim": int(X.shape[1]),
        "target_dim": int(y.shape[1]),
        "control_config": asdict(control_config),
        "image_stat_meta": {
            "familiar_mode": control_config.familiar_mode,
            "deviant_mode": control_config.deviant_mode,
            "brightness_factor": BRIGHTNESS_FACTOR,
            "contrast_factor": CONTRAST_FACTOR,
            "transform_seed_reference": _stable_seed(control_config.random_state, session_id, "stat"),
        },
    }

    print(
        f"Prepared stat-controlled data for session {session_id} / pred={pred} / "
        f"pred_mean={pred_mean} / sup_feat={need_sup_feat}"
    )
    print(f"  control: {control_config.describe()}")
    print(f"  raw trials: {meta['ntrials_raw']}, unique trials: {meta['ntrials_unique']}")
    print(f"  X shape: {X.shape}, y shape: {y.shape}")

    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32), meta


def _shap_fit_params(xgb_params: Dict) -> Dict:
    params = dict(xgb_params)
    params.pop("early_stopping_rounds", None)
    return params


def _compute_filtered_shap_rows(
    selected_session_infos: list[dict],
    xgb_params: Dict,
    shap_dir: Optional[Path] = None,
) -> list[dict]:
    per_dim_rows = []
    shap_params = _shap_fit_params(xgb_params)

    for session_info in selected_session_infos:
        session_id = session_info["session_id"]
        X_raw = session_info["X_raw"]
        y = session_info["y"]
        meta = session_info["meta"]
        keep_indices = session_info["keep_indices"]
        d_img = int(meta["img_feature_dim"])

        print(f"\n=== SHAP for session: {session_id} / kept_neurons={len(keep_indices)} ===")
        for neuron_idx in keep_indices:
            model = XGBRegressor(**shap_params)
            model.fit(X_raw, y[:, neuron_idx], verbose=False)

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_raw)
            mean_abs = np.mean(np.abs(shap_values), axis=0)

            familiar_imp = float(mean_abs[:d_img].sum())
            deviant_imp = float(mean_abs[d_img : 2 * d_img].sum())
            support_imp = float(mean_abs[2 * d_img :].sum()) if mean_abs.shape[0] > 2 * d_img else 0.0

            per_dim_rows.append(
                {
                    "session_id": session_id,
                    "neuron_index": int(neuron_idx),
                    "familiar_importance": familiar_imp,
                    "deviant_importance": deviant_imp,
                    "support_importance": support_imp,
                    "r2": float(session_info["r2_per_neuron"][neuron_idx]),
                    "pearson": float(session_info["pearson_per_neuron"][neuron_idx]),
                }
            )

            if shap_dir is not None:
                np.save(shap_dir / f"shap_values_{session_id}_neuron{neuron_idx}.npy", shap_values)

    return per_dim_rows


def run_fe_neuron_stat_pipeline(
    session_ids=SESSION_IDS,
    config: Optional[FENeuronStatConfig] = None,
    xgb_params: Optional[Dict] = None,
):
    if config is None:
        config = FENeuronStatConfig()
    if xgb_params is None:
        xgb_params = default_xgb_params(random_state=config.random_state)

    control_config = _build_control_config(config)
    results_dir = Path(config.results_root) / config.pred / control_config.tag
    results_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir = results_dir / "oof_predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    shap_dir = results_dir / "shap_values_top_sessions" if config.save_shap_values else None
    if shap_dir is not None:
        shap_dir.mkdir(parents=True, exist_ok=True)

    all_session_rows = []
    all_neuron_rows = []
    session_infos = []

    print("\n=== Running FE + neuron filter on all sessions ===")
    for session_id in session_ids:
        print(f"\n=== Session: {session_id} ===")
        X_raw, y, meta = prepare_data_with_stat_control(
            session_id=session_id,
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

        np.save(prediction_dir / f"{session_id}_y_true.npy", y)
        np.save(prediction_dir / f"{session_id}_y_pred_oof.npy", y_pred_oof)

        r2_per_neuron, pearson_per_neuron = _compute_prediction_metrics_per_dim(y, y_pred_oof)
        keep_mask, drop_indices = _select_neurons_by_pearson(
            pearson_per_neuron=pearson_per_neuron,
            keep_fraction=config.neuron_keep_fraction,
            min_neurons_to_keep=config.min_neurons_to_keep,
        )
        session_eval = _summarize_filtered_session(
            y_true=y,
            y_pred=y_pred_oof,
            keep_mask=keep_mask,
            r2_per_neuron=r2_per_neuron,
            pearson_per_neuron=pearson_per_neuron,
        )

        session_row = {
            "session_id": session_id,
            "ntrials_raw": int(meta["ntrials_raw"]),
            "ntrials_unique": int(meta["ntrials_unique"]),
            "img_feature_dim": int(meta["img_feature_dim"]),
            "input_feature_dim_raw": int(meta["input_feature_dim"]),
            "target_dim": int(meta["target_dim"]),
            "dropped_neuron_indices": json.dumps(drop_indices.tolist()),
            "kept_neuron_indices": json.dumps(np.flatnonzero(keep_mask).tolist()),
            **session_eval,
        }
        all_session_rows.append(session_row)

        for neuron_idx in range(y.shape[1]):
            all_neuron_rows.append(
                {
                    "session_id": session_id,
                    "neuron_index": int(neuron_idx),
                    "is_kept": bool(keep_mask[neuron_idx]),
                    "is_dropped": bool(not keep_mask[neuron_idx]),
                    "r2_per_neuron": float(r2_per_neuron[neuron_idx]),
                    "pearson_per_neuron": float(pearson_per_neuron[neuron_idx]),
                }
            )

        session_infos.append(
            {
                "session_id": session_id,
                "X_raw": X_raw,
                "y": y,
                "meta": meta,
                "keep_mask": keep_mask,
                "keep_indices": np.flatnonzero(keep_mask),
                "drop_indices": drop_indices,
                "r2_per_neuron": r2_per_neuron,
                "pearson_per_neuron": pearson_per_neuron,
                "session_row": session_row,
            }
        )

        print(
            f"  [Filter] dropped={len(drop_indices)}, kept={keep_mask.sum()}, "
            f"pearson_mean_kept={session_eval['pearson_mean_kept']:.4f}, "
            f"r2_mean_kept={session_eval['r2_mean_kept']:.4f}"
        )

    df_all_sessions = pd.DataFrame(all_session_rows).sort_values(
        "pearson_mean_kept",
        ascending=False,
    ).reset_index(drop=True)
    df_all_neurons = pd.DataFrame(all_neuron_rows)

    selected_session_ids = df_all_sessions["session_id"].head(config.top_k_sessions).tolist()
    selected_session_infos = [info for info in session_infos if info["session_id"] in selected_session_ids]
    selected_session_infos = sorted(
        selected_session_infos,
        key=lambda row: row["session_row"]["pearson_mean_kept"],
        reverse=True,
    )
    df_top_sessions = df_all_sessions[df_all_sessions["session_id"].isin(selected_session_ids)].copy()
    df_top_sessions = df_top_sessions.sort_values("pearson_mean_kept", ascending=False).reset_index(drop=True)
    df_top_neurons = df_all_neurons[df_all_neurons["session_id"].isin(selected_session_ids)].copy()

    print("\nSelected top sessions by filtered Pearson:")
    print(selected_session_ids)

    per_dim_rows = _compute_filtered_shap_rows(
        selected_session_infos=selected_session_infos,
        xgb_params=xgb_params,
        shap_dir=shap_dir,
    )
    df_shap = pd.DataFrame(per_dim_rows)

    session_tag = f"{config.pred}_{'mean' if config.pred_mean else 'full'}_fe_neuron_stat"
    all_session_path = results_dir / f"all_sessions_{session_tag}.csv"
    all_neuron_path = results_dir / f"all_neurons_{session_tag}.csv"
    top_session_path = results_dir / f"top_sessions_{session_tag}.csv"
    top_neuron_path = results_dir / f"top_neurons_{session_tag}.csv"
    shap_path = results_dir / f"per_dim_top_sessions_{session_tag}.csv"
    metric_plot_path = results_dir / f"top_sessions_metrics_{session_tag}.pdf"
    shap_plot_path = results_dir / f"familiar_vs_deviant_top_sessions_{config.plot_metric}_{session_tag}.pdf"
    metadata_path = results_dir / f"metadata_{session_tag}.json"

    df_all_sessions.to_csv(all_session_path, index=False)
    df_all_neurons.to_csv(all_neuron_path, index=False)
    df_top_sessions.to_csv(top_session_path, index=False)
    df_top_neurons.to_csv(top_neuron_path, index=False)
    df_shap.to_csv(shap_path, index=False)

    metric_plot_title = (
        f"Top Sessions Ranked by Filtered Pearson Mean\n"
        f"familiar_mode={control_config.familiar_mode}, deviant_mode={control_config.deviant_mode}"
    )
    _plot_top_sessions(
        df_eval=df_top_sessions,
        output_path=metric_plot_path,
        title=metric_plot_title,
        figure_dpi=config.figure_dpi,
    )

    plot_title = (
        f"Familiar vs Deviant importance\n"
        f"top {config.top_k_sessions} sessions after neuron/session filtering / pred={config.pred}\n"
        f"familiar_mode={control_config.familiar_mode}, deviant_mode={control_config.deviant_mode}"
    )
    _plot_filtered_shap_scatter(
        df_shap=df_shap,
        output_path=shap_plot_path,
        metric=config.plot_metric,
        title=plot_title,
        figure_dpi=config.figure_dpi,
    )

    metadata = {
        "config": asdict(config),
        "control_config": asdict(control_config),
        "xgb_params": xgb_params,
        "selected_session_ids": selected_session_ids,
        "notes": {
            "session_selection_rule": "Run all sessions, keep the top 20% neurons by pearson_per_neuron within each session with a minimum of 3 neurons, then rank sessions by pearson_mean_kept and keep top-K sessions.",
            "feature_rule": "Within each CV fold, familiar and deviant embedding blocks are each reduced to 6 PCA dimensions, then concatenated into 12 dimensions.",
            "shap_rule": "SHAP is computed only for kept neurons in selected sessions, using raw controlled X so familiar/deviant feature blocks remain interpretable.",
            "image_stat_rule": "Each role can independently use the original image, a high-brightness image, or a low-contrast image before image embedding extraction.",
        },
        "files": {
            "all_sessions_csv": str(all_session_path),
            "all_neurons_csv": str(all_neuron_path),
            "top_sessions_csv": str(top_session_path),
            "top_neurons_csv": str(top_neuron_path),
            "per_dim_shap_csv": str(shap_path),
            "metric_plot_pdf": str(metric_plot_path),
            "shap_plot_pdf": str(shap_plot_path),
            "oof_prediction_dir": str(prediction_dir),
        },
    }
    if shap_dir is not None:
        metadata["files"]["shap_values_dir"] = str(shap_dir)

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("\nSaved image_stat_control outputs:")
    print(f"  {all_session_path}")
    print(f"  {all_neuron_path}")
    print(f"  {top_session_path}")
    print(f"  {top_neuron_path}")
    print(f"  {shap_path}")
    print(f"  {metric_plot_path}")
    print(f"  {shap_plot_path}")
    print(f"  {metadata_path}")

    return {
        "df_all_sessions": df_all_sessions,
        "df_all_neurons": df_all_neurons,
        "df_top_sessions": df_top_sessions,
        "df_top_neurons": df_top_neurons,
        "df_shap": df_shap,
        "metadata": metadata,
        "paths": {
            "all_sessions_csv": all_session_path,
            "all_neurons_csv": all_neuron_path,
            "top_sessions_csv": top_session_path,
            "top_neurons_csv": top_neuron_path,
            "per_dim_shap_csv": shap_path,
            "metric_plot_pdf": metric_plot_path,
            "shap_plot_pdf": shap_plot_path,
            "metadata_json": metadata_path,
            "oof_prediction_dir": prediction_dir,
        },
    }


def run_plot(config: Optional[FENeuronStatConfig] = None):
    if config is None:
        config = FENeuronStatConfig()

    control_config = _build_control_config(config)
    results_dir = Path(config.results_root) / config.pred / control_config.tag
    session_tag = f"{config.pred}_{'mean' if config.pred_mean else 'full'}_fe_neuron_stat"

    top_session_path = results_dir / f"top_sessions_{session_tag}.csv"
    shap_path = results_dir / f"per_dim_top_sessions_{session_tag}.csv"
    metric_plot_path = results_dir / f"top_sessions_metrics_{session_tag}.pdf"
    shap_plot_path = results_dir / f"familiar_vs_deviant_top_sessions_{config.plot_metric}_{session_tag}.pdf"

    if not top_session_path.exists():
        raise FileNotFoundError(
            f"Missing experiment result file: {top_session_path}. "
            "Run the experiment first before calling run_plot()."
        )
    if not shap_path.exists():
        raise FileNotFoundError(
            f"Missing experiment result file: {shap_path}. "
            "Run the experiment first before calling run_plot()."
        )

    df_top_sessions = pd.read_csv(top_session_path)
    df_shap = pd.read_csv(shap_path)

    if df_top_sessions.empty:
        raise ValueError(f"No session rows found in {top_session_path}")
    if df_shap.empty:
        raise ValueError(f"No SHAP rows found in {shap_path}")

    metric_plot_title = (
        f"Top Sessions Ranked by Filtered Pearson Mean\n"
        f"familiar_mode={control_config.familiar_mode}, deviant_mode={control_config.deviant_mode}"
    )
    _plot_top_sessions(
        df_eval=df_top_sessions,
        output_path=metric_plot_path,
        title=metric_plot_title,
        figure_dpi=config.figure_dpi,
    )

    plot_title = (
        f"Familiar vs Deviant importance\n"
        f"top {config.top_k_sessions} sessions after neuron/session filtering / pred={config.pred}\n"
        f"familiar_mode={control_config.familiar_mode}, deviant_mode={control_config.deviant_mode}"
    )
    _plot_filtered_shap_scatter(
        df_shap=df_shap,
        output_path=shap_plot_path,
        metric=config.plot_metric,
        title=plot_title,
        figure_dpi=config.figure_dpi,
    )

    print("Re-generated image_stat_control plots:")
    print(f"  {metric_plot_path}")
    print(f"  {shap_plot_path}")

    return {
        "paths": {
            "top_sessions_csv": top_session_path,
            "per_dim_shap_csv": shap_path,
            "metric_plot_pdf": metric_plot_path,
            "shap_plot_pdf": shap_plot_path,
        }
    }


if __name__ == "__main__":
    config = FENeuronStatConfig(
        pred="burst_B",
        pred_mean=False,
        need_sup_feat=False,
        n_splits=5,
        random_state=42,
        results_root="./results/fe_neuron_stat",
        variance_threshold=0.95,
        max_components=30,
        fixed_embedding_components=6,
        familiar_mode="none",  # "none", "brightness", "contrast"
        deviant_mode="contrast",  # "none", "brightness", "contrast"
        top_k_sessions=10,
        neuron_keep_fraction=0.2,
        min_neurons_to_keep=3,
        figure_dpi=180,
        plot_metric="pearson",
        save_shap_values=False,
    )

    run_fe_neuron_stat_pipeline(
        session_ids=SESSION_IDS,
        config=config,
        xgb_params=default_xgb_params(random_state=config.random_state),
    )

    run_plot(config=config)
