import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib.pyplot as plt
import numpy as np

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
    ControlConfig,
    _load_trial_structure,
    prepare_data_with_control,
)


@dataclass
class FEUnique1Config:
    pred: str = "burst_B"
    pred_mean: bool = False
    need_sup_feat: bool = False
    n_splits: int = 5
    random_state: int = 42
    results_root: str = "./results/fe_unique_1"
    variance_threshold: float = 0.95
    max_components: int = 30
    fixed_embedding_components: int = 6
    familiar_mode: str = "gaussian_trials"
    deviant_mode: str = "gaussian_trials"
    top_k_sessions: int = 10
    neuron_keep_fraction: float = 0.2
    min_neurons_to_keep: int = 3
    figure_dpi: int = 180


def _build_control_config(config: FEUnique1Config) -> ControlConfig:
    control_config = ControlConfig(
        familiar_mode=config.familiar_mode,
        deviant_mode=config.deviant_mode,
        random_state=config.random_state,
    )
    control_config.validate()

    allowed_modes = {"none", "gaussian_trials", "scramble_pixels"}
    if control_config.familiar_mode not in allowed_modes:
        raise ValueError(
            f"Unsupported familiar_mode={control_config.familiar_mode!r} for unique_image_count_analysis. "
            f"Allowed modes: {sorted(allowed_modes)}"
        )
    if control_config.deviant_mode not in allowed_modes:
        raise ValueError(
            f"Unsupported deviant_mode={control_config.deviant_mode!r} for unique_image_count_analysis. "
            f"Allowed modes: {sorted(allowed_modes)}"
        )
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


def _count_unique_familiar_minus_deviant(
    session_id: str,
) -> tuple[int, int, int]:
    unique_pairs, _, _ = _load_trial_structure(session_id=session_id)
    familiar_count = int(np.unique(unique_pairs[:, 0].astype(str)).size)
    deviant_count = int(np.unique(unique_pairs[:, 1].astype(str)).size)
    return familiar_count, deviant_count, familiar_count - deviant_count


def _plot_neuron_level_difference_bars(
    diffs: np.ndarray,
    output_path: Path,
    title: str,
    figure_dpi: int = 180,
) -> None:
    diffs = np.asarray(diffs, dtype=np.int32).reshape(-1)
    if diffs.size == 0:
        raise ValueError("No neuron-level difference values available for plotting.")

    x = np.arange(diffs.size, dtype=np.int32)
    fig_width = max(10.0, min(28.0, diffs.size * 0.14))

    plt.figure(figsize=(fig_width, 5.8), dpi=figure_dpi)
    plt.bar(
        x,
        diffs,
        width=0.9,
        color="#4c78a8",
        edgecolor="black",
        linewidth=0.5,
        alpha=0.92,
    )
    plt.axhline(0.0, color="black", linewidth=0.9)
    plt.xlabel("Kept neurons")
    plt.ylabel("#unique familiar images - #unique deviant images")
    plt.title(title)
    plt.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)

    if diffs.size <= 80:
        plt.xticks(x, [str(i + 1) for i in x], rotation=90, fontsize=8)
    else:
        step = max(int(np.ceil(diffs.size / 25)), 1)
        tick_positions = x[::step]
        plt.xticks(tick_positions, [str(i + 1) for i in tick_positions], fontsize=9)

    plt.text(
        0.98,
        0.97,
        f"neurons = {diffs.size}\nmean = {diffs.mean():.3f}\nstd = {diffs.std():.3f}",
        transform=plt.gca().transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.92, "edgecolor": "#999999"},
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", format="pdf")
    plt.close()


def run_fe_unique_1_pipeline(
    session_ids=SESSION_IDS,
    config: Optional[FEUnique1Config] = None,
    xgb_params: Optional[Dict] = None,
):
    if config is None:
        config = FEUnique1Config()
    if xgb_params is None:
        xgb_params = default_xgb_params(random_state=config.random_state)

    control_config = _build_control_config(config)
    results_dir = Path(config.results_root) / config.pred / control_config.tag
    results_dir.mkdir(parents=True, exist_ok=True)

    session_infos = []

    print("\n=== Running unique_image_count_analysis session scan ===")
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
            xgb_params=xgb_params,
            variance_threshold=config.variance_threshold,
            max_components=config.max_components,
            fixed_components=config.fixed_embedding_components,
            n_splits=config.n_splits,
            random_state=config.random_state,
        )

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
        familiar_count, deviant_count, unique_diff = _count_unique_familiar_minus_deviant(
            session_id=session_id
        )

        session_row = {
            "session_id": session_id,
            "target_dim": int(meta["target_dim"]),
            "keep_indices": np.flatnonzero(keep_mask),
            "drop_indices": drop_indices,
            "n_neurons_kept": int(keep_mask.sum()),
            "unique_familiar_count": familiar_count,
            "unique_deviant_count": deviant_count,
            "unique_familiar_minus_deviant": unique_diff,
            **session_eval,
        }
        session_infos.append(session_row)

        print(
            f"  [Filter] dropped={len(drop_indices)}, kept={keep_mask.sum()}, "
            f"pearson_mean_kept={session_eval['pearson_mean_kept']:.4f}, "
            f"unique_fam={familiar_count}, unique_dev={deviant_count}, diff={unique_diff}"
        )

    selected_session_infos = sorted(
        session_infos,
        key=lambda row: row["pearson_mean_kept"],
        reverse=True,
    )[: config.top_k_sessions]
    selected_session_ids = [row["session_id"] for row in selected_session_infos]

    print("\nSelected top sessions by filtered Pearson:")
    print(selected_session_ids)

    neuron_level_diffs = []
    for session_info in selected_session_infos:
        keep_indices = np.asarray(session_info["keep_indices"], dtype=int)
        session_diff = int(session_info["unique_familiar_minus_deviant"])
        for neuron_idx in keep_indices:
            neuron_level_diffs.append(session_diff)

    if not neuron_level_diffs:
        raise ValueError("No kept neurons found in selected sessions.")

    session_tag = f"{config.pred}_{'mean' if config.pred_mean else 'full'}_fe_unique_1"
    plot_path = (
        results_dir
        / f"kept_neurons_unique_familiar_minus_deviant_density_{session_tag}.pdf"
    )
    plot_title = (
        "Neuron-level (#unique familiar images - #unique deviant images)\n"
        f"for kept neurons in top {len(selected_session_infos)} sessions / pred={config.pred}\n"
        f"familiar_mode={control_config.familiar_mode}, deviant_mode={control_config.deviant_mode}"
    )
    _plot_neuron_level_difference_bars(
        diffs=np.asarray(neuron_level_diffs, dtype=np.int32),
        output_path=plot_path,
        title=plot_title,
        figure_dpi=config.figure_dpi,
    )

    print("\nSaved unique_image_count_analysis output:")
    print(f"  {plot_path}")

    return {
        "selected_session_ids": selected_session_ids,
        "neuron_level_diffs": np.asarray(neuron_level_diffs, dtype=np.int32),
        "plot_path": plot_path,
    }


if __name__ == "__main__":
    config = FEUnique1Config(
        pred="burst_B",
        pred_mean=False,
        need_sup_feat=False,
        n_splits=5,
        random_state=42,
        results_root="./results/fe_unique_1",
        variance_threshold=0.95,
        max_components=30,
        fixed_embedding_components=6,
        familiar_mode="none",  # "none", "gaussian_trials", "scramble_pixels"
        deviant_mode="none",  # "none", "gaussian_trials", "scramble_pixels"
        top_k_sessions=10,
        neuron_keep_fraction=0.2,
        min_neurons_to_keep=3,
        figure_dpi=180,
    )

    run_fe_unique_1_pipeline(
        session_ids=SESSION_IDS,
        config=config,
        xgb_params=default_xgb_params(random_state=config.random_state),
    )
