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

from embedding_xgboost_baseline import (
    _compute_prediction_metrics_per_dim,
    _compute_prediction_metrics_per_trial,
    _nanmean_or_nan,
    _nanstd_or_nan,
    compute_oof_predictions_with_fe,
    default_xgb_params,
)
from importance_control import SESSION_IDS, ControlConfig, prepare_data_with_control


@dataclass
class FENeuronConfig:
    pred: str = "burst_B"
    pred_mean: bool = False
    need_sup_feat: bool = False
    n_splits: int = 5
    random_state: int = 42
    results_root: str = "./results/fe_neuron"
    variance_threshold: float = 0.95
    max_components: int = 30
    familiar_mode: str = "gaussian_trials"
    deviant_mode: str = "gaussian_trials"
    top_k_sessions: int = 10
    neuron_drop_fraction: float = 1.0 / 3.0
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
            f"Unsupported familiar_mode={control_config.familiar_mode!r} for neuron_filtering. "
            f"Allowed modes: {sorted(allowed_modes)}"
        )
    if control_config.deviant_mode not in allowed_modes:
        raise ValueError(
            f"Unsupported deviant_mode={control_config.deviant_mode!r} for neuron_filtering. "
            f"Allowed modes: {sorted(allowed_modes)}"
        )
    return control_config


def _collect_session_metadata(
    session_ids,
    config: FENeuronConfig,
    control_config: ControlConfig,
) -> list[dict]:
    session_infos = []
    for session_id in session_ids:
        X_raw, y, meta = prepare_data_with_control(
            session_id=session_id,
            pred=config.pred,
            pred_mean=config.pred_mean,
            need_sup_feat=config.need_sup_feat,
            control_config=control_config,
        )
        session_infos.append(
            {
                "session_id": session_id,
                "X_raw": X_raw,
                "y": y,
                "meta": meta,
                "target_dim": int(meta["target_dim"]),
            }
        )
    return session_infos


def _select_top_sessions_by_neuron_count(
    session_infos: list[dict],
    top_k_sessions: int,
) -> list[dict]:
    return sorted(
        session_infos,
        key=lambda row: (row["target_dim"], row["session_id"]),
        reverse=True,
    )[:top_k_sessions]


def _select_neurons_by_pearson(
    pearson_per_neuron: np.ndarray,
    drop_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    n_neurons = len(pearson_per_neuron)
    if n_neurons == 0:
        raise ValueError("No neurons found for filtering.")

    if n_neurons == 1:
        keep_mask = np.array([True], dtype=bool)
        return keep_mask, np.array([], dtype=int)

    n_drop = int(math.floor(n_neurons * drop_fraction))
    if drop_fraction > 0 and n_drop == 0:
        n_drop = 1
    n_drop = min(max(n_drop, 0), n_neurons - 1)

    sort_key = np.where(np.isfinite(pearson_per_neuron), pearson_per_neuron, -np.inf)
    drop_indices = np.argsort(sort_key, kind="stable")[:n_drop]
    keep_mask = np.ones(n_neurons, dtype=bool)
    keep_mask[drop_indices] = False
    return keep_mask, drop_indices


def _summarize_filtered_session(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    keep_mask: np.ndarray,
    r2_per_neuron: np.ndarray,
    pearson_per_neuron: np.ndarray,
    r2_per_trial: np.ndarray,
    pearson_per_trial: np.ndarray,
) -> dict[str, float]:
    y_true_kept = y_true[:, keep_mask]
    y_pred_kept = y_pred[:, keep_mask]

    return {
        "n_neurons_kept": int(keep_mask.sum()),
        "n_neurons_dropped": int((~keep_mask).sum()),
        "y_true_std_kept": float(np.std(y_true_kept)),
        "y_pred_std_kept": float(np.std(y_pred_kept)),
        "pearson_mean_kept": _nanmean_or_nan(pearson_per_neuron[keep_mask]),
        "pearson_std_kept": _nanstd_or_nan(pearson_per_neuron[keep_mask]),
        "r2_mean_kept": _nanmean_or_nan(r2_per_neuron[keep_mask]),
        "r2_std_kept": _nanstd_or_nan(r2_per_neuron[keep_mask]),
        "pearson_mean_per_trial_kept": _nanmean_or_nan(pearson_per_trial),
        "pearson_std_per_trial_kept": _nanstd_or_nan(pearson_per_trial),
        "r2_mean_per_trial_kept": _nanmean_or_nan(r2_per_trial),
        "r2_std_per_trial_kept": _nanstd_or_nan(r2_per_trial),
    }


def _plot_top_sessions(
    df_eval: pd.DataFrame,
    output_path: Path,
    figure_dpi: int = 180,
) -> None:
    df_plot = df_eval.sort_values("pearson_mean_kept", ascending=False).reset_index(drop=True)
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
    axes[0].set_title("Top 10 Sessions Ranked by Filtered Pearson Mean")
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.4)

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
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=30, ha="right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=figure_dpi, bbox_inches="tight")
    plt.close(fig)


def run_fe_neuron_pipeline(
    session_ids=SESSION_IDS,
    config: Optional[FENeuronConfig] = None,
    xgb_params: Optional[Dict] = None,
):
    if config is None:
        config = FENeuronConfig()
    if xgb_params is None:
        xgb_params = default_xgb_params(random_state=config.random_state)

    control_config = _build_control_config(config)
    results_dir = Path(config.results_root) / config.pred / control_config.tag
    results_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Scanning sessions and keeping top sessions by neuron count ===")
    session_infos = _collect_session_metadata(
        session_ids=session_ids,
        config=config,
        control_config=control_config,
    )
    selected_sessions = _select_top_sessions_by_neuron_count(
        session_infos=session_infos,
        top_k_sessions=config.top_k_sessions,
    )

    selected_session_ids = [row["session_id"] for row in selected_sessions]
    print("Selected sessions:", selected_session_ids)

    evaluation_rows = []
    per_neuron_rows = []
    prediction_dir = results_dir / "oof_predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)

    for session_info in selected_sessions:
        session_id = session_info["session_id"]
        X_raw = session_info["X_raw"]
        y = session_info["y"]
        meta = session_info["meta"]

        print(f"\n=== Session: {session_id} / neurons={meta['target_dim']} ===")
        y_pred_oof = compute_oof_predictions_with_fe(
            X_raw=X_raw,
            y=y,
            xgb_params=xgb_params,
            variance_threshold=config.variance_threshold,
            max_components=config.max_components,
            n_splits=config.n_splits,
            random_state=config.random_state,
        )

        np.save(prediction_dir / f"{session_id}_y_true.npy", y)
        np.save(prediction_dir / f"{session_id}_y_pred_oof.npy", y_pred_oof)

        r2_per_neuron, pearson_per_neuron = _compute_prediction_metrics_per_dim(y, y_pred_oof)
        keep_mask, drop_indices = _select_neurons_by_pearson(
            pearson_per_neuron=pearson_per_neuron,
            drop_fraction=config.neuron_drop_fraction,
        )
        r2_per_trial_kept, pearson_per_trial_kept = _compute_prediction_metrics_per_trial(
            y[:, keep_mask],
            y_pred_oof[:, keep_mask],
        )
        session_eval = _summarize_filtered_session(
            y_true=y,
            y_pred=y_pred_oof,
            keep_mask=keep_mask,
            r2_per_neuron=r2_per_neuron,
            pearson_per_neuron=pearson_per_neuron,
            r2_per_trial=r2_per_trial_kept,
            pearson_per_trial=pearson_per_trial_kept,
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

        for neuron_idx in range(y.shape[1]):
            per_neuron_rows.append(
                {
                    "session_id": session_id,
                    "neuron_index": neuron_idx,
                    "is_kept": bool(keep_mask[neuron_idx]),
                    "is_dropped": bool(not keep_mask[neuron_idx]),
                    "r2_per_neuron": float(r2_per_neuron[neuron_idx]),
                    "pearson_per_neuron": float(pearson_per_neuron[neuron_idx]),
                }
            )

        print(
            f"  [Filter] dropped={len(drop_indices)}, kept={keep_mask.sum()}, "
            f"pearson_mean_kept={session_eval['pearson_mean_kept']:.4f}, "
            f"pearson_std_kept={session_eval['pearson_std_kept']:.4f}, "
            f"r2_mean_kept={session_eval['r2_mean_kept']:.4f}, "
            f"r2_std_kept={session_eval['r2_std_kept']:.4f}"
        )

    df_eval = pd.DataFrame(evaluation_rows).sort_values(
        "pearson_mean_kept",
        ascending=False,
    )
    df_per_neuron = pd.DataFrame(per_neuron_rows)

    tag = f"{config.pred}_{'mean' if config.pred_mean else 'full'}_fe_neuron"
    eval_path = results_dir / f"evaluation_{tag}.csv"
    per_neuron_path = results_dir / f"per_neuron_{tag}.csv"
    plot_path = results_dir / f"top10_sessions_filtered_metrics_{tag}.png"
    metadata_path = results_dir / f"metadata_{tag}.json"

    df_eval.to_csv(eval_path, index=False)
    df_per_neuron.to_csv(per_neuron_path, index=False)
    _plot_top_sessions(
        df_eval=df_eval,
        output_path=plot_path,
        figure_dpi=config.figure_dpi,
    )

    metadata = {
        "config": asdict(config),
        "selected_session_ids": selected_session_ids,
        "xgb_params": xgb_params,
        "control_config": asdict(control_config),
        "files": {
            "evaluation_csv": str(eval_path),
            "per_neuron_csv": str(per_neuron_path),
            "plot_png": str(plot_path),
            "oof_prediction_dir": str(prediction_dir),
        },
        "notes": {
            "top_session_rule": "Keep only top-K sessions with the largest target_dim from raw prepared data.",
            "neuron_filter_rule": "Within each selected session, drop the bottom 1/3 neurons ranked by pearson_per_neuron and compute summary stats on the kept neurons only.",
            "filter_metric": "pearson_per_neuron",
        },
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("\nSaved neuron_filtering outputs:")
    print(f"  {eval_path}")
    print(f"  {per_neuron_path}")
    print(f"  {plot_path}")
    print(f"  {metadata_path}")

    return {
        "df_evaluation": df_eval,
        "df_per_neuron": df_per_neuron,
        "metadata": metadata,
        "paths": {
            "evaluation_csv": eval_path,
            "per_neuron_csv": per_neuron_path,
            "plot_png": plot_path,
            "metadata_json": metadata_path,
            "oof_prediction_dir": prediction_dir,
        },
    }


if __name__ == "__main__":
    config = FENeuronConfig(
        pred="burst_B",
        pred_mean=False,
        need_sup_feat=False,
        n_splits=5,
        random_state=42,
        results_root="./results/fe_neuron",
        variance_threshold=0.95,
        max_components=30,
        familiar_mode="none",
        deviant_mode="none",
        top_k_sessions=10,
        neuron_drop_fraction=1.0 / 3.0,
        figure_dpi=180,
    )

    run_fe_neuron_pipeline(
        session_ids=SESSION_IDS,
        config=config,
        xgb_params=default_xgb_params(random_state=config.random_state),
    )
