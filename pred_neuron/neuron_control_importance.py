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
    ControlConfig,
    _default_color_limits_for_metric,
    _load_trial_structure,
    _validate_plot_metric,
    plot_familiar_vs_deviant,
    prepare_data_with_control,
)


@dataclass
class FENeuronControlConfig:
    pred: str = "burst_B"
    pred_mean: bool = False
    need_sup_feat: bool = False
    n_splits: int = 5
    random_state: int = 42
    results_root: str = "./results/fe_neuron_control"
    variance_threshold: float = 0.95
    max_components: int = 30
    fixed_embedding_components: int = 6
    familiar_mode: str = "gaussian_trials"
    deviant_mode: str = "gaussian_trials"
    top_k_sessions: int = 10
    neuron_keep_fraction: float = 0.2
    min_neurons_to_keep: int = 3
    figure_dpi: int = 180
    plot_metric: str = "pearson"
    save_shap_values: bool = False


PAPER_EXPORT_DIRNAME = "paper_experiment_data"
PAPER_METADATA_DIRNAME = "meta_data"
PAPER_PREDICT_DIRNAME = "predict"
PAPER_EVAL_DIRNAME = "eval"
PAPER_IMPORTANCE_DIRNAME = "importance"
PAPER_METADATA_FILENAME = "experiment_metadata.json"
PAPER_EVAL_FILENAME = "session_eval.csv"


def _build_control_config(config: FENeuronControlConfig) -> ControlConfig:
    control_config = ControlConfig(
        familiar_mode=config.familiar_mode,
        deviant_mode=config.deviant_mode,
        random_state=config.random_state,
    )
    control_config.validate()

    allowed_modes = {"none", "gaussian_trials", "scramble_pixels"}
    if control_config.familiar_mode not in allowed_modes:
        raise ValueError(
            f"Unsupported familiar_mode={control_config.familiar_mode!r} for neuron_control_importance. "
            f"Allowed modes: {sorted(allowed_modes)}"
        )
    if control_config.deviant_mode not in allowed_modes:
        raise ValueError(
            f"Unsupported deviant_mode={control_config.deviant_mode!r} for neuron_control_importance. "
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


def _plot_top_sessions(
    df_eval: pd.DataFrame,
    output_path: Path,
    title: Optional[str] = None,
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

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(16, 10),
        sharex=True,
        gridspec_kw={"hspace": 0.18},
        constrained_layout=True,
    )

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
    axes[0].set_title(title or "Top Sessions Ranked by Filtered Pearson Mean")
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


def _shap_fit_params(xgb_params: Dict) -> Dict:
    params = dict(xgb_params)
    params.pop("early_stopping_rounds", None)
    return params


def _paper_export_paths(results_dir: Path) -> dict[str, Path]:
    export_dir = results_dir / PAPER_EXPORT_DIRNAME
    return {
        "export_dir": export_dir,
        "meta_data_dir": export_dir / PAPER_METADATA_DIRNAME,
        "predict_dir": export_dir / PAPER_PREDICT_DIRNAME,
        "eval_dir": export_dir / PAPER_EVAL_DIRNAME,
        "importance_dir": export_dir / PAPER_IMPORTANCE_DIRNAME,
        "metadata_json": export_dir / PAPER_METADATA_DIRNAME / PAPER_METADATA_FILENAME,
        "eval_csv": export_dir / PAPER_EVAL_DIRNAME / PAPER_EVAL_FILENAME,
    }


def _parse_index_list(raw_value: object, field_name: str) -> list[int]:
    if isinstance(raw_value, str):
        parsed = json.loads(raw_value)
    elif isinstance(raw_value, (list, tuple, np.ndarray)):
        parsed = list(raw_value)
    else:
        raise TypeError(f"Unsupported type for {field_name}: {type(raw_value)!r}")
    return [int(v) for v in parsed]


def _export_paper_experiment_data(
    results_dir: Path,
    config: FENeuronControlConfig,
    control_config: ControlConfig,
    selected_session_infos: list[dict],
    df_shap: pd.DataFrame,
) -> dict[str, object]:
    if not selected_session_infos:
        raise ValueError("No selected sessions available for paper data export.")
    if df_shap.empty:
        raise ValueError("No SHAP rows available for paper data export.")

    paths = _paper_export_paths(results_dir)
    paths["export_dir"].mkdir(parents=True, exist_ok=True)
    paths["meta_data_dir"].mkdir(parents=True, exist_ok=True)
    paths["predict_dir"].mkdir(parents=True, exist_ok=True)
    paths["eval_dir"].mkdir(parents=True, exist_ok=True)
    paths["importance_dir"].mkdir(parents=True, exist_ok=True)

    shap_lookup = df_shap.set_index(["session_id", "neuron_index"], verify_integrity=True)
    session_order = [info["session_id"] for info in selected_session_infos]
    session_entries = []
    eval_rows: list[dict[str, object]] = []

    for session_rank, session_info in enumerate(selected_session_infos):
        session_id = session_info["session_id"]
        X_raw = np.asarray(session_info["X_raw"], dtype=np.float32)
        y_true = np.asarray(session_info["y"], dtype=np.float32)
        y_pred_raw = session_info.get("y_pred_oof")
        if y_pred_raw is None or np.size(y_pred_raw) == 0:
            prediction_path = results_dir / "oof_predictions" / f"{session_id}_y_pred_oof.npy"
            if not prediction_path.exists():
                raise FileNotFoundError(
                    f"Missing OOF prediction file for session {session_id}: {prediction_path}"
                )
            y_pred = np.load(prediction_path).astype(np.float32, copy=False)
        else:
            y_pred = np.asarray(y_pred_raw, dtype=np.float32)
        meta = session_info["meta"]
        keep_indices = np.asarray(session_info["keep_indices"], dtype=np.int64)
        session_row = session_info["session_row"]
        keep_mask = np.asarray(session_info["keep_mask"], dtype=bool)
        d_img = int(meta["img_feature_dim"])

        unique_pairs, _, _ = _load_trial_structure(session_id=session_id)
        if unique_pairs.shape[0] != X_raw.shape[0]:
            raise ValueError(
                f"Session {session_id} has {X_raw.shape[0]} unique rows in X_raw but "
                f"{unique_pairs.shape[0]} unique image pairs."
            )
        if X_raw.shape[1] < 2 * d_img:
            raise ValueError(
                f"Session {session_id} has input_feature_dim={X_raw.shape[1]}, "
                f"which is smaller than the expected 2 * img_feature_dim = {2 * d_img}."
            )

        session_embeddings = np.stack(
            [X_raw[:, :d_img], X_raw[:, d_img : 2 * d_img]],
            axis=1,
        ).astype(np.float32, copy=False)
        y_true_kept = y_true[:, keep_mask].astype(np.float32, copy=False)
        y_pred_kept = y_pred[:, keep_mask].astype(np.float32, copy=False)
        predict_path = paths["predict_dir"] / f"{session_id}.npz"
        np.savez_compressed(
            predict_path,
            raw_embedding=session_embeddings,
            predict_neuron=y_pred_kept,
            groundtruth_neuron=y_true_kept,
            kept_neuron_indices=keep_indices,
            trial_image_pairs=np.asarray(unique_pairs),
        )

        familiar_shap = np.zeros(len(keep_indices), dtype=np.float32)
        deviant_shap = np.zeros(len(keep_indices), dtype=np.float32)
        pearson_kept = np.zeros(len(keep_indices), dtype=np.float32)
        for row_idx, neuron_idx in enumerate(keep_indices.tolist()):
            key = (session_id, int(neuron_idx))
            if key not in shap_lookup.index:
                raise KeyError(
                    f"Missing SHAP row for session={session_id}, neuron_index={neuron_idx}."
                )
            shap_row = shap_lookup.loc[key]
            familiar_shap[row_idx] = float(shap_row["familiar_importance"])
            deviant_shap[row_idx] = float(shap_row["deviant_importance"])
            pearson_kept[row_idx] = float(shap_row["pearson"])

        importance_path = paths["importance_dir"] / f"{session_id}.npz"
        np.savez_compressed(
            importance_path,
            neuron_index=keep_indices,
            familiar_shap=familiar_shap,
            deviant_shap=deviant_shap,
            pearson=pearson_kept,
        )

        session_entries.append(
            {
                "session_id": session_id,
                "session_rank": int(session_rank),
                "ntrials_raw": int(meta["ntrials_raw"]),
                "ntrials_unique": int(meta["ntrials_unique"]),
                "img_feature_dim": d_img,
                "input_feature_dim_raw": int(meta["input_feature_dim"]),
                "target_dim": int(meta["target_dim"]),
                "kept_neuron_indices": keep_indices.tolist(),
                "trial_image_pairs": unique_pairs.tolist(),
                "predict_file": str(Path(PAPER_PREDICT_DIRNAME) / f"{session_id}.npz"),
                "importance_file": str(Path(PAPER_IMPORTANCE_DIRNAME) / f"{session_id}.npz"),
            }
        )

        eval_rows.append(
            {
                "session_id": session_id,
                "ntrials_raw": int(meta["ntrials_raw"]),
                "ntrials_unique": int(meta["ntrials_unique"]),
                "n_neurons_kept": int(session_row["n_neurons_kept"]),
                "groundtruth_mean": float(np.mean(y_true_kept)),
                "groundtruth_std": float(np.std(y_true_kept)),
                "prediction_mean": float(np.mean(y_pred_kept)),
                "prediction_std": float(np.std(y_pred_kept)),
                "pearson_mean": float(session_row["pearson_mean_kept"]),
                "pearson_std": float(session_row["pearson_std_kept"]),
                "r2_mean": float(session_row["r2_mean_kept"]),
                "r2_std": float(session_row["r2_std_kept"]),
            }
        )

    df_eval_export = pd.DataFrame(eval_rows).sort_values("pearson_mean", ascending=False).reset_index(
        drop=True
    )
    df_eval_export.to_csv(paths["eval_csv"], index=False)

    export_metadata = {
        "description": (
            "Paper-ready experiment package exported from neuron_control_importance. "
            "Data are organized into meta_data, predict, eval, and importance folders."
        ),
        "config": asdict(config),
        "control_config": asdict(control_config),
        "session_order": session_order,
        "n_selected_sessions": int(len(session_entries)),
        "role_order": ["familiar", "deviant"],
        "files": {
            "metadata_json": str(Path(PAPER_METADATA_DIRNAME) / PAPER_METADATA_FILENAME),
            "eval_csv": str(Path(PAPER_EVAL_DIRNAME) / PAPER_EVAL_FILENAME),
        },
        "folder_specs": {
            "meta_data": {
                "path": PAPER_METADATA_DIRNAME,
                "contents": [PAPER_METADATA_FILENAME],
            },
            "predict": {
                "path": PAPER_PREDICT_DIRNAME,
                "file_format": "one .npz file per session",
                "keys": [
                    "raw_embedding",
                    "predict_neuron",
                    "groundtruth_neuron",
                    "kept_neuron_indices",
                    "trial_image_pairs",
                ],
            },
            "eval": {
                "path": PAPER_EVAL_DIRNAME,
                "file_format": "single CSV summary",
                "contents": [PAPER_EVAL_FILENAME],
            },
            "importance": {
                "path": PAPER_IMPORTANCE_DIRNAME,
                "file_format": "one .npz file per session",
                "keys": ["neuron_index", "familiar_shap", "deviant_shap", "pearson"],
            },
        },
        "sessions": session_entries,
    }

    with open(paths["metadata_json"], "w", encoding="utf-8") as f:
        json.dump(export_metadata, f, indent=2, ensure_ascii=False)

    return {
        "metadata": export_metadata,
        "paths": paths,
        "eval_shape": df_eval_export.shape,
    }


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


def run_fe_neuron_control_pipeline(
    session_ids=SESSION_IDS,
    config: Optional[FENeuronControlConfig] = None,
    xgb_params: Optional[Dict] = None,
):
    if config is None:
        config = FENeuronControlConfig()
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
                "y_pred_oof": y_pred_oof,
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

    session_tag = f"{config.pred}_{'mean' if config.pred_mean else 'full'}_fe_neuron_control"
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
        f"pred={config.pred}, familiar_mode={control_config.familiar_mode}, "
        f"deviant_mode={control_config.deviant_mode}"
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

    paper_export = _export_paper_experiment_data(
        results_dir=results_dir,
        config=config,
        control_config=control_config,
        selected_session_infos=selected_session_infos,
        df_shap=df_shap,
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
            "paper_experiment_data_dir": str(paper_export["paths"]["export_dir"]),
            "paper_meta_data_dir": str(paper_export["paths"]["meta_data_dir"]),
            "paper_predict_dir": str(paper_export["paths"]["predict_dir"]),
            "paper_eval_dir": str(paper_export["paths"]["eval_dir"]),
            "paper_importance_dir": str(paper_export["paths"]["importance_dir"]),
            "paper_experiment_metadata_json": str(paper_export["paths"]["metadata_json"]),
            "paper_session_eval_csv": str(paper_export["paths"]["eval_csv"]),
        },
    }
    if shap_dir is not None:
        metadata["files"]["shap_values_dir"] = str(shap_dir)

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("\nSaved neuron_control_importance outputs:")
    print(f"  {all_session_path}")
    print(f"  {all_neuron_path}")
    print(f"  {top_session_path}")
    print(f"  {top_neuron_path}")
    print(f"  {shap_path}")
    print(f"  {metric_plot_path}")
    print(f"  {shap_plot_path}")
    print(f"  {metadata_path}")
    print(f"  {paper_export['paths']['export_dir']}")

    return {
        "df_all_sessions": df_all_sessions,
        "df_all_neurons": df_all_neurons,
        "df_top_sessions": df_top_sessions,
        "df_top_neurons": df_top_neurons,
        "df_shap": df_shap,
        "metadata": metadata,
        "paper_export": paper_export,
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
            "paper_experiment_data_dir": paper_export["paths"]["export_dir"],
            "paper_meta_data_dir": paper_export["paths"]["meta_data_dir"],
            "paper_predict_dir": paper_export["paths"]["predict_dir"],
            "paper_eval_dir": paper_export["paths"]["eval_dir"],
            "paper_importance_dir": paper_export["paths"]["importance_dir"],
            "paper_experiment_metadata_json": paper_export["paths"]["metadata_json"],
            "paper_session_eval_csv": paper_export["paths"]["eval_csv"],
        },
    }


def export_paper_experiment_data(config: Optional[FENeuronControlConfig] = None):
    if config is None:
        config = FENeuronControlConfig()

    control_config = _build_control_config(config)
    results_dir = Path(config.results_root) / config.pred / control_config.tag
    session_tag = f"{config.pred}_{'mean' if config.pred_mean else 'full'}_fe_neuron_control"
    top_session_path = results_dir / f"top_sessions_{session_tag}.csv"
    shap_path = results_dir / f"per_dim_top_sessions_{session_tag}.csv"

    if not top_session_path.exists():
        raise FileNotFoundError(
            f"Missing experiment result file: {top_session_path}. "
            "Run the experiment first before exporting paper data."
        )
    if not shap_path.exists():
        raise FileNotFoundError(
            f"Missing experiment result file: {shap_path}. "
            "Run the experiment first before exporting paper data."
        )

    df_top_sessions = pd.read_csv(top_session_path)
    df_shap = pd.read_csv(shap_path)
    if df_top_sessions.empty:
        raise ValueError(f"No session rows found in {top_session_path}")
    if df_shap.empty:
        raise ValueError(f"No SHAP rows found in {shap_path}")

    selected_session_infos = []
    for row in df_top_sessions.sort_values("pearson_mean_kept", ascending=False).itertuples(index=False):
        keep_indices = np.asarray(
            _parse_index_list(row.kept_neuron_indices, field_name="kept_neuron_indices"),
            dtype=np.int64,
        )
        X_raw, y, meta = prepare_data_with_control(
            session_id=row.session_id,
            pred=config.pred,
            pred_mean=config.pred_mean,
            need_sup_feat=config.need_sup_feat,
            control_config=control_config,
        )

        selected_session_infos.append(
            {
                "session_id": row.session_id,
                "X_raw": X_raw,
                "y": y,
                "y_pred_oof": np.asarray([]),  # placeholder, not used during re-export
                "meta": meta,
                "keep_mask": np.isin(np.arange(y.shape[1]), keep_indices),
                "keep_indices": keep_indices,
                "session_row": {
                    "n_neurons_kept": int(len(keep_indices)),
                    "pearson_mean_kept": float(row.pearson_mean_kept),
                    "pearson_std_kept": float(row.pearson_std_kept),
                    "r2_mean_kept": float(row.r2_mean_kept),
                    "r2_std_kept": float(row.r2_std_kept),
                },
            }
        )

    paper_export = _export_paper_experiment_data(
        results_dir=results_dir,
        config=config,
        control_config=control_config,
        selected_session_infos=selected_session_infos,
        df_shap=df_shap,
    )

    print("Exported paper experiment data:")
    print(f"  {paper_export['paths']['export_dir']}")
    print(f"  {paper_export['paths']['meta_data_dir']}")
    print(f"  {paper_export['paths']['predict_dir']}")
    print(f"  {paper_export['paths']['eval_dir']}")
    print(f"  {paper_export['paths']['importance_dir']}")
    print(f"  {paper_export['paths']['metadata_json']}")
    print(f"  {paper_export['paths']['eval_csv']}")

    return paper_export


def run_plot(config: Optional[FENeuronControlConfig] = None):
    if config is None:
        config = FENeuronControlConfig()

    control_config = _build_control_config(config)
    results_dir = Path(config.results_root) / config.pred / control_config.tag
    session_tag = f"{config.pred}_{'mean' if config.pred_mean else 'full'}_fe_neuron_control"

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
        f"pred={config.pred}, familiar_mode={control_config.familiar_mode}, "
        f"deviant_mode={control_config.deviant_mode}"
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

    print("Re-generated neuron_control_importance plots:")
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
    config = FENeuronControlConfig(
        pred="tonic_B",  # tonic_B, burst_B
        pred_mean=False,
        need_sup_feat=False,
        n_splits=5,
        random_state=42,
        results_root="./results/fe_neuron_control",
        variance_threshold=0.95,
        max_components=30,
        fixed_embedding_components=6,
        familiar_mode="none",  # "none", "gaussian_trials", "scramble_pixels"
        deviant_mode="none",  # "none", "gaussian_trials", "scramble_pixels"
        top_k_sessions=10,
        neuron_keep_fraction=0.2,
        min_neurons_to_keep=3,
        figure_dpi=180,
        plot_metric="pearson",
        save_shap_values=False,
    )

    run_fe_neuron_control_pipeline(
        session_ids=SESSION_IDS,
        config=config,
        xgb_params=default_xgb_params(random_state=config.random_state),
    )

    run_plot(config=config)
