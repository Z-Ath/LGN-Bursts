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

from neuron_control_importance import (
    _compute_filtered_shap_rows,
    _plot_filtered_shap_scatter,
    _plot_top_sessions,
    _select_neurons_by_pearson,
    _summarize_filtered_session,
)
from embedding_xgboost_baseline import (
    _compute_prediction_metrics_per_dim,
    compute_oof_predictions_with_fe,
    default_xgb_params,
)
from importance_control import (
    SESSION_IDS,
    ControlConfig,
    _aggregate_support_features,
    _aggregate_trial_targets,
    _build_gaussian_role_embeddings,
    _build_original_embeddings,
    _build_role_to_embedding,
    _load_trial_structure,
    _role_seed,
)


@dataclass
class FERandomImgConfig:
    pred: str = "burst_B"
    pred_mean: bool = False
    need_sup_feat: bool = False
    n_splits: int = 5
    random_state: int = 42
    results_root: str = "./results/fe_random_img"
    variance_threshold: float = 0.95
    max_components: int = 30
    fixed_embedding_components: int = 6
    familiar_mode: str = "none"
    deviant_mode: str = "none"
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


def _build_control_config(config: FERandomImgConfig) -> ControlConfig:
    control_config = ControlConfig(
        familiar_mode=config.familiar_mode,
        deviant_mode=config.deviant_mode,
        random_state=config.random_state,
    )
    control_config.validate()

    allowed_modes = {"none", "gaussian_trials", "scramble_pixels"}
    if control_config.familiar_mode not in allowed_modes:
        raise ValueError(
            f"Unsupported familiar_mode={control_config.familiar_mode!r} for random_image_control. "
            f"Allowed modes: {sorted(allowed_modes)}"
        )
    if control_config.deviant_mode not in allowed_modes:
        raise ValueError(
            f"Unsupported deviant_mode={control_config.deviant_mode!r} for random_image_control. "
            f"Allowed modes: {sorted(allowed_modes)}"
        )
    return control_config


def _sample_other_image_labels(
    actual_labels: np.ndarray,
    available_labels: list[str],
    seed: int,
) -> np.ndarray:
    if len(available_labels) < 2:
        raise ValueError(
            "Random-image control requires at least two images so each trial can use "
            "a different image's embedding."
        )

    sorted_labels = list(sorted(str(label) for label in available_labels))
    label_to_index = {label: idx for idx, label in enumerate(sorted_labels)}
    rng = np.random.default_rng(seed)

    sampled_labels: list[str] = []
    for label in np.asarray(actual_labels, dtype=str):
        if label not in label_to_index:
            raise KeyError(f"Image label '{label}' not found in embedding dictionary.")

        current_idx = label_to_index[label]
        random_idx = int(rng.integers(0, len(sorted_labels) - 1))
        if random_idx >= current_idx:
            random_idx += 1
        sampled_labels.append(sorted_labels[random_idx])

    return np.asarray(sampled_labels, dtype=str)


def prepare_data_with_random_img(
    session_id: str,
    pred: str = "tonic_A",
    pred_mean: bool = False,
    need_sup_feat: bool = False,
    control_config: Optional[ControlConfig] = None,
    emb_cache_root: str = "./data/embeddings/shap_control",
) -> tuple[np.ndarray, np.ndarray, Dict]:
    if control_config is None:
        control_config = ControlConfig()
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

    modes_in_use = {control_config.familiar_mode, control_config.deviant_mode}
    role_to_emb = None
    if any(mode != "gaussian_trials" for mode in modes_in_use):
        role_to_emb = _build_role_to_embedding(
            control_config=control_config,
            cache_root=emb_cache_root,
        )
        example_emb = next(iter(role_to_emb["familiar"].values()))
    else:
        example_emb = next(iter(_build_original_embeddings().values()))

    d_img = int(example_emb.shape[0])
    row_gaussian_embeddings: dict[str, np.ndarray] = {}
    gaussian_meta: dict[str, int] = {}

    if control_config.familiar_mode == "gaussian_trials":
        familiar_gaussian, familiar_seed = _build_gaussian_role_embeddings(
            session_id=session_id,
            unique_pairs=unique_pairs,
            role="familiar",
            control_config=control_config,
            d_img=d_img,
        )
        row_gaussian_embeddings["familiar"] = familiar_gaussian
        gaussian_meta["familiar_seed"] = familiar_seed

    if control_config.deviant_mode == "gaussian_trials":
        deviant_gaussian, deviant_seed = _build_gaussian_role_embeddings(
            session_id=session_id,
            unique_pairs=unique_pairs,
            role="deviant",
            control_config=control_config,
            d_img=d_img,
        )
        row_gaussian_embeddings["deviant"] = deviant_gaussian
        gaussian_meta["deviant_seed"] = deviant_seed

    actual_familiar_labels = unique_pairs[:, 0].astype(str)
    actual_deviant_labels = unique_pairs[:, 1].astype(str)
    remapped_labels: dict[str, np.ndarray] = {}
    random_img_meta: dict[str, object] = {
        "strategy": "per_trial_sample_other_image_embedding",
        "familiar_applied": control_config.familiar_mode != "gaussian_trials",
        "deviant_applied": control_config.deviant_mode != "gaussian_trials",
    }

    if control_config.familiar_mode != "gaussian_trials":
        if role_to_emb is None:
            raise RuntimeError("Image embeddings were not loaded for familiar role.")
        familiar_seed = _role_seed(control_config, session_id, "familiar", "random_img")
        familiar_available_labels = sorted(role_to_emb["familiar"].keys())
        remapped_labels["familiar"] = _sample_other_image_labels(
            actual_labels=actual_familiar_labels,
            available_labels=familiar_available_labels,
            seed=familiar_seed,
        )
        random_img_meta["familiar_seed"] = int(familiar_seed)
        random_img_meta["familiar_available_images"] = len(familiar_available_labels)
        random_img_meta["familiar_same_label_count"] = int(
            np.sum(remapped_labels["familiar"] == actual_familiar_labels)
        )

    if control_config.deviant_mode != "gaussian_trials":
        if role_to_emb is None:
            raise RuntimeError("Image embeddings were not loaded for deviant role.")
        deviant_seed = _role_seed(control_config, session_id, "deviant", "random_img")
        deviant_available_labels = sorted(role_to_emb["deviant"].keys())
        remapped_labels["deviant"] = _sample_other_image_labels(
            actual_labels=actual_deviant_labels,
            available_labels=deviant_available_labels,
            seed=deviant_seed,
        )
        random_img_meta["deviant_seed"] = int(deviant_seed)
        random_img_meta["deviant_available_images"] = len(deviant_available_labels)
        random_img_meta["deviant_same_label_count"] = int(
            np.sum(remapped_labels["deviant"] == actual_deviant_labels)
        )

    img_emb_unique = np.zeros((unique_pairs.shape[0], d_img * 2), dtype=example_emb.dtype)

    for k, _ in enumerate(unique_pairs):
        if control_config.familiar_mode == "gaussian_trials":
            familiar_emb = row_gaussian_embeddings["familiar"][k]
        else:
            familiar_label = remapped_labels["familiar"][k]
            familiar_emb = role_to_emb["familiar"][familiar_label]

        if control_config.deviant_mode == "gaussian_trials":
            deviant_emb = row_gaussian_embeddings["deviant"][k]
        else:
            deviant_label = remapped_labels["deviant"][k]
            deviant_emb = role_to_emb["deviant"][deviant_label]

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
        "gaussian_meta": gaussian_meta,
        "random_img_meta": random_img_meta,
    }

    print(
        f"Prepared random-image data for session {session_id} / pred={pred} / "
        f"pred_mean={pred_mean} / sup_feat={need_sup_feat}"
    )
    print(f"  control: {control_config.describe()}")
    print(f"  raw trials: {meta['ntrials_raw']}, unique trials: {meta['ntrials_unique']}")
    print(f"  X shape: {X.shape}, y shape: {y.shape}")

    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32), meta


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
    config: FERandomImgConfig,
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
            "Paper-ready experiment package exported from random_image_control. "
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


def run_fe_random_img_pipeline(
    session_ids=SESSION_IDS,
    config: Optional[FERandomImgConfig] = None,
    xgb_params: Optional[Dict] = None,
):
    if config is None:
        config = FERandomImgConfig()
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

    print("\n=== Running FE + random-image embedding + neuron filter on all sessions ===")
    for session_id in session_ids:
        print(f"\n=== Session: {session_id} ===")
        X_raw, y, meta = prepare_data_with_random_img(
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

    session_tag = f"{config.pred}_{'mean' if config.pred_mean else 'full'}_fe_random_img"
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
        f"pred={config.pred}, random_img=other-image, "
        f"familiar_mode={control_config.familiar_mode}, "
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
        f"random_img=other-image, familiar_mode={control_config.familiar_mode}, "
        f"deviant_mode={control_config.deviant_mode}"
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
            "random_img_rule": "For each unique trial row and role, if the role mode is not gaussian_trials, the actual image label is replaced with a random different image label before looking up the role-specific embedding.",
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

    print("\nSaved random_image_control outputs:")
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


def export_paper_experiment_data(config: Optional[FERandomImgConfig] = None):
    if config is None:
        config = FERandomImgConfig()

    control_config = _build_control_config(config)
    results_dir = Path(config.results_root) / config.pred / control_config.tag
    session_tag = f"{config.pred}_{'mean' if config.pred_mean else 'full'}_fe_random_img"
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
        X_raw, y, meta = prepare_data_with_random_img(
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
                "y_pred_oof": np.asarray([]),  # placeholder, loaded from disk during re-export
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


def run_plot(config: Optional[FERandomImgConfig] = None):
    if config is None:
        config = FERandomImgConfig()

    control_config = _build_control_config(config)
    results_dir = Path(config.results_root) / config.pred / control_config.tag
    session_tag = f"{config.pred}_{'mean' if config.pred_mean else 'full'}_fe_random_img"

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
        f"pred={config.pred}, random_img=other-image, "
        f"familiar_mode={control_config.familiar_mode}, "
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
        f"random_img=other-image, familiar_mode={control_config.familiar_mode}, "
        f"deviant_mode={control_config.deviant_mode}"
    )
    _plot_filtered_shap_scatter(
        df_shap=df_shap,
        output_path=shap_plot_path,
        metric=config.plot_metric,
        title=plot_title,
        figure_dpi=config.figure_dpi,
    )

    print("Re-generated random_image_control plots:")
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
    config = FERandomImgConfig(
        pred="tonic_B",  # tonic_B, burst_A
        pred_mean=False,
        need_sup_feat=False,
        n_splits=5,
        random_state=42,
        results_root="./results/fe_random_img",
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

    run_fe_random_img_pipeline(
        session_ids=SESSION_IDS,
        config=config,
        xgb_params=default_xgb_params(random_state=config.random_state),
    )

    run_plot(config=config)
