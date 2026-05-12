import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Union

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from xgboost import XGBRegressor


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


VALID_CONTROL_MODES = {"none", "scramble_pixels", "shuffle_trials", "gaussian_trials"}
VALID_PLOT_METRICS = {"r2", "pearson"}
DEFAULT_EMB_CACHE_PATH = "./data/embeddings/vit_b16_embeddings.npz"
DEFAULT_PLOT_CMAP = "Blues"
DEFAULT_TRIAL_EMBED_DIM = 768
_IN_MEMORY_EMBEDDING_CACHE: dict[str, Dict[str, np.ndarray]] = {}

POP_VECTOR_PATHS = {
    "tonic_A": (
        "./data/Model_features_predicting_neural_data/"
        "Predicting_tonic_A/pop_vector_mean_tonic_rate_epochA.h5"
    ),
    "tonic_B": (
        "./data/Model_features_predicting_neural_data/"
        "Predicting_tonic_B/pop_vector_mean_tonic_rate_epochB.h5"
    ),
    "burst_B": (
        "./data/Model_features_predicting_neural_data/"
        "Predicting_burst_B/pop_vector_mean_burst_rate_epochB.h5"
    ),
}


def _resolve_pop_vector_path(predict: str) -> str:
    if predict not in POP_VECTOR_PATHS:
        raise ValueError(
            f"Unsupported predict {predict!r}. "
            f"Expected one of {sorted(POP_VECTOR_PATHS)}."
        )
    return POP_VECTOR_PATHS[predict]


def load_sup_feat(session_id: str = "session_1091039376", pred: str = "tonic_A"):
    if pred == "tonic_A":
        filename = (
            "./data/Model_features_predicting_neural_data/"
            "Predicting_tonic_A/supporting_features_to_predict_tonic_epochA.h5"
        )
        feature_names = [
            "flashes_since_HG",
            "tonic_rate_BL",
            "layerVI",
            "layerV",
            "layerIV",
            "layer23",
        ]
    elif pred == "burst_B":
        filename = (
            "./data/Model_features_predicting_neural_data/"
            "Predicting_burst_B/supporting_features_to_predict_burst_epochB_W_behaviour.h5"
        )
        feature_names = [
            "flashes_since_HG",
            "burst_rate_BL",
            "layerVI_A",
            "layerV_A",
            "layerIV_A",
            "layer23_A",
            "layerVI_preB",
            "layerV_preB",
            "run_speed",
            "lick_rate",
            "pupil_area",
            "pupil_move",
        ]
    else:
        raise ValueError(f"Unsupported pred={pred}, expected 'tonic_A' or 'burst_B'.")

    def to_trials_column(arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 1:
            vec = arr
        elif arr.ndim == 2:
            if 1 in arr.shape:
                vec = np.squeeze(arr)
                if vec.ndim != 1:
                    raise ValueError(f"Cannot squeeze to vector from shape {arr.shape}")
            else:
                if arr.shape[0] <= arr.shape[1]:
                    vec = arr.mean(axis=0)
                else:
                    vec = arr.mean(axis=1)
        else:
            raise ValueError(f"Unsupported ndim={arr.ndim} with shape {arr.shape}")

        return np.asarray(vec).reshape(-1, 1)

    with h5py.File(filename, "r") as f:
        if session_id not in f.keys():
            raise KeyError(
                f"Session {session_id} not found. "
                f"Available sessions: {list(f.keys())}"
            )
        g = f[session_id]

        missing = [k for k in feature_names if k not in g.keys()]
        if missing:
            raise KeyError(f"Missing features in {session_id}: {missing}")

        cols = []
        lengths = []
        for name in feature_names:
            arr = np.array(g[name])
            col = to_trials_column(arr)
            cols.append(col)
            lengths.append(col.shape[0])

    if len(set(lengths)) != 1:
        detail = {feature_names[i]: lengths[i] for i in range(len(feature_names))}
        raise ValueError(f"Inconsistent ntrials across features: {detail}")

    support_features = np.concatenate(cols, axis=1)
    print(f"[{pred}] sup feat shape:", support_features.shape)
    return support_features, feature_names


def load_pop_vec(session_id: str = "session_1091039376", predict: str = "tonic_A"):
    filename = _resolve_pop_vector_path(predict)

    with h5py.File(filename, "r") as f:
        if session_id not in f.keys():
            raise KeyError(f"Session {session_id} not found. Available: {list(f.keys())}")
        pop_vec = np.array(f[session_id]["pop_vector"])
        if pop_vec.ndim != 2:
            raise ValueError(f"Expected 2D array [ntrials, ncells], got shape {pop_vec.shape}")
    print("pop vec shape:", pop_vec.shape)
    return pop_vec


def load_pop_vec_mean(session_id: str = "session_1091039376", predict: str = "tonic_A"):
    filename = _resolve_pop_vector_path(predict)

    with h5py.File(filename, "r") as f:
        if session_id not in f.keys():
            raise KeyError(f"Session {session_id} not found. Available: {list(f.keys())}")
        pop_vec = np.array(f[session_id]["pop_vector"])
        if pop_vec.ndim != 2:
            raise ValueError(f"Expected 2D array [ntrials, ncells], got shape {pop_vec.shape}")
        pop_mean = pop_vec.mean(axis=1)
    print("pop vec mean shape:", pop_mean.shape)
    return pop_mean


def load_image_trial(
    session_id: str = "session_1091039376",
    filename: str = (
        "./data/Model_features_predicting_neural_data/"
        "Predicting_burst_B/hit_go_imgs_initial_changed.h5"
    ),
):
    with h5py.File(filename, "r") as f:
        if session_id not in f.keys():
            raise KeyError(f"Session {session_id} not found. Available: {list(f.keys())}")

        session_group = f[session_id]
        if "hit_go_imgs" not in session_group.keys():
            raise KeyError(
                f"'hit_go_imgs' not found in {session_id}. "
                f"Available keys: {list(session_group.keys())}"
            )

        imgs_id = np.array(session_group["hit_go_imgs"])
        if imgs_id.ndim != 2:
            raise ValueError(f"Unexpected shape for hit_go_imgs: {imgs_id.shape}")

        imgs_id = imgs_id.T
        imgs_id = np.array(
            [
                [
                    str(x.decode("utf-8")) if isinstance(x, (bytes, np.bytes_)) else str(x)
                    for x in row
                ]
                for row in imgs_id
            ],
            dtype=str,
        )

    print("image trial shape:", imgs_id.shape)
    return imgs_id


def load_image(path: str = "./data/exp_img/Set"):
    files = sorted([f for f in os.listdir(path) if f.startswith("image_") and f.endswith(".npy")])
    if not files:
        raise FileNotFoundError(f"No 'image_*.npy' files found in {path}")

    X_list, y_list = [], []
    for f in files:
        file_path = os.path.join(path, f)
        img = np.load(file_path)
        if img.ndim != 2:
            raise ValueError(f"Unexpected image shape {img.shape} in file {f}. Expected (1200, 1920).")

        X_list.append(img[None, ...])
        y_list.append(f[len("image_") : -len(".npy")])

    X = np.concatenate(X_list, axis=0)
    y = np.array(y_list)
    print(f"Loaded {len(files)} images: X shape = {X.shape}, y shape = {y.shape}")
    return X, y


def build_vit_embeddings_from_X(
    X: np.ndarray,
    y: np.ndarray,
    cache_path: str = DEFAULT_EMB_CACHE_PATH,
    batch_size: int = 1,
    l2_normalize: bool = True,
) -> Dict[str, np.ndarray]:
    if os.path.exists(cache_path):
        with np.load(cache_path) as npz:
            return {k: npz[k] for k in npz.files}

    try:
        import torch
        import torch.nn as nn
        from PIL import Image
        from torchvision import models, transforms
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Building image embeddings requires torch, torchvision, and pillow "
            f"when cache is missing. Missing module: {exc.name}. "
            f"Expected cache path: {cache_path}"
        ) from exc

    # Use a conservative CPU-only path here. In this environment, batched execution
    # has shown unstable behavior for scrambled-image embedding generation.
    torch.set_num_threads(1)
    device = torch.device("cpu")
    weights = models.ViT_B_16_Weights.IMAGENET1K_V1
    model = models.vit_b_16(weights=weights)
    model.heads = nn.Identity()
    model.eval().to(device)

    preprocess = transforms.Compose(
        [
            transforms.Lambda(lambda x: Image.fromarray(x)),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Lambda(lambda im: im.convert("RGB")),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )

    embs = []
    with torch.no_grad():
        for i, img in enumerate(X):
            tensor = preprocess(img).unsqueeze(0).to(device)
            out = model(tensor)
            if l2_normalize:
                out = torch.nn.functional.normalize(out, p=2, dim=1)
            embs.append(out.cpu().numpy())
            print(f"Embedded image {i + 1}/{len(X)}")

    embs = np.concatenate(embs, axis=0)
    data = {label: embs[i] for i, label in enumerate(y)}

    cache_path_obj = Path(cache_path)
    cache_path_obj.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path_obj, **data)
    print(f"Saved embeddings to {cache_path_obj} ({len(data)} items)")
    return data


@dataclass(frozen=True)
class ControlConfig:
    familiar_mode: str = "none"
    deviant_mode: str = "none"
    random_state: int = 42

    def validate(self) -> None:
        for role, mode in (
            ("familiar", self.familiar_mode),
            ("deviant", self.deviant_mode),
        ):
            if mode not in VALID_CONTROL_MODES:
                raise ValueError(
                    f"Unsupported {role}_mode={mode}. "
                    f"Expected one of {sorted(VALID_CONTROL_MODES)}."
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


def _default_xgb_params(random_state: int = 42) -> Dict:
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


def _stable_seed(*parts: object) -> int:
    text = "::".join(str(p) for p in parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**32 - 1)


def _role_seed(
    control_config: ControlConfig,
    session_id: str,
    role: str,
    mode: str,
    label: Optional[str] = None,
) -> int:
    pieces = [control_config.random_state, session_id, role, mode]
    if label is not None:
        pieces.append(label)
    return _stable_seed(*pieces)


def _scramble_pixels(img: np.ndarray, seed: int) -> np.ndarray:
    flat = np.asarray(img).reshape(-1).copy()
    rng = np.random.default_rng(seed)
    perm = rng.permutation(flat.size)
    return flat[perm].reshape(img.shape)


def _build_gaussian_role_embeddings(
    session_id: str,
    unique_pairs: np.ndarray,
    role: str,
    control_config: ControlConfig,
    d_img: int = DEFAULT_TRIAL_EMBED_DIM,
) -> tuple[np.ndarray, int]:
    if role not in {"familiar", "deviant"}:
        raise ValueError(f"Unsupported role '{role}'.")
    if d_img < 1:
        raise ValueError(f"Expected d_img >= 1, got {d_img}.")

    seed = _role_seed(control_config, session_id, role, "gaussian_trials")
    rng = np.random.default_rng(seed)
    row_embeddings = rng.standard_normal((unique_pairs.shape[0], d_img)).astype(np.float32)
    return row_embeddings, int(seed)


def _validate_plot_metric(metric: str) -> str:
    metric_norm = metric.lower()
    if metric_norm not in VALID_PLOT_METRICS:
        raise ValueError(
            f"Unsupported plot metric '{metric}'. Expected one of {sorted(VALID_PLOT_METRICS)}."
        )
    return metric_norm


def _default_color_limits_for_metric(metric: str) -> tuple[float, float]:
    metric = _validate_plot_metric(metric)
    if metric == "pearson":
        return (-1.0, 1.0)
    return (-3.0, 1.0)


def _safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.size != y.size:
        raise ValueError(f"Pearson inputs must have the same size, got {x.size} and {y.size}.")
    if x.size < 2:
        return float("nan")
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _nanmean_or_nan(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    if not finite.any():
        return float("nan")
    return float(values[finite].mean())


def _compute_oof_predictions(
    X: np.ndarray,
    y: np.ndarray,
    xgb_params: Dict,
    n_splits: int = 5,
    random_state: int = 42,
) -> np.ndarray:
    n_samples = X.shape[0]
    n_splits_eff = min(n_splits, n_samples)
    if n_splits_eff < 2:
        raise ValueError(f"Too few samples for CV: {n_samples}")

    kf = KFold(n_splits=n_splits_eff, shuffle=True, random_state=random_state)
    n_targets = y.shape[1]
    y_pred_oof = np.zeros_like(y)

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]
        for d in range(n_targets):
            model = XGBRegressor(**xgb_params)
            model.fit(X_train, y_train[:, d])
            y_pred_oof[test_idx, d] = model.predict(X_test)

    return y_pred_oof


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


def _summarize_session_evaluation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    r2_vals: np.ndarray,
    pearson_vals: np.ndarray,
) -> dict[str, float]:
    flat_true = np.asarray(y_true).reshape(-1)
    flat_pred = np.asarray(y_pred).reshape(-1)
    return {
        "r2_mean_per_dim": _nanmean_or_nan(r2_vals),
        "pearson_mean_per_dim": _nanmean_or_nan(pearson_vals),
        "r2_flattened": float(r2_score(flat_true, flat_pred)),
        "pearson_flattened": _safe_pearson(flat_true, flat_pred),
    }


def _load_cached_embedding_dict(cache_path: Union[str, Path]) -> Dict[str, np.ndarray]:
    with np.load(cache_path) as npz:
        return {k: npz[k] for k in npz.files}


def _load_trial_structure(session_id: str) -> tuple[np.ndarray, list[np.ndarray], int]:
    imgs_id = load_image_trial(session_id=session_id)
    unique_pairs, inverse_indices = np.unique(imgs_id, axis=0, return_inverse=True)

    duplicate_groups = []
    for k in range(unique_pairs.shape[0]):
        duplicate_groups.append(np.where(inverse_indices == k)[0])

    return unique_pairs, duplicate_groups, int(imgs_id.shape[0])


def _aggregate_trial_targets(
    session_id: str,
    pred: str,
    pred_mean: bool,
    duplicate_groups: list[np.ndarray],
    expected_ntrials: int,
) -> np.ndarray:
    if pred_mean:
        pop_raw = load_pop_vec_mean(session_id=session_id, predict=pred)
    else:
        pop_raw = load_pop_vec(session_id=session_id, predict=pred)

    pop_raw = np.asarray(pop_raw)
    if pop_raw.ndim == 1:
        pop_raw = pop_raw.reshape(-1, 1)
    if pop_raw.shape[0] != expected_ntrials:
        raise ValueError(
            f"pop_vec ntrials {pop_raw.shape[0]} != image trials {expected_ntrials}"
        )

    pop_unique = np.zeros((len(duplicate_groups), pop_raw.shape[1]), dtype=pop_raw.dtype)
    for k, idx in enumerate(duplicate_groups):
        pop_unique[k] = pop_raw[idx].mean(axis=0)
    return pop_unique


def _aggregate_support_features(
    session_id: str,
    pred: str,
    duplicate_groups: list[np.ndarray],
    expected_ntrials: int,
) -> tuple[np.ndarray, list[str]]:
    sup_raw, sup_feat_names = load_sup_feat(session_id=session_id, pred=pred)
    sup_raw = np.asarray(sup_raw)
    if sup_raw.shape[0] != expected_ntrials:
        raise ValueError(
            f"supporting features ntrials {sup_raw.shape[0]} != image trials {expected_ntrials}"
        )
    sup_unique = np.zeros((len(duplicate_groups), sup_raw.shape[1]), dtype=sup_raw.dtype)
    for k, idx in enumerate(duplicate_groups):
        sup_unique[k] = sup_raw[idx].mean(axis=0)
    return sup_unique, sup_feat_names


def _build_original_embeddings() -> Dict[str, np.ndarray]:
    cache_key = f"original::{DEFAULT_EMB_CACHE_PATH}"
    if cache_key in _IN_MEMORY_EMBEDDING_CACHE:
        return _IN_MEMORY_EMBEDDING_CACHE[cache_key]

    if Path(DEFAULT_EMB_CACHE_PATH).exists():
        emb = _load_cached_embedding_dict(DEFAULT_EMB_CACHE_PATH)
        _IN_MEMORY_EMBEDDING_CACHE[cache_key] = emb
        return emb

    X_imgs, y_labels = load_image()
    emb = build_vit_embeddings_from_X(X_imgs, y_labels, cache_path=DEFAULT_EMB_CACHE_PATH)
    _IN_MEMORY_EMBEDDING_CACHE[cache_key] = emb
    return emb


def _build_scrambled_embeddings_for_role(
    role: str,
    control_config: ControlConfig,
    cache_root: str = "./data/embeddings/shap_control",
) -> Dict[str, np.ndarray]:
    cache_root_path = Path(cache_root)
    cache_root_path.mkdir(parents=True, exist_ok=True)
    cache_path = cache_root_path / f"vit_b16_{role}_{control_config.tag}.npz"
    cache_key = f"{role}::{cache_path}"

    if cache_key in _IN_MEMORY_EMBEDDING_CACHE:
        return _IN_MEMORY_EMBEDDING_CACHE[cache_key]

    if cache_path.exists():
        emb = _load_cached_embedding_dict(cache_path)
        _IN_MEMORY_EMBEDDING_CACHE[cache_key] = emb
        return emb

    X_imgs, y_labels = load_image()
    X_scrambled = np.empty_like(X_imgs)
    for idx, label in enumerate(y_labels):
        seed = _stable_seed(control_config.random_state, role, "scramble_pixels", str(label))
        X_scrambled[idx] = _scramble_pixels(X_imgs[idx], seed=seed)

    emb = build_vit_embeddings_from_X(
        X_scrambled,
        y_labels,
        cache_path=str(cache_path),
    )
    _IN_MEMORY_EMBEDDING_CACHE[cache_key] = emb
    return emb


def _build_role_to_embedding(
    control_config: ControlConfig,
    cache_root: str = "./data/embeddings/shap_control",
) -> dict[str, Dict[str, np.ndarray]]:
    control_config.validate()
    original = _build_original_embeddings()

    role_to_emb = {
        "familiar": original,
        "deviant": original,
    }

    if control_config.familiar_mode == "scramble_pixels":
        role_to_emb["familiar"] = _build_scrambled_embeddings_for_role(
            role="familiar",
            control_config=control_config,
            cache_root=cache_root,
        )
    if control_config.deviant_mode == "scramble_pixels":
        role_to_emb["deviant"] = _build_scrambled_embeddings_for_role(
            role="deviant",
            control_config=control_config,
            cache_root=cache_root,
        )

    return role_to_emb


def _apply_shuffle_control(
    img_emb_unique: np.ndarray,
    d_img: int,
    session_id: str,
    control_config: ControlConfig,
) -> dict[str, list[int]]:
    shuffle_meta: dict[str, list[int]] = {}

    if control_config.familiar_mode == "shuffle_trials":
        rng = np.random.default_rng(
            _role_seed(control_config, session_id, "familiar", "shuffle_trials")
        )
        perm = rng.permutation(img_emb_unique.shape[0])
        img_emb_unique[:, :d_img] = img_emb_unique[perm, :d_img]
        shuffle_meta["familiar_trial_permutation"] = perm.tolist()

    if control_config.deviant_mode == "shuffle_trials":
        rng = np.random.default_rng(
            _role_seed(control_config, session_id, "deviant", "shuffle_trials")
        )
        perm = rng.permutation(img_emb_unique.shape[0])
        img_emb_unique[:, d_img : 2 * d_img] = img_emb_unique[perm, d_img : 2 * d_img]
        shuffle_meta["deviant_trial_permutation"] = perm.tolist()

    return shuffle_meta


def prepare_data_with_control(
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
        # Gaussian-only controls still need the embedding dimensionality to match
        # the real image embedding space used elsewhere in the pipeline.
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

    img_emb_unique = np.zeros((unique_pairs.shape[0], d_img * 2), dtype=example_emb.dtype)

    for k, (img_familiar, img_deviant) in enumerate(unique_pairs):
        if control_config.familiar_mode == "gaussian_trials":
            familiar_emb = row_gaussian_embeddings["familiar"][k]
        else:
            if role_to_emb is None:
                raise RuntimeError("Image embeddings were not loaded for familiar role.")
            if img_familiar not in role_to_emb["familiar"]:
                raise KeyError(f"Image label '{img_familiar}' not found in familiar embedding dict.")
            familiar_emb = role_to_emb["familiar"][img_familiar]

        if control_config.deviant_mode == "gaussian_trials":
            deviant_emb = row_gaussian_embeddings["deviant"][k]
        else:
            if role_to_emb is None:
                raise RuntimeError("Image embeddings were not loaded for deviant role.")
            if img_deviant not in role_to_emb["deviant"]:
                raise KeyError(f"Image label '{img_deviant}' not found in deviant embedding dict.")
            deviant_emb = role_to_emb["deviant"][img_deviant]

        img_emb_unique[k] = np.concatenate([familiar_emb, deviant_emb], axis=0)

    shuffle_meta = _apply_shuffle_control(
        img_emb_unique=img_emb_unique,
        d_img=d_img,
        session_id=session_id,
        control_config=control_config,
    )

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
        "shuffle_meta": shuffle_meta,
        "gaussian_meta": gaussian_meta,
    }

    print(
        f"Prepared controlled data for session {session_id} / pred={pred} / "
        f"pred_mean={pred_mean} / sup_feat={need_sup_feat}"
    )
    print(f"  control: {control_config.describe()}")
    print(f"  raw trials: {meta['ntrials_raw']}, unique trials: {meta['ntrials_unique']}")
    print(f"  X shape: {X.shape}, y shape: {y.shape}")

    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32), meta


def _make_result_tag(
    pred: str,
    pred_mean: bool,
    need_sup_feat: bool,
    control_config: ControlConfig,
) -> str:
    pred_tag = "mean" if pred_mean else "full"
    sup_tag = "sup" if need_sup_feat else "nosup"
    return f"{pred}_{pred_tag}_{sup_tag}_{control_config.tag}"


def _make_results_dir(
    results_root: str,
    pred: str,
    control_config: ControlConfig,
) -> Path:
    path = Path(results_root) / pred / control_config.tag
    path.mkdir(parents=True, exist_ok=True)
    return path


def _default_plot_path_from_csv(csv_path: Union[str, Path], color_metric: str) -> Path:
    csv_path = Path(csv_path)
    return csv_path.with_name(f"{csv_path.stem}_colorby-{color_metric}.pdf")


def _resolve_plot_inputs(
    input_path: Union[str, Path],
) -> tuple[Path, Optional[str], Optional[str], Optional[tuple[float, float]]]:
    input_path = Path(input_path)
    if input_path.suffix.lower() != ".json":
        return input_path, None, None, None

    with open(input_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    files = metadata.get("files", {})
    per_dim_csv = files.get("per_dim_csv")
    if not per_dim_csv:
        raise KeyError(f"'per_dim_csv' not found in metadata file {input_path}")

    plot_defaults = metadata.get("plot_defaults", {})
    color_limits_raw = plot_defaults.get("color_limits")
    color_limits = None
    if color_limits_raw is not None:
        color_limits = tuple(color_limits_raw)
        if len(color_limits) != 2:
            raise ValueError(f"Invalid color_limits in metadata file {input_path}: {color_limits}")
        color_limits = (float(color_limits[0]), float(color_limits[1]))

    return (
        Path(per_dim_csv),
        plot_defaults.get("base_title"),
        plot_defaults.get("color_metric"),
        color_limits,
    )


def plot_familiar_vs_deviant(
    input_path: Union[str, Path],
    save_path: Optional[Union[str, Path]] = None,
    color_metric: Optional[str] = None,
    title: Optional[str] = None,
    dpi: int = 160,
    color_limits: Optional[tuple[float, float]] = None,
) -> Path:
    csv_path, default_title, default_metric, default_limits = _resolve_plot_inputs(input_path)
    metric = _validate_plot_metric(color_metric or default_metric or "r2")
    limits = color_limits or default_limits or _default_color_limits_for_metric(metric)

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"No data found in {csv_path}")
    if metric not in df.columns:
        raise KeyError(f"Column '{metric}' not found in {csv_path}")

    x = df["familiar_importance"].to_numpy()
    y = df["deviant_importance"].to_numpy()
    colors = df[metric].to_numpy(dtype=np.float64)
    valid = np.isfinite(colors)

    plt.figure(figsize=(6.5, 6.5), dpi=dpi)
    if valid.any():
        sc = plt.scatter(
            x[valid],
            y[valid],
            c=colors[valid],
            cmap=DEFAULT_PLOT_CMAP,
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
    plot_title = title or default_title or "Familiar vs Deviant importance"
    plt.title(plot_title)
    plt.grid(alpha=0.25, linestyle="--")
    plt.tight_layout()

    if save_path is None:
        save_path = _default_plot_path_from_csv(csv_path, metric)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", format="pdf")
    plt.close()
    print(f"[Saved] {save_path}")
    return save_path


def run_shap_importance_control(
    session_ids,
    pred: str = "tonic_A",
    pred_mean: bool = False,
    need_sup_feat: bool = False,
    n_splits: int = 5,
    random_state: int = 42,
    xgb_params: Optional[Dict] = None,
    control_config: Optional[ControlConfig] = None,
    results_root: str = "./results/shap_control",
    save_shap_values: bool = False,
    plot_metric: str = "r2",
    plot_color_limits: Optional[tuple[float, float]] = None,
):
    if control_config is None:
        control_config = ControlConfig(random_state=random_state)
    control_config.validate()
    plot_metric = _validate_plot_metric(plot_metric)
    plot_color_limits = plot_color_limits or _default_color_limits_for_metric(plot_metric)

    if xgb_params is None:
        xgb_params = _default_xgb_params(random_state=random_state)

    results_dir = _make_results_dir(results_root=results_root, pred=pred, control_config=control_config)
    tag = _make_result_tag(
        pred=pred,
        pred_mean=pred_mean,
        need_sup_feat=need_sup_feat,
        control_config=control_config,
    )

    per_dim_rows = []
    evaluation_rows = []
    session_metadata_rows = []
    shap_dir = results_dir / f"shap_values_{tag}" if save_shap_values else None
    if shap_dir is not None:
        shap_dir.mkdir(parents=True, exist_ok=True)

    for session_id in session_ids:
        X, y, meta = prepare_data_with_control(
            session_id=session_id,
            pred=pred,
            pred_mean=pred_mean,
            need_sup_feat=need_sup_feat,
            control_config=control_config,
        )

        d_img = int(meta["img_feature_dim"])
        y_pred_oof = _compute_oof_predictions(
            X,
            y,
            xgb_params=xgb_params,
            n_splits=n_splits,
            random_state=random_state,
        )
        r2_vals, pearson_vals = _compute_prediction_metrics_per_dim(y, y_pred_oof)
        session_eval = _summarize_session_evaluation(
            y_true=y,
            y_pred=y_pred_oof,
            r2_vals=r2_vals,
            pearson_vals=pearson_vals,
        )
        evaluation_rows.append(
            {
                "session_id": session_id,
                "ntrials_raw": int(meta["ntrials_raw"]),
                "ntrials_unique": int(meta["ntrials_unique"]),
                "target_dim": int(meta["target_dim"]),
                **session_eval,
            }
        )
        session_metadata_rows.append(
            {
                "session_id": session_id,
                "ntrials_raw": int(meta["ntrials_raw"]),
                "ntrials_unique": int(meta["ntrials_unique"]),
                "img_feature_dim": int(meta["img_feature_dim"]),
                "input_feature_dim": int(meta["input_feature_dim"]),
                "target_dim": int(meta["target_dim"]),
                "sup_feat_names": list(meta["sup_feat_names"] or []),
                "shuffle_controls": sorted(meta["shuffle_meta"].keys()),
            }
        )

        for d in range(y.shape[1]):
            model = XGBRegressor(**xgb_params)
            model.fit(X, y[:, d])
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            mean_abs = np.mean(np.abs(shap_values), axis=0)

            familiar_imp = float(mean_abs[:d_img].sum())
            deviant_imp = float(mean_abs[d_img : 2 * d_img].sum())
            support_imp = float(mean_abs[2 * d_img :].sum()) if mean_abs.shape[0] > 2 * d_img else 0.0
            per_dim_rows.append(
                {
                    "session_id": session_id,
                    "dim": d,
                    "familiar_importance": familiar_imp,
                    "deviant_importance": deviant_imp,
                    "support_importance": support_imp,
                    "r2": float(r2_vals[d]),
                    "pearson": float(pearson_vals[d]),
                }
            )

            if shap_dir is not None:
                np.save(shap_dir / f"shap_values_{session_id}_dim{d}.npy", shap_values)

    df_per_dim = pd.DataFrame(per_dim_rows)
    df_evaluation = pd.DataFrame(evaluation_rows)

    per_dim_path = results_dir / f"per_dim_{tag}.csv"
    evaluation_path = results_dir / f"evaluation_per_session_{tag}.csv"
    metadata_path = results_dir / f"metadata_{tag}.json"
    plot_path = _default_plot_path_from_csv(per_dim_path, plot_metric)

    df_per_dim.to_csv(per_dim_path, index=False)
    df_evaluation.to_csv(evaluation_path, index=False)

    plot_title = (
        f"Familiar vs Deviant importance\n"
        f"{control_config.describe()} / pred={pred}"
    )
    plot_familiar_vs_deviant(
        input_path=per_dim_path,
        save_path=plot_path,
        color_metric=plot_metric,
        title=plot_title,
        color_limits=plot_color_limits,
    )

    metadata = {
        "pred": pred,
        "pred_mean": pred_mean,
        "need_sup_feat": need_sup_feat,
        "n_splits": n_splits,
        "random_state": random_state,
        "session_ids": list(session_ids),
        "control_config": asdict(control_config),
        "xgb_params": xgb_params,
        "results_dir": str(results_dir),
        "session_metadata": session_metadata_rows,
        "plot_defaults": {
            "color_metric": plot_metric,
            "color_limits": [float(plot_color_limits[0]), float(plot_color_limits[1])],
            "base_title": plot_title,
        },
        "files": {
            "per_dim_csv": str(per_dim_path),
            "evaluation_csv": str(evaluation_path),
            "plot_pdf": str(plot_path),
        },
    }
    if shap_dir is not None:
        metadata["files"]["shap_values_dir"] = str(shap_dir)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("Saved SHAP control outputs:")
    print(f"  {per_dim_path}")
    print(f"  {evaluation_path}")
    print(f"  {metadata_path}")
    print(f"  {plot_path}")

    return {
        "df_per_dim": df_per_dim,
        "df_evaluation": df_evaluation,
        "metadata": metadata,
        "paths": {
            "per_dim_csv": per_dim_path,
            "evaluation_csv": evaluation_path,
            "metadata_json": metadata_path,
            "plot_pdf": plot_path,
        },
    }


def run_plot():
    plot_familiar_vs_deviant(
        input_path="results/shap_control/burst_B/fam-none_dev-none_seed42/per_dim_burst_B_full_nosup_fam-none_dev-none_seed42.csv",
        color_metric="pearson",
    )


if __name__ == "__main__":
    # Examples:
    # 1) scramble familiar image: familiar_mode="scramble_pixels"
    # 2) shuffle familiar image across trials: familiar_mode="shuffle_trials"
    # 3) scramble deviant image: deviant_mode="scramble_pixels"
    # 4) shuffle deviant image across trials: deviant_mode="shuffle_trials"
    # 5) replace familiar embeddings with standard Gaussian noise: familiar_mode="gaussian_trials"
    # 6) replace deviant embeddings with standard Gaussian noise: deviant_mode="gaussian_trials"
    
    control_config = ControlConfig(
        familiar_mode="gaussian_trials",  # none | scramble_pixels | shuffle_trials | gaussian_trials
        deviant_mode="gaussian_trials",  # none | scramble_pixels | shuffle_trials | gaussian_trials
        random_state=42,
    )

    run_shap_importance_control(
        session_ids=SESSION_IDS[:10],  # tonic_A: 4, burst_B: 6
        pred="burst_B",
        pred_mean=False,
        need_sup_feat=False,
        n_splits=5,
        random_state=42,
        control_config=control_config,
        results_root="./results/shap_control",
        save_shap_values=False,
        plot_metric="pearson",  # r2, pearson
    )

    # run_plot()
