from pathlib import Path
from typing import Dict

import numpy as np


def _load_embedding_cache(cache_path: Path) -> Dict[str, np.ndarray]:
    with np.load(cache_path) as npz:
        return {k: npz[k] for k in npz.files}


def build_vit_embeddings_from_X_layer(
    X,
    y,
    layer_index: int = 0,
    token_pool: str = "cls",
    cache_path=None,
    batch_size: int = 8,
    l2_normalize: bool = True,
) -> Dict[str, np.ndarray]:
    """Build ViT-B/16 embeddings from a selected encoder block."""
    if cache_path is None:
        cache_path = f"./data/embeddings/vit_b16_layer{layer_index}_{token_pool}.npz"
    cache_path = Path(cache_path)

    if cache_path.exists():
        return _load_embedding_cache(cache_path)

    try:
        import torch
        from PIL import Image
        from torchvision import models, transforms
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Building early-layer ViT embeddings requires torch, torchvision, and pillow "
            f"when cache is missing. Missing module: {exc.name}. "
            f"Expected cache path: {cache_path}"
        ) from exc

    token_pool = str(token_pool).lower()
    if token_pool not in {"cls", "mean"}:
        raise ValueError("token_pool must be 'cls' or 'mean'.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = models.ViT_B_16_Weights.IMAGENET1K_V1
    model = models.vit_b_16(weights=weights).to(device).eval()

    num_layers = len(model.encoder.layers)
    layer_index = int(layer_index)
    if not 0 <= layer_index < num_layers:
        raise ValueError(f"layer_index {layer_index} out of range [0, {num_layers - 1}].")

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

    def forward_to_layer(batch_tensor):
        x = model._process_input(batch_tensor)
        batch_class_token = model.class_token.expand(x.shape[0], -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = x + model.encoder.pos_embedding
        x = model.encoder.dropout(x)

        for idx, block in enumerate(model.encoder.layers):
            x = block(x)
            if idx == layer_index:
                break

        x = model.encoder.ln(x)
        if token_pool == "cls":
            pooled = x[:, 0]
        else:
            pooled = x[:, 1:].mean(dim=1)

        if l2_normalize:
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return pooled

    labels = [str(label) for label in y]
    tensors = [preprocess(img) for img in X]
    embeddings = []

    with torch.no_grad():
        for start in range(0, len(tensors), int(batch_size)):
            batch = torch.stack(tensors[start : start + int(batch_size)], dim=0).to(device)
            embeddings.append(forward_to_layer(batch).cpu().numpy())

    embeddings = np.concatenate(embeddings, axis=0)
    data = {label: embeddings[idx] for idx, label in enumerate(labels)}

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, **data)
    print(f"Saved layer {layer_index} ({token_pool}) embeddings to {cache_path} ({len(data)} items)")
    return data
