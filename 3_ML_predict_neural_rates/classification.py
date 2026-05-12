import os
import random
from statistics import mean, pstdev

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.models import vit_b_16, ViT_B_16_Weights, resnet18, ResNet18_Weights, vgg16, VGG16_Weights
from torchvision.models.vision_transformer import VisionTransformer
from torchvision.utils import save_image


# -------------- utils --------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def accuracy(logits, y):
    preds = logits.argmax(dim=1)
    return (preds == y).float().mean().item()


def run_one_model(model_name: str,
                  build_model_fn,
                  train_tfms,
                  test_tfms,
                  epochs=10,
                  batch_size=128,
                  lr=3e-4,
                  trials=3,
                  device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=== {model_name} on device: {device} ===")

    test_scores = []

    for trial in range(trials):
        seed = 2025 + trial
        set_seed(seed)

        # dataset per trial to keep augmentation randomness isolated
        trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=train_tfms)
        testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=test_tfms)

        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

        model = build_model_fn().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)

        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            running_acc = 0.0
            for x, y in trainloader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                optimizer.zero_grad()
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * x.size(0)
                running_acc += accuracy(logits, y) * x.size(0)

            train_loss = running_loss / len(trainset)
            train_acc = running_acc / len(trainset)

            # eval
            model.eval()
            test_acc_sum = 0.0
            with torch.no_grad():
                for x, y in testloader:
                    x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                    logits = model(x)
                    test_acc_sum += accuracy(logits, y) * x.size(0)
            test_acc = test_acc_sum / len(testset)

            print(f"[{model_name}] Trial {trial + 1}/{trials} Epoch {epoch + 1}/{epochs} "
                  f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} test_acc={test_acc:.4f}")

        test_scores.append(test_acc)

    mu = mean(test_scores)
    sd = pstdev(test_scores)  # population std
    print(f"\n*** {model_name} 3 trial result: "
          f"mean={mu:.4f} std={sd:.4f} "
          f"(scores={', '.join(f'{s:.4f}' for s in test_scores)})")
    return model


# -------------- models --------------
class Perceptron(nn.Module):
    # single linear layer on flattened pixels
    def __init__(self, in_shape=(3, 32, 32), num_classes=10):
        super().__init__()
        c, h, w = in_shape
        self.classifier = nn.Linear(c * h * w, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32 -> 16
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16 -> 8
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),  # B x 256 x 1 x 1
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def build_vit(num_classes=10, pretrained=True, freeze_backbone=True):
    weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
    model = vit_b_16(weights=weights)
    in_dim = model.heads.head.in_features
    model.heads.head = nn.Linear(in_dim, num_classes)

    if freeze_backbone:
        for name, p in model.named_parameters():
            if not name.startswith("heads.head"):
                p.requires_grad = False
    return model


def build_resnet18(num_classes=10, pretrained=True, freeze_backbone=True):
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)
    in_dim = model.fc.in_features
    model.fc = nn.Linear(in_dim, num_classes)

    if freeze_backbone:
        for name, p in model.named_parameters():
            if not name.startswith("fc."):
                p.requires_grad = False
    return model


def build_vgg16(num_classes=10, pretrained=True, freeze_backbone=True):
    weights = VGG16_Weights.IMAGENET1K_V1 if pretrained else None
    model = vgg16(weights=weights)
    in_dim = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_dim, num_classes)

    if freeze_backbone:
        for name, p in model.named_parameters():
            if not name.startswith("classifier.6"):
                p.requires_grad = False
    return model


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        def fwd_hook(module, inp, out):
            self.activations = out.detach()

        def bwd_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.h1 = target_layer.register_forward_hook(fwd_hook)
        self.h2 = target_layer.register_full_backward_hook(bwd_hook)

    def remove(self):
        self.h1.remove()
        self.h2.remove()

    def __call__(self, images, class_idx=None):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(images)  # [B, K]
        if class_idx is None:
            class_idx = logits.argmax(dim=1)
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, class_idx.view(-1, 1), 1.0)
        loss = (one_hot * logits).sum()
        loss.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=images.shape[-2:], mode="bilinear", align_corners=False)
        cam_min = cam.amin(dim=(2, 3), keepdim=True)
        cam_max = cam.amax(dim=(2, 3), keepdim=True)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        return cam  # [B,1,H,W]


def run(model_name):
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cuda:4"
    trials = 3
    epochs_perceptron = 15
    epochs_cnn = 20
    epochs_vit = 20
    bs = 128
    # Transforms
    # Perceptron and CNN use 32 size, standard CIFAR10 augmentation
    train_tfms_32 = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465),
                    (0.2023, 0.1994, 0.2010)),
    ])
    test_tfms_32 = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465),
                    (0.2023, 0.1994, 0.2010)),
    ])
    # ViT expects 224 resolution
    train_tfms_224 = T.Compose([
        T.Resize(224, antialias=True),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=(0.5, 0.5, 0.5),
                    std=(0.5, 0.5, 0.5)),
    ])
    test_tfms_224 = T.Compose([
        T.Resize(224, antialias=True),
        T.ToTensor(),
        T.Normalize(mean=(0.5, 0.5, 0.5),
                    std=(0.5, 0.5, 0.5)),
    ])
    # # Perceptron
    if model_name == "Perceptron":
        model = run_one_model(
            "Perceptron",
            build_model_fn=lambda: Perceptron(in_shape=(3, 32, 32), num_classes=10),
            train_tfms=train_tfms_32,
            test_tfms=test_tfms_32,
            epochs=epochs_perceptron,
            batch_size=bs,
            lr=1e-3,
            trials=trials,
            device=device
        )
    elif model_name == "SimpleCNN":
        model = run_one_model(
            "SimpleCNN",
            build_model_fn=lambda: SimpleCNN(num_classes=10),
            train_tfms=train_tfms_32,
            test_tfms=test_tfms_32,
            epochs=20,
            batch_size=bs,
            lr=3e-4,
            trials=trials,
            device=device
        )
    elif model_name == "ViT":
        model = run_one_model(
            "ViT_B_16_head_only",
            build_model_fn=lambda: build_vit(num_classes=10, pretrained=True, freeze_backbone=True),
            train_tfms=train_tfms_224,
            test_tfms=test_tfms_224,
            epochs=20,  # Slightly longer training.
            batch_size=bs,
            lr=3e-4,
            trials=trials,
            device=device
        )
    elif model_name == "ResNet18":
        model = run_one_model(
            "ResNet18",
            build_model_fn=lambda: build_resnet18(num_classes=10, pretrained=True),
            train_tfms=train_tfms_224,  # Use 224 input.
            test_tfms=test_tfms_224,
            epochs=epochs_cnn,
            batch_size=bs,
            lr=3e-4,
            trials=trials,
            device=device
        )
    elif model_name == "VGG16":
        model = run_one_model(
            "VGG16",
            build_model_fn=lambda: build_vgg16(num_classes=10, pretrained=False),
            train_tfms=train_tfms_224,  # Use 224 input.
            test_tfms=test_tfms_224,
            epochs=20,
            batch_size=bs,
            lr=3e-3,
            trials=trials,
            device=device
        )
    else:
        raise ValueError()

    model.eval()

    if isinstance(model, SimpleCNN):
        testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=test_tfms_32)
        testloader = DataLoader(testset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
        disable_inplace_relu(model)  # Disable in-place ReLU for Grad-CAM hooks.
        save_gradcam_for_simplecnn(model, testloader, out_dir="./vis_gradcam", num_images=4, device=device)
    elif isinstance(model, VisionTransformer):
        testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=test_tfms_224)
        testloader = DataLoader(testset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
        save_vit_attention_rollout(model, testloader, out_dir="./vis_vit", num_images=4, device=device)


def save_gradcam_for_simplecnn(model, dataloader, out_dir="./vis_gradcam", num_images=32, device="cuda"):
    os.makedirs(out_dir, exist_ok=True)
    model = model.to(device).eval()
    # The final SimpleCNN convolution is Conv2d(128,256,3) inside features.
    # It is immediately before AdaptiveAvgPool2d.
    target_layer = None
    for m in model.features.modules():
        if isinstance(m, torch.nn.Conv2d):
            target_layer = m
    if target_layer is None:
        raise RuntimeError("No convolution layer found.")

    cam = GradCAM(model, target_layer)
    saved = 0
    for x, y in dataloader:
        x = x.to(device, non_blocking=True)
        cam_map = cam(x)  # [B,1,H,W]
        for i in range(x.size(0)):
            if saved >= num_images:
                cam.remove()
                return
            overlay = 0.55 * x[i:i + 1] + 0.45 * cam_map[i:i + 1].repeat(1, x.size(1), 1, 1)
            save_image(x[i], os.path.join(out_dir, f"img_{saved:03d}_orig.png"))
            save_image(overlay.clamp(0, 1), os.path.join(out_dir, f"img_{saved:03d}_gradcam.png"))
            saved += 1
    cam.remove()


def disable_inplace_relu(module: torch.nn.Module):
    for m in module.modules():
        if isinstance(m, torch.nn.ReLU) and getattr(m, "inplace", False):
            m.inplace = False


@torch.no_grad()
def _rollout_from_attn_list(attn_list):
    proc = []
    for A in attn_list:
        # A: [B, T, T] or [B, H, T, T].
        if A.dim() == 4:
            A = A.mean(dim=1)  # Average heads.
        T = A.size(-1)
        eye = torch.eye(T, device=A.device).unsqueeze(0)
        A = A + eye
        A = A / A.sum(dim=-1, keepdim=True)  # Row-normalize.
        proc.append(A)

    joint = proc[0]
    for k in range(1, len(proc)):
        joint = torch.bmm(proc[k], joint)
    return joint[:, 0]  # [B, T]


def _patch_mha_to_return_weights(model, per_head=True):
    """Force all MHA modules to return attention weights and keep restore handles."""
    import torch.nn as nn
    orig = {}
    for m in model.modules():
        if isinstance(m, nn.MultiheadAttention):
            orig[m] = m.forward
            def make_wrapped(orig_forward):
                def wrapped(*args, **kwargs):
                    kwargs["need_weights"] = True
                    # Per-head weights are preferred for rollout.
                    kwargs["average_attn_weights"] = False if per_head else True
                    return orig_forward(*args, **kwargs)
                return wrapped
            m.forward = make_wrapped(m.forward)
    return orig


def _restore_mha_forward(orig):
    for m, f in orig.items():
        m.forward = f


@torch.no_grad()
def save_vit_attention_rollout(model, dataloader, out_dir="./vis_vit", num_images=32, device="cuda"):
    if not isinstance(model, VisionTransformer):
        raise TypeError("model must be torchvision.models.vision_transformer.VisionTransformer")

    os.makedirs(out_dir, exist_ok=True)
    model = model.to(device).eval()

    # 1) Force MHA modules to return attention weights.
    orig_fwds = _patch_mha_to_return_weights(model, per_head=True)

    attn_list, handles = [], []

    def make_hook():
        def hook(module, inp, out):
            # out = (attn_output, attn_weights); attn_weights should not be None here.
            attn_weights = out[1]
            # Keep a defensive check for compatibility.
            if attn_weights is not None:
                attn_list.append(attn_weights.detach())
        return hook

    import torch.nn as nn
    for m in model.modules():
        if isinstance(m, nn.MultiheadAttention):
            handles.append(m.register_forward_hook(make_hook()))

    saved = 0
    try:
        # 2) Disable Flash and memory-efficient SDPA for this forward pass.
        try:
            from torch.backends.cuda import sdp_kernel
            sdpa_ctx = sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True)
        except Exception:
            class _Dummy:
                def __enter__(self): pass
                def __exit__(self, *args): pass
            sdpa_ctx = _Dummy()

        for x, y in dataloader:
            x = x.to(device, non_blocking=True)
            attn_list.clear()

            with sdpa_ctx:
                _ = model(x)

            if len(attn_list) == 0:
                raise RuntimeError(
                    "No attention weights were captured. Check the torch/torchvision "
                    "versions or custom acceleration paths that may suppress weights."
                )

            rollout = _rollout_from_attn_list(attn_list)  # [B, T]
            patch_scores = rollout[:, 1:]  # Drop cls.
            B, N = patch_scores.shape
            grid = int(math.sqrt(N))
            if grid * grid != N:
                raise ValueError(f"Patch count {N} is not a perfect square and cannot form a grid.")

            heat = patch_scores.reshape(B, 1, grid, grid)
            heat = F.interpolate(heat, size=x.shape[-2:], mode="bilinear", align_corners=False)
            # Normalize to 0..1.
            heat = (heat - heat.amin(dim=(2, 3), keepdim=True)) / (
                    heat.amax(dim=(2, 3), keepdim=True) - heat.amin(dim=(2, 3), keepdim=True) + 1e-8)

            # If x was ImageNet-normalized, denormalize it for visualization.
            def _maybe_denorm(img):
                # Expect 3 channels with ImageNet mean/std.
                if img.size(0) == 3:
                    mean = torch.tensor([0.485, 0.456, 0.406], device=img.device).view(3, 1, 1)
                    std  = torch.tensor([0.229, 0.224, 0.225], device=img.device).view(3, 1, 1)
                    return (img * std + mean).clamp(0, 1)
                return img.clamp(0, 1)

            for i in range(B):
                if saved >= num_images:
                    return
                x_vis = _maybe_denorm(x[i].detach().cpu())
                overlay = 0.55 * x_vis + 0.45 * heat[i, 0].detach().cpu().expand_as(x_vis)
                save_image(x_vis, os.path.join(out_dir, f"img_{saved:03d}_orig.png"))
                save_image(overlay.clamp(0, 1), os.path.join(out_dir, f"img_{saved:03d}_vit_attn.png"))
                saved += 1
    finally:
        for h in handles:
            h.remove()
        _restore_mha_forward(orig_fwds)


if __name__ == "__main__":
    run(model_name="VGG16")
