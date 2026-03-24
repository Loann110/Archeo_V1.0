# predictor_arcface.py
# DinoV2 backbone (timm) + ArcFace head predictor

import io
import os
import math
import threading
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
from PIL import Image, ImageOps

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
import timm


# =====================================================
# CHECKPOINT PATH (à ajuster selon le setup local, ou via env SEG_MODEL_PATH)
# =====================================================
CKPT_PATH = os.getenv("SEG_MODEL_PATH", r"C:\Users\loann\Desktop\projets\Archeo_V1\DINOV2\best_model.pt")


# -----------------------------
# Preprocessing (crop autour du motif + padding carré)
# -----------------------------
class AlphaCropAndMask:
    def __init__(self, alpha_threshold=5, pad=6, bg_rgb=(0, 0, 0), random_bg=False):
        self.alpha_threshold = alpha_threshold
        self.pad = pad
        self.bg_rgb = bg_rgb
        self.random_bg = random_bg

    def _pick_bg(self):
        return self.bg_rgb

    def __call__(self, img: Image.Image) -> Image.Image:
        img = img.convert("RGBA")
        alpha = np.array(img.split()[-1])

        coords = np.argwhere(alpha > self.alpha_threshold)
        if coords.size > 0:
            y0, x0 = coords.min(axis=0)
            y1, x1 = coords.max(axis=0) + 1

            y0 = max(0, y0 - self.pad)
            x0 = max(0, x0 - self.pad)
            y1 = min(alpha.shape[0], y1 + self.pad)
            x1 = min(alpha.shape[1], x1 + self.pad)

            img = img.crop((x0, y0, x1, y1))

        bg = Image.new("RGB", img.size, self._pick_bg())
        bg.paste(img.convert("RGB"), mask=img.split()[-1])
        return bg


class PadToSquare:
    def __init__(self, fill=(0, 0, 0), random_fill=False):
        self.fill = fill
        self.random_fill = random_fill

    def _pick_fill(self):
        return self.fill

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        s = max(w, h)
        pad_left = (s - w) // 2
        pad_right = s - w - pad_left
        pad_top = (s - h) // 2
        pad_bottom = s - h - pad_top
        return ImageOps.expand(
            img,
            border=(pad_left, pad_top, pad_right, pad_bottom),
            fill=self._pick_fill()
        )


# -----------------------------
# Backbone et utils DinoV2 + ArcFace
# -----------------------------
def make_backbone(model_name: str) -> nn.Module:
    # IMPORTANT: pretrained=False => évite tout téléchargement depuis HuggingFace Hub, 
    # et garantit que le modèle est bien chargé depuis le checkpoint local (qui doit lui-même être à jour)
    backbone = timm.create_model(
        model_name,
        pretrained=False,  # important pour éviter tout téléchargement depuis HuggingFace Hub
        num_classes=0,
        dynamic_img_size=True,
        dynamic_img_pad=True,
    )

    # DinoV2 patch_embed peut être configuré pour ne pas exiger une taille d'image fixe, 
    # ce qui est crucial pour notre cas d'usage où les images peuvent être de tailles variables après le crop+pad. On s'assure que ces options sont bien désactivées pour permettre la flexibilité.
    if hasattr(backbone, "patch_embed"):
        pe = backbone.patch_embed
        if hasattr(pe, "strict_img_size"):
            pe.strict_img_size = False
        if hasattr(pe, "img_size"):
            pe.img_size = None

    return backbone


def backbone_forward(backbone: nn.Module, x: torch.Tensor, pool: str = "cls") -> torch.Tensor:
    if hasattr(backbone, "forward_features"):
        feats = backbone.forward_features(x)
    else:
        feats = backbone(x)

    if isinstance(feats, dict):
        feats = feats.get("x", next(iter(feats.values())))
    if isinstance(feats, (tuple, list)):
        feats = feats[0]

    if feats.ndim == 3:  # (B, N, C)
        if pool == "mean":
            feats = feats[:, 1:].mean(dim=1)
        else:
            feats = feats[:, 0]
    return feats


# -----------------------------
# ArcFace head
# -----------------------------
class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super().__init__()
        self.s = float(s)
        self.m = float(m)
        self.easy_margin = easy_margin

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

    def forward(self, x, label=None):
        x = F.normalize(x, dim=1)
        W = F.normalize(self.weight, dim=1)
        cosine = F.linear(x, W).clamp(-1.0, 1.0)

        if label is None:
            return cosine * self.s

        sine = torch.sqrt((1.0 - cosine * cosine).clamp(0.0, 1.0))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)

        output = one_hot * phi + (1.0 - one_hot) * cosine
        return output * self.s


class DinoFineGrainedClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, num_classes: int, pool_mode: str = "cls", arc_s=30.0, arc_m=0.50):
        super().__init__()
        self.backbone = backbone
        self.pool_mode = pool_mode

        feat_dim = getattr(backbone, "num_features", None)
        if feat_dim is None:
            raise ValueError("Backbone has no num_features attribute.")

        self.feat_norm = nn.LayerNorm(feat_dim)
        self.arc = ArcMarginProduct(feat_dim, num_classes, s=arc_s, m=arc_m)

    def forward(self, x, y=None):
        feats = backbone_forward(self.backbone, x, pool=self.pool_mode)
        feats = self.feat_norm(feats)
        logits = self.arc(feats, label=y if self.training else None)
        return logits


# =====================================================
# Lazy-loaded predictor
# =====================================================
_LOCK = threading.Lock()
_READY = False

_DEVICE = None
_MODEL = None
_TF = None
_IDX_TO_CLASS = None
_INFO: Dict[str, Any] = {}


def _ensure_loaded():
    global _READY, _DEVICE, _MODEL, _TF, _IDX_TO_CLASS, _INFO

    if _READY:
        return

    with _LOCK:
        if _READY:
            return

        ckpt_path = CKPT_PATH
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint introuvable: {ckpt_path}")

        _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt = torch.load(ckpt_path, map_location=_DEVICE)

        class_to_idx = ckpt["class_to_idx"]
        _IDX_TO_CLASS = {v: k for k, v in class_to_idx.items()}
        num_classes = len(class_to_idx)

        cfg = ckpt.get("cfg", {})
        model_name = cfg.get("model_name", "vit_small_patch14_dinov2.lvd142m")
        img_size = int(cfg.get("img_size", 518))
        pool_mode = cfg.get("pool_mode", "cls")

        backbone = make_backbone(model_name)
        model = DinoFineGrainedClassifier(backbone, num_classes=num_classes, pool_mode=pool_mode).to(_DEVICE)

        # strict=True => si jamais mismatch, tu verras direct l'erreur précise
        model.load_state_dict(ckpt["model_state"], strict=True)
        model.eval()

        pre_eval = AlphaCropAndMask(alpha_threshold=5, pad=6, bg_rgb=(0, 0, 0), random_bg=False)
        pad_eval = PadToSquare(fill=(0, 0, 0), random_fill=False)

        _TF = T.Compose([
            pre_eval,
            pad_eval,
            T.Resize((img_size, img_size), interpolation=InterpolationMode.BICUBIC, antialias=True),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
        ])

        _MODEL = model
        _INFO = {
            "ckpt_path": ckpt_path,
            "device": _DEVICE,
            "model_name": model_name,
            "img_size": img_size,
            "pool_mode": pool_mode,
            "num_classes": num_classes,
        }

        _READY = True


def predictor_info() -> Dict[str, Any]:
    _ensure_loaded()
    return dict(_INFO)


@torch.inference_mode()
def predict_cutout_bytes(png_bytes: bytes, topk: int = 5) -> Dict[str, Any]:
    _ensure_loaded()

    topk = max(1, min(int(topk), 20))

    img = Image.open(io.BytesIO(png_bytes))
    x = _TF(img).unsqueeze(0).to(_DEVICE)

    logits = _MODEL(x)
    probs = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()

    order = np.argsort(-probs)[:topk]
    items: List[Dict[str, Any]] = []
    for i in order:
        i = int(i)
        items.append({"idx": i, "label": _IDX_TO_CLASS[i], "prob": float(probs[i])})

    return {
        "top1": items[0],
        "topk": items,
        "meta": predictor_info(),
    }