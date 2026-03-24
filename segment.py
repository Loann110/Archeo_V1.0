# segment.py
"""
Segmentation TorchScript pour Archeo (API utilisée par app.py)

Dépendances:
  pip install torch pillow numpy

Variables d'environnement (optionnelles):
  SEG_MODEL_PATH    : chemin vers le modèle TorchScript (.pt/.ptc). Défaut: ./models/best_model_traced.pt (ou best_model.pt)
  SEG_DEVICE        : 'cuda' | 'cpu'. Défaut: auto (cuda si dispo)
  SEG_INPUT_SIZE    : taille d'entrée carrée (int). Défaut: 448
  SEG_OVERLAY_ALPHA : alpha du mélange overlay (0..1). Défaut: 0.5

Couleurs (palette par défaut):
  0 -> fond (noir), 1 -> motif (rouge), 2 -> vert, 3 -> bleu
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F


# --------------------------
# Configuration & utilitaires
# --------------------------

_ID2COLOR: Dict[int, Tuple[int, int, int]] = {
    0: (0,   0,   0),    # fond
    1: (255, 60,  60),   # motif
    2: (60,  180, 75),   # (optionnel)
    3: (0,   120, 255),  # (optionnel)
}

_ADE_MEAN = [123.675 / 255.0, 116.280 / 255.0, 103.530 / 255.0]
_ADE_STD  = [ 58.395 / 255.0,  57.120 / 255.0,  57.375 / 255.0]


def _base_dir() -> Path:
    """
    Base directory robuste (support PyInstaller via sys._MEIPASS).
    - Dev: dossier du fichier segment.py
    - PyInstaller: dossier temporaire d'extraction
    """
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        return Path(meipass)
    return Path(__file__).resolve().parent


def resolve_model_path() -> str:
    """Chemin du modèle, priorise env SEG_MODEL_PATH, sinon ./models/."""
    env = os.getenv("SEG_MODEL_PATH", "").strip()
    if env:
        return env

    base = _base_dir()
    for name in ("best_model_traced.pt", "best_model.pt"):
        p = base / "models" / name
        if p.exists():
            return str(p)

    return str(base / "models" / "best_model_traced.pt")


def _pick_device() -> str:
    """Retourne 'cuda' si dispo ou si forcé, sinon 'cpu'."""
    forced = os.getenv("SEG_DEVICE", "").strip().lower()
    if forced in ("cuda", "cpu"):
        return forced
    return "cuda" if torch.cuda.is_available() else "cpu"


def _safe_alpha() -> float:
    try:
        a = float(os.getenv("SEG_OVERLAY_ALPHA", "0.5"))
        return float(np.clip(a, 0.0, 1.0))
    except Exception:
        return 0.5


def _input_size() -> int:
    try:
        v = int(os.getenv("SEG_INPUT_SIZE", "448"))
        return int(np.clip(v, 64, 2048))
    except Exception:
        return 448


def _colorize_mask(mask: np.ndarray, id2color: Dict[int, Tuple[int, int, int]] = _ID2COLOR) -> np.ndarray:
    """(H,W uint8) -> (H,W,3) uint8 colorisé par palette."""
    h, w = mask.shape[:2]
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for k, rgb in id2color.items():
        out[mask == k] = rgb
    return out


def _blend(img_rgb: np.ndarray, color_rgb: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Mélange deux images RGB uint8 (H,W,3) avec alpha [0..1]."""
    img = img_rgb.astype(np.float32)
    col = color_rgb.astype(np.float32)
    out = (1.0 - alpha) * img + alpha * col
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def _preprocess_pil(
    img: Image.Image,
    in_size: int,
    mean: list,
    std: list,
    device: torch.device,
) -> torch.Tensor:
    """
    Remplacement de torchvision.transforms:
      - Resize -> PIL
      - ToTensor -> numpy + torch
      - Normalize -> torch
    Retour: Tensor [1,C,H,W] float32 sur device
    """
    # Resize (PIL)
    img_r = img.resize((in_size, in_size), Image.BILINEAR)

    # HWC float32 [0..1]
    arr = np.asarray(img_r, dtype=np.float32) / 255.0

    # CHW tensor
    t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # [C,H,W]
    t = t.to(device=device, non_blocking=True)

    mean_t = torch.tensor(mean, dtype=t.dtype, device=device)[:, None, None]
    std_t  = torch.tensor(std,  dtype=t.dtype, device=device)[:, None, None]
    t = (t - mean_t) / std_t

    return t.unsqueeze(0)  # [1,C,H,W]


# --------------------------
# Segmenter TorchScript
# --------------------------

class TorchScriptSegmenter:
    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        in_size: Optional[int] = None,
        id2color: Optional[Dict[int, Tuple[int, int, int]]] = None,
        mean: Optional[list] = None,
        std: Optional[list] = None,
    ):
        self.model_path = model_path
        self.device = torch.device(device or _pick_device())
        self.in_size = in_size or _input_size()
        self.id2color = id2color or _ID2COLOR
        self.mean = mean or _ADE_MEAN
        self.std = std or _ADE_STD
        self.alpha = _safe_alpha()

        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Modèle introuvable: {self.model_path}")

        # map_location pour éviter les erreurs CUDA/CPU figés
        self.model = torch.jit.load(self.model_path, map_location=self.device)
        self.model.eval()

    @staticmethod
    def _first_tensor_ndim_ge3(x: Any) -> torch.Tensor:
        """Cherche un tensor ndim>=3 dans une structure (dict/tuple/list)."""
        if isinstance(x, torch.Tensor):
            return x
        if isinstance(x, (tuple, list)):
            for it in x:
                if isinstance(it, torch.Tensor) and it.ndim >= 3:
                    return it
        if isinstance(x, dict):
            # clés fréquentes
            for key in ("logits", "out", "pred", "y", "y_hat"):
                if key in x and isinstance(x[key], torch.Tensor):
                    return x[key]
            for it in x.values():
                if isinstance(it, torch.Tensor) and it.ndim >= 3:
                    return it
        raise RuntimeError("Impossible d'extraire des logits (tensor ndim>=3) depuis la sortie du modèle.")

    def _extract_logits(self, out: Any) -> torch.Tensor:
        """
        Cas possibles:
          - Tensor [B,C,H,W]
          - (loss, logits) ou (logits, ...)
          - dict {'logits': Tensor, ...}
        """
        if isinstance(out, torch.Tensor):
            return out

        if isinstance(out, (tuple, list)):
            if len(out) >= 2 and isinstance(out[1], torch.Tensor):
                return out[1]
            return self._first_tensor_ndim_ge3(out)

        if isinstance(out, dict):
            if "logits" in out and isinstance(out["logits"], torch.Tensor):
                return out["logits"]
            return self._first_tensor_ndim_ge3(out)

        return self._first_tensor_ndim_ge3(out)

    def segment_file(self, img_path: str) -> Dict[str, np.ndarray]:
        """
        Charge une image (PNG/JPG), effectue la segmentation, renvoie:
          {
            "mask":    (H,W) uint8 labels,
            "overlay": (H,W,3) uint8 image mélangée
          }
        """
        p = Path(img_path)
        if not p.exists():
            raise FileNotFoundError(f"Image introuvable: {img_path}")

        img = Image.open(str(p)).convert("RGB")
        W, H = img.size

        x = _preprocess_pil(img, self.in_size, self.mean, self.std, self.device)

        with torch.no_grad():
            out = self.model(x)
            logits = self._extract_logits(out)

            if logits.device != self.device:
                logits = logits.to(self.device)

            # upsample vers taille originale
            logits_up = F.interpolate(
                logits, size=(H, W), mode="bilinear", align_corners=False
            )

            pred = logits_up.argmax(dim=1)  # [1,H,W]
            mask = (
                pred[0]
                .detach()
                .to("cpu", non_blocking=True)
                .numpy()
                .astype(np.uint8)
            )

        img_np = np.asarray(img, dtype=np.uint8)
        colored = _colorize_mask(mask, self.id2color)
        overlay = _blend(img_np, colored, alpha=self.alpha)

        return {"mask": mask, "overlay": overlay}


# --------------------------
# Singleton (utilisé par app.py)
# --------------------------

_SEGMENTER: Optional[TorchScriptSegmenter] = None

def get_segmenter() -> TorchScriptSegmenter:
    """Retourne un singleton du segmenter; charge au premier appel."""
    global _SEGMENTER
    if _SEGMENTER is None:
        model_path = resolve_model_path()
        device = _pick_device()
        _SEGMENTER = TorchScriptSegmenter(model_path=model_path, device=device)
    return _SEGMENTER


# --------------------------
# Exécution directe (debug)
# --------------------------

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("img", help="Chemin vers une image (png/jpg)")
    ap.add_argument("--out", default=None, help="Dossier de sortie (optionnel)")
    args = ap.parse_args()

    out_dir = Path(args.out) if args.out else (_base_dir() / "captures")
    out_dir.mkdir(parents=True, exist_ok=True)

    seg = get_segmenter()
    out = seg.segment_file(args.img)

    mask = out["mask"]
    overlay = out["overlay"]

    base = Path(args.img).stem
    Image.fromarray(mask, mode="L").save(out_dir / f"{base}-mask.png", format="PNG")
    Image.fromarray(overlay, mode="RGB").save(out_dir / f"{base}-overlay.png", format="PNG")

    print("OK:", out_dir / f"{base}-mask.png", "|", out_dir / f"{base}-overlay.png")
