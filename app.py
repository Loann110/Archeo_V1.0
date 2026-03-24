# app_fastapi.py — FastAPI + routes + caches mémoire + segmentation
# - Réglages APSS temps-réel via /meshlab_params
# - Rendu MJPEG unique /stream.mjpg quel que soit le moteur actif
# - Streaming WebSocket binaire /ws_stream (JPEG par message)

import io, time, queue, base64, threading, datetime, os, uuid, tempfile, atexit, asyncio
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
from PIL import Image

from fastapi import FastAPI, Body, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    Response,
    StreamingResponse,
    FileResponse,
)
from fastapi.staticfiles import StaticFiles

# --------- Template (équivalent render_template_string) ----------
from jinja2 import Template

def render_template_string(tmpl: str, **ctx) -> str:
    return Template(tmpl).render(**ctx)

# --------- Moteurs de rendu ----------
from engine import STATE as O3D_STATE
from engine_meshlab import STATE as ML_STATE

# --------- Segmenter (PyTorch) ---------
import segment as segmod
from segment import get_segmenter, resolve_model_path

#--------- Prédicteur DinoV2 + ArcFace (PNG bytes in, features out) ---------
from predictor_arcface import predict_cutout_bytes, predictor_info

# --------- Dossiers ---------
ROOT = Path(__file__).resolve().parent
CAPTURE_DIR = ROOT / "captures"
UPLOADS_DIR = ROOT / "uploads"
CAPTURE_DIR.mkdir(exist_ok=True)
UPLOADS_DIR.mkdir(exist_ok=True)

app = FastAPI()

CAPTURE_STORE = {} 

# --------- Gestion moteur actif & dernier OBJ ----------
ACTIVE_ENGINE = {"name": "open3d"}  # "open3d" | "meshlab"
STATE = O3D_STATE                   # pointeur dynamique vers l'état du moteur courant
LAST_OBJ_PATH: Optional[str] = None # dernier .obj (chemin)

# --------- Nettoyage des .obj  ---------
def clean_uploads(pattern="*.obj", older_than_seconds: float = 60.0):
    """Supprime les .obj plus anciens que older_than_seconds (sécurisé)."""
    try:
        now = time.time()
        for p in UPLOADS_DIR.glob(pattern):
            try:
                age = now - p.stat().st_mtime
                if age >= older_than_seconds:
                    p.unlink(missing_ok=True)
            except Exception as e:
                print(f"[cleanup uploads] impossible de supprimer {p.name}: {e}")
    except Exception as e:
        print("[cleanup uploads] erreur:", e)

clean_uploads(older_than_seconds=60.0)
atexit.register(lambda: clean_uploads(older_than_seconds=60.0))

# --------- Helpers ---------
def safe_name(name: str) -> str:
    if not name:
        return "model.obj"
    out = "".join([c for c in name if c.isalnum() or c in ("-","_",".")]) or "model.obj"
    return out.split("/")[-1].split("\\")[-1]

def unique_obj_path(basename: str) -> Path:
    p = UPLOADS_DIR / safe_name(basename or "model.obj")
    if p.exists():
        ts = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        p = p.with_name(f"{p.stem}-{ts}{p.suffix or '.obj'}")
    return p

def _now_str():
    return datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

def latest_capture_path() -> Optional[Path]:
    try:
        files = sorted(
            CAPTURE_DIR.glob("capture-*.png"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        return files[0] if files else None
    except Exception:
        return None

def resolve_capture_input(v: Optional[str]) -> Optional[Path]:
    if not v:
        return latest_capture_path()
    v = str(v)
    if v.startswith("/captures/"):
        v = v.replace("/captures/", "")
        p = CAPTURE_DIR / v
    else:
        p = Path(v)
        try:
            if p.is_absolute():
                p_rel = p.resolve().relative_to(CAPTURE_DIR.resolve())
                p = CAPTURE_DIR / p_rel
        except Exception:
            p = CAPTURE_DIR / p.name
    return p if p.exists() else None

# --------- Cache mémoire (captures & seg) ---------
MAX_TEMP_ITEMS = 24

def _put_evict(cache: "OrderedDict[str, Dict]", key: str, value):
    cache[key] = value
    while len(cache) > MAX_TEMP_ITEMS:
        try:
            cache.popitem(last=False)
        except Exception:
            break

TEMP_CAPTURES: "OrderedDict[str, Dict]" = OrderedDict()
TEMP_SEGS_OVERLAY: "OrderedDict[str, Dict]" = OrderedDict()
TEMP_SEGS_MASK: "OrderedDict[str, Dict]" = OrderedDict()
TEMP_SEGS_CUTOUT: "OrderedDict[str, Dict]" = OrderedDict()
LAST_CAPTURE_TOKEN: Optional[str] = None

# --------- Segmentation helper ---------
def _segment_png_bytes(png_bytes: bytes):
    """
    Renvoie:
      - mask_png_bytes    (L)
      - overlay_png_bytes (RGB)  = image + overlay couleur (sortie segmenter)
      - cutout_png_bytes  (RGBA) = image originale découpée (alpha = mask), et RGB hors masque = 0
    """
    seg = get_segmenter()
    tmp_path = None
    try:
        # --- image source (RGBA) ---
        src_rgba_img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
        src_rgba = np.array(src_rgba_img)  # (H,W,4) uint8
        src_rgb  = src_rgba[..., :3]
        src_a    = src_rgba[..., 3]        # alpha original 0..255

        # --- pour le segmenter : composite sur fond blanc ---
        a_f = (src_a.astype(np.float32) / 255.0)[..., None]  # (H,W,1)
        bg  = np.array([255, 255, 255], dtype=np.float32)    # blanc
        seg_rgb = (src_rgb.astype(np.float32) * a_f + bg * (1.0 - a_f)).astype(np.uint8)

        # --- segmenter via fichier temporaire (RGB sans transparence) ---
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            Image.fromarray(seg_rgb, mode="RGB").save(tmp, format="PNG", optimize=True)
            tmp.flush()
            tmp_path = tmp.name

        out = seg.segment_file(tmp_path)

        mask = out["mask"]        # (H,W) ou (H,W,1) selon impl
        overlay = out["overlay"]  # (H,W,3) uint8

        # --- normalize shapes ---
        if mask.ndim == 3:
            mask = mask[..., 0]

        H, W = src_rgb.shape[:2]

        # resize safety si tailles diff
        if mask.shape[:2] != (H, W):
            mask = np.array(Image.fromarray(mask.astype(np.uint8), mode="L").resize((W, H), Image.NEAREST))
        if overlay.shape[:2] != (H, W):
            overlay = np.array(Image.fromarray(overlay.astype(np.uint8), mode="RGB").resize((W, H), Image.BILINEAR))

        m = mask

        # convert -> uint8 0..255
        if m.dtype != np.uint8:
            # si float, on essaie d'interpréter 0..1
            if np.issubdtype(m.dtype, np.floating):
                m = np.clip(m, 0.0, 1.0)
                m = (m * 255.0).astype(np.uint8)
            else:
                m = np.clip(m, 0, 255).astype(np.uint8)

        # si mask binaire 0/1 ou faible dynamique -> scale
        mmax = int(m.max()) if m.size else 0
        if mmax <= 1:
            m = (m * 255).astype(np.uint8)
        elif mmax <= 10:
            m = (m.astype(np.float32) * (255.0 / max(1, mmax))).astype(np.uint8)

        # inversion auto (si trop de pixels non-nuls => probablement fond)
        if (m > 0).mean() > 0.5:
            m = (255 - m).astype(np.uint8)

        alpha_mask = m.astype(np.uint8)  # 0..255

        # IMPORTANT : si la capture source est déjà en RGBA (objet seul),
        # on combine le masque avec l’alpha original pour éviter du noir hors-objet.
        # (src_a existe si on a bien fait Image.open(...).convert("RGBA"))
        if "src_a" in locals() and src_a is not None:
            alpha = (alpha_mask.astype(np.uint16) * src_a.astype(np.uint16) // 255).astype(np.uint8)
        else:
            alpha = alpha_mask

        # NE PAS multiplier le RGB par alpha (sinon noir/halo à l’affichage)
        cutout_rgb = src_rgb.copy()

        cutout = np.dstack([cutout_rgb, alpha]).astype(np.uint8)  # RGBA (alpha "straight")

        # --- encode PNGs ---
        mask_png = io.BytesIO()
        Image.fromarray(alpha, mode="L").save(mask_png, format="PNG", optimize=True)

        overlay_png = io.BytesIO()
        Image.fromarray(overlay.astype(np.uint8), mode="RGB").save(overlay_png, format="PNG", optimize=True)

        cutout_png = io.BytesIO()
        Image.fromarray(cutout, mode="RGBA").save(cutout_png, format="PNG", optimize=True)

        return mask_png.getvalue(), overlay_png.getvalue(), cutout_png.getvalue()

    except Exception as e:
        raise RuntimeError(f"{type(e).__name__}: {e}") from e
    finally:
        if tmp_path:
            try:
                os.remove(tmp_path)
            except Exception:
                pass

# --------- Boucles de rendu : une par moteur ----------
def o3d_render_loop():
    while True:
        O3D_STATE.render_once()

def ml_render_loop():
    while True:
        ML_STATE.render_once()

@app.on_event("startup")
def _startup():
    # 2 threads de rendu, un par moteur
    t1 = threading.Thread(target=o3d_render_loop, daemon=True)
    t2 = threading.Thread(target=ml_render_loop, daemon=True)
    t1.start()
    t2.start()

@app.on_event("shutdown")
def _shutdown():
    O3D_STATE.running = False
    ML_STATE.running = False

# --------- Routes HTTP ----------
@app.get("/", response_class=FileResponse)
def index_html():
    p = ROOT / "index.html"
    if not p.exists():
        raise HTTPException(404, "index.html introuvable")
    return FileResponse(p)

@app.get("/health")
def health():
    try:
        model_path = resolve_model_path()
        exists = os.path.isfile(model_path)
        torch_ver = getattr(segmod.torch, "__version__", "n/a")
        device = segmod._pick_device()
        seg = {
            "model_path": model_path,
            "model_exists": exists,
            "torch": torch_ver,
            "device": device,
        }
    except Exception as e:
        seg = {"error": str(e)}

    engine_detail = getattr(STATE, "engine", None) or getattr(STATE, "mode", "?")

    return {
        "ok": True,
        "active_engine": ACTIVE_ENGINE["name"],
        "engine_detail": engine_detail,
        "invert_zoom": getattr(STATE, "invert_zoom", False),
        "zoom_locked": getattr(STATE, "zoom_locked", False),
        "seg": seg,
    }

# --------- Streaming HTTP MJPEG ----------
@app.get("/stream.mjpg")
def stream_mjpg():
    boundary = b"--frame"

    def gen():
        while True:
            frame = STATE.latest_jpeg
            if frame is not None:
                yield (
                    boundary
                    + b"\r\nContent-Type: image/jpeg\r\nContent-Length: "
                    + str(len(frame)).encode()
                    + b"\r\n\r\n"
                    + frame
                    + b"\r\n"
                )
            time.sleep(0.008)

    return StreamingResponse(
        gen(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store"},
    )

@app.get("/frame.jpg")
def frame_jpg():
    if STATE.latest_jpeg is None:
        return Response(content=b"no frame yet", status_code=503, media_type="text/plain")
    return Response(
        content=STATE.latest_jpeg,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store"},
    )

# --------- Streaming WebSocket binaire ----------
@app.websocket("/ws_stream")
async def ws_stream(ws: WebSocket):
    await ws.accept()

    try:
        while STATE.latest_jpeg is None:
            await asyncio.sleep(0.01)

        target_fps = 30.0
        delay = 1.0 / target_fps

        while True:
            frame = STATE.latest_jpeg
            if frame is not None:
                await ws.send_bytes(frame)
            await asyncio.sleep(delay)

    except WebSocketDisconnect:
        return
    except Exception as e:
        print("[ws_stream] closed:", e)
        return

# --------- Contrôles ----------
@app.post("/control")
def control(data: Optional[Dict[str, Any]] = Body(None)):
    data = data or {}
    a = data.get("action")
    if a in ("rotate", "pan", "zoom", "zoom_dir", "zoom_sem"):
        STATE.cmd_q.put({"action": a, **data})
        return {"ok": True}
    raise HTTPException(400, "bad action")

@app.post("/settings")
def settings(data: Optional[Dict[str, Any]] = Body(None)):
    data = data or {}
    STATE.cmd_q.put({"action": "settings", **data})
    return {"ok": True}

@app.post("/lights")
def lights(data: Optional[Dict[str, Any]] = Body(None)):
    data = data or {}
    STATE.cmd_q.put({"action": "lights", **data})
    return {"ok": True}

@app.post("/reset")
def reset():
    STATE.cmd_q.put({"action": "reset"})
    return {"ok": True}

@app.post("/fit")
def fit():
    STATE.cmd_q.put({"action": "fit"})
    return {"ok": True}

@app.post("/remove")
def remove():
    global LAST_OBJ_PATH
    LAST_OBJ_PATH = None
    STATE.cmd_q.put({"action": "remove"})
    return {"ok": True}

@app.post("/resize")
def resize(data: Optional[Dict[str, Any]] = Body(None)):
    data = data or {}
    STATE.cmd_q.put({"action": "resize", **data})
    return {"ok": True}

# --------- Switch de moteur ----------
@app.post("/engine")
def set_engine(data: Optional[Dict[str, Any]] = Body(None)):
    global STATE, ACTIVE_ENGINE
    data = data or {}
    name = str(data.get("engine", "")).lower()
    if name not in ("open3d", "meshlab"):
        raise HTTPException(400, "engine must be 'open3d' or 'meshlab'")

    ACTIVE_ENGINE["name"] = name
    STATE = O3D_STATE if name == "open3d" else ML_STATE
    return {"ok": True, "engine": name}

# --------- Réglages MeshLab temps réel ----------
@app.post("/meshlab_params")
def meshlab_params(data: Optional[Dict[str, Any]] = Body(None)):
    data = data or {}
    ML_STATE.cmd_q.put(
        {
            "action": "ml_params",
            "filterscale": data.get("filterscale"),
            "sphericalparameter": data.get("sphericalparameter"),
            "curvaturetype": data.get("curvaturetype"),
            "rotate": data.get("rotate"),
        }
    )
    return {"ok": True}

@app.post("/upload_obj")
def upload_obj(data: Optional[Dict[str, Any]] = Body(None)):
    data = data or {}
    obj_name = safe_name(data.get("obj_name") or "model.obj")
    obj_b64 = data.get("obj_b64")
    if not obj_b64:
        raise HTTPException(400, "obj_b64 manquant")

    try:
        text = base64.b64decode(obj_b64).decode("utf-8", errors="ignore")
    except Exception:
        raise HTTPException(400, "OBJ base64 invalide")

    clean_uploads(older_than_seconds=60.0)

    path = unique_obj_path(obj_name)
    try:
        path.write_text(text, encoding="utf-8")
    except Exception as e:
        raise HTTPException(500, f"Écriture OBJ échouée: {e}")

    global LAST_OBJ_PATH
    LAST_OBJ_PATH = str(path)

    O3D_STATE.cmd_q.put({"action": "load_obj", "path": str(path)})
    ML_STATE.cmd_q.put({"action": "load_obj", "path": str(path)})

    return {"ok": True, "token": path.name, "path": str(path)}

# --------- Capture mémoire + page d’aperçu ----------
@app.post("/capture_png")
def capture_png(opts: Optional[Dict[str, Any]] = Body(None)):
    opts = opts or {}
    want_save = bool(opts.get("save", False))

    # --- réglages "photo mode" ---
    scale = float(opts.get("scale", 3.0))     # 640x360 -> ~1920x1080
    ssaa  = int(opts.get("ssaa", 2))          # supersampling (anti-aliasing)
    max_side = int(opts.get("max_side", 2560))

    ssaa = max(1, min(ssaa, 4))
    max_side = max(320, min(max_side, 4096))

    # --- taille réelle de la frame courante (stream) ---
    view_w = opts.get("view_w")
    view_h = opts.get("view_h")

    base_w, base_h = 640, 360

    try:
        vw = int(view_w) if view_w is not None else 0
        vh = int(view_h) if view_h is not None else 0
        if vw >= 64 and vh >= 64:
            base_w, base_h = vw, vh
        elif STATE.latest_jpeg:
            im = Image.open(io.BytesIO(STATE.latest_jpeg))
            base_w, base_h = im.size
    except Exception:
        pass

    # taille de sortie demandée (indépendante du stream)
    out_w = int(round(base_w * scale))
    out_h = int(round(base_h * scale))

    aspect = base_w / float(max(1, base_h))

    def clamp_keep_aspect(w, h, min_w=320, min_h=240, max_side=2560):
        # cap max_side
        m = max(w, h)
        if m > max_side:
            k = max_side / float(m)
            w = int(round(w * k))
            h = int(round(h * k))

        # mins en gardant le ratio
        if w < min_w:
            w = min_w
            h = int(round(w / aspect))
        if h < min_h:
            h = min_h
            w = int(round(h * aspect))

        # re-cap max_side au cas où les mins ont fait dépasser
        m = max(w, h)
        if m > max_side:
            k = max_side / float(m)
            w = int(round(w * k))
            h = int(round(h * k))

        return max(64, w), max(64, h)

    out_w = int(round(base_w * scale))
    out_h = int(round(base_h * scale))

    out_w, out_h = clamp_keep_aspect(out_w, out_h, min_w=320, min_h=240, max_side=max_side)


    # cap max_side en conservant ratio
    m = max(out_w, out_h)
    if m > max_side:
        k = max_side / float(m)
        out_w = int(round(out_w * k))
        out_h = int(round(out_h * k))

    out_w = max(320, out_w)
    out_h = max(240, out_h)

    # --- demande au moteur une capture OBJET SEUL (RGBA) ---
    result_q = queue.Queue(maxsize=1)
    STATE.cmd_q.put({
        "action": "capture_object_hq", 
        "out_w": out_w,
        "out_h": out_h,
        "ssaa": ssaa,
        "result_q": result_q
    })

    try:
        status, payload = result_q.get(timeout=8.0)
    except queue.Empty:
        raise HTTPException(504, "Capture timeout")

    if status != "ok":
        raise HTTPException(500, f"Capture échouée: {payload}")

    arr = np.asarray(payload)

    # sécurité: forcer RGBA uint8
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 4:
        # fallback: si jamais ça sort en RGB
        if arr.ndim == 3 and arr.shape[2] == 3:
            alpha = np.full((arr.shape[0], arr.shape[1], 1), 255, dtype=np.uint8)
            arr = np.concatenate([arr, alpha], axis=2)
        else:
            raise HTTPException(500, f"Capture RGBA invalide: shape={arr.shape}")

    # encode PNG RGBA (fond transparent)
    buf = io.BytesIO()
    Image.fromarray(arr, mode="RGBA").save(buf, format="PNG", compress_level=6)
    png_bytes = buf.getvalue()

    token = uuid.uuid4().hex
    ts = _now_str()
    _put_evict(TEMP_CAPTURES, token, {"bytes": png_bytes, "ts": ts, "mime": "image/png"})
    global LAST_CAPTURE_TOKEN
    LAST_CAPTURE_TOKEN = token

    file_url = None
    if want_save:
        fname = f"capture-{ts}.png"
        out = CAPTURE_DIR / fname
        try:
            out.write_bytes(png_bytes)
            file_url = f"/captures/{fname}"
        except Exception as e:
            raise HTTPException(500, f"Écriture PNG échouée: {e}")

    return {"ok": True, "token": token, "url": f"/capture_view?token={token}", "file": file_url}

@app.get("/temp_capture/{token}.png")
def temp_capture_png(token: str):
    meta = TEMP_CAPTURES.get(token)
    if not meta:
        raise HTTPException(404, "Not found")
    return Response(
        content=meta["bytes"],
        media_type=meta.get("mime", "image/png"),
        headers={"Cache-Control": "no-store"},
    )

# --------- Segmentation : API + flux mémoire ----------
@app.get("/seg_temp/{kind}/{token}.png")
def seg_temp_png(kind: str, token: str):
    if kind == "overlay":
        meta = TEMP_SEGS_OVERLAY.get(token)
    elif kind == "cutout":
        meta = TEMP_SEGS_CUTOUT.get(token)
    else:  # "mask"
        meta = TEMP_SEGS_MASK.get(token)

    if not meta:
        raise HTTPException(404, "Not found")

    return Response(content=meta["bytes"], media_type="image/png", headers={"Cache-Control": "no-store"})

@app.post("/segment_last")
def segment_last(data: Optional[Dict[str, Any]] = Body(None)):
    data = data or {}
    req_file = data.get("file")
    req_token = data.get("token")
    want_save = bool(data.get("save", False))

    png_bytes: Optional[bytes] = None
    src_info = ""

    p = resolve_capture_input(req_file) if req_file else None
    if p and p.exists():
        try:
            png_bytes = Path(p).read_bytes()
            src_info = f"file:{p.name}"
        except Exception as e:
            raise HTTPException(500, f"Lecture capture échouée: {e}")

    if png_bytes is None and req_token:
        meta = TEMP_CAPTURES.get(req_token)
        if not meta:
            raise HTTPException(400, "Token capture inconnu")
        png_bytes = meta["bytes"]
        src_info = f"token:{req_token}"

    global LAST_CAPTURE_TOKEN
    if png_bytes is None and LAST_CAPTURE_TOKEN:
        meta = TEMP_CAPTURES.get(LAST_CAPTURE_TOKEN)
        if meta:
            png_bytes = meta["bytes"]
            src_info = f"token:{LAST_CAPTURE_TOKEN}"

    if png_bytes is None:
        raise HTTPException(400, "Aucune capture disponible (fichier ou token).")

    try:
        mask_png_bytes, overlay_png_bytes, cutout_png_bytes = _segment_png_bytes(png_bytes)
    except Exception as e:
        raise HTTPException(500, f"Segmentation échouée: {e}")

    ts = _now_str()
    ov_token = uuid.uuid4().hex
    ct_token = uuid.uuid4().hex
    mk_token = uuid.uuid4().hex

    _put_evict(TEMP_SEGS_OVERLAY, ov_token, {"bytes": overlay_png_bytes, "ts": ts})
    _put_evict(TEMP_SEGS_CUTOUT, ct_token, {"bytes": cutout_png_bytes, "ts": ts})
    _put_evict(TEMP_SEGS_MASK, mk_token, {"bytes": mask_png_bytes, "ts": ts})

    overlay_url = f"/seg_temp/overlay/{ov_token}.png"
    cutout_url  = f"/seg_temp/cutout/{ct_token}.png"
    mask_url    = f"/seg_temp/mask/{mk_token}.png"

    disk_overlay = disk_cutout = disk_mask = None
    if want_save:
        stem = "capture-" + ts
        try:
            (CAPTURE_DIR / f"{stem}-overlay.png").write_bytes(overlay_png_bytes)
            (CAPTURE_DIR / f"{stem}-cutout.png").write_bytes(cutout_png_bytes)
            (CAPTURE_DIR / f"{stem}-mask.png").write_bytes(mask_png_bytes)
            disk_overlay = f"/captures/{stem}-overlay.png"
            disk_cutout  = f"/captures/{stem}-cutout.png"
            disk_mask    = f"/captures/{stem}-mask.png"
        except Exception as e:
            raise HTTPException(500, f"Enregistrement résultats échoué: {e}")

    return {
        "ok": True,
        "src": src_info,
        "overlay_url": disk_overlay or overlay_url,
        "cutout_url": disk_cutout or cutout_url,
        "mask_url": disk_mask or mask_url,
        "view_url": f"/segment_view?ov_token={ov_token}&ct_token={ct_token}&mk_token={mk_token}",
    }

# --------- Pages HTML ----------
@app.get("/capture_view", response_class=HTMLResponse)
def capture_view(token: str):
    meta = TEMP_CAPTURES.get(token or "")
    if not meta:
        raise HTTPException(404, "Capture inconnue ou expirée")

    ts = meta.get("ts", _now_str())
    tmpl = r"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8" />
<title>Capture — Aperçu</title>
<meta name="viewport" content="width=device-width, initial-scale=1" />
<style>
  :root{
    --bg:#0b0b10; --card:#12121a; --muted:#a9b0bf; --text:#eef1f8;
    --primary1:#082696; --primary2:#085996; --border:rgba(255,255,255,.10);
    --glow: 0 10px 30px rgba(104,128,255,.22), 0 2px 10px rgba(0,0,0,.45);
  }
  *{ box-sizing:border-box; }
  html,body{ height:100%; margin:0; }
  body{
    font:14px/1.5 system-ui,Segoe UI,Roboto,Arial;
    color:var(--text);
    background:
      radial-gradient(60rem 40rem at 10% 10%, rgba(80,120,255,.10), transparent 60%),
      radial-gradient(50rem 30rem at 90% 20%, rgba(150,80,255,.10), transparent 60%),
      var(--bg);
  }

  /* Layout: full height, no crop, internal scroll only if needed */
  .wrap{
    min-height:100dvh;               /* modern mobile */
    min-height:100vh;                /* fallback */
    width:min(1100px, 96vw);
    margin:0 auto;
    padding:clamp(10px, 2vw, 16px);
    display:flex;
    flex-direction:column;
    gap:clamp(10px, 1.8vw, 14px);
  }

  .bar{
    flex:0 0 auto;
    display:flex;
    align-items:center;
    gap:10px;
    flex-wrap:wrap;
    justify-content:space-between;

    background:linear-gradient(180deg, rgba(255,255,255,.06), rgba(255,255,255,.03));
    border:1px solid var(--border);
    border-radius:14px;
    padding:clamp(8px, 1.6vw, 12px);
    backdrop-filter: blur(6px);
    box-shadow: var(--glow);
  }

  /* A “row” inside bar that can stack nicely */
  .bar-left, .bar-right{
    display:flex;
    gap:10px;
    align-items:center;
    flex-wrap:wrap;
  }
  .bar-left{ flex:1 1 240px; }
  .bar-right{ flex:0 1 auto; justify-content:flex-end; }

  .btn{
    --padY: .55rem;
    --padX: .8rem;
    display:inline-flex; align-items:center; gap:.55rem;
    padding:var(--padY) var(--padX);
    border-radius:12px;
    border:1px solid var(--border);
    color:var(--text);
    background:rgba(255,255,255,.05);
    cursor:pointer;
    text-decoration:none;
    user-select:none;
    transition:transform .08s ease, filter .12s ease, box-shadow .12s ease, background .12s ease;
    white-space:nowrap;
  }
  .btn svg{ width:18px; height:18px; display:block; }
  .btn:hover{ background:rgba(255,255,255,.08); }
  .btn:active{ transform:translateY(1px); }
  .btn.primary{
    background:linear-gradient(135deg, var(--primary1), var(--primary2));
    border-color:transparent;
    box-shadow: var(--glow);
  }
  .btn.ghost{
    background:transparent;
    border-color:var(--border);
    color:#b8c3ff;
  }

  /* Make buttons more compact on small screens */
  @media (max-width: 520px){
    .btn{ --padY:.5rem; --padX:.65rem; border-radius:11px; }
    .btn svg{ width:17px; height:17px; }
  }

  #busy{
    display:none;
    color:#cdd6ff;
    font-size:13px;
    padding:.35rem .55rem;
    border:1px solid var(--border);
    border-radius:999px;
    background:rgba(255,255,255,.04);
  }
  #busy::after{
    content:"";
    display:inline-block;
    width:12px; height:12px;
    margin-left:8px;
    border-radius:50%;
    border:2px solid rgba(205,214,255,.35);
    border-top-color:#cdd6ff;
    animation: spin .8s linear infinite;
    vertical-align:-2px;
  }
  @keyframes spin{ to{ transform:rotate(360deg); } }

  /* Card fills remaining space but DOES NOT force crop: it can grow */
  .card{
    flex:1 1 auto;
    min-height: min(60vh, 720px); /* gives room for image on desktop */
    background:var(--card);
    border:1px solid var(--border);
    border-radius:14px;
    padding:clamp(8px, 1.4vw, 12px);
    box-shadow: 0 8px 22px rgba(0,0,0,.35);
    display:flex;
  }

  .imgbox{
    flex:1 1 auto;
    min-height:0;
    border-radius:12px;
    overflow:auto; /* key: if image is huge, user can scroll inside card */
    display:grid;
    place-items:center;
    padding:clamp(8px, 1.8vw, 14px);

    background-color:#111;
    background-image:
      linear-gradient(45deg, rgba(255,255,255,.06) 25%, transparent 25%),
      linear-gradient(-45deg, rgba(255,255,255,.06) 25%, transparent 25%),
      linear-gradient(45deg, transparent 75%, rgba(255,255,255,.06) 75%),
      linear-gradient(-45deg, transparent 75%, rgba(255,255,255,.06) 75%);
    background-size: 20px 20px;
    background-position: 0 0, 0 10px, 10px -10px, -10px 0px;
  }

  /* Responsive image: never cut; center; allow zoom by opening in new tab */
  .imgbox img{
    display:block;
    max-width:100%;
    max-height: calc(100dvh - 190px); /* avoids bottom crop on small screens */
    height:auto;
    width:auto;
    object-fit:contain;
    border-radius:10px;
    box-shadow: 0 10px 26px rgba(0,0,0,.45);
  }

  /* On very short screens: reduce max-height constraint a bit */
  @media (max-height: 650px){
    .imgbox img{ max-height: calc(100dvh - 220px); }
  }

  /* Optional: tiny helper text area (hidden by default) */
  .hint{
    color:var(--muted);
    font-size:12px;
    margin-left:4px;
  }
</style>
</head>
<body>
  <div class="wrap">
    <div class="bar">
      <div class="bar-left">
        <a class="btn ghost" href="/" title="Ouvrir le viewer">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
            <path d="M15 3h6v6"/><path d="M10 14L21 3"/><path d="M21 14v7H3V3h7"/>
          </svg><span>Viewer</span>
        </a>
        <span id="busy">Traitement…</span>
      </div>

      <div class="bar-right">
        <a class="btn" href="/temp_capture/{{token}}.png" download="capture-{{ts}}.png">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><path d="M7 10l5 5 5-5"/><path d="M12 15V3"/>
          </svg><span>Enregistrer</span>
        </a>

        <button id="btnSeg" class="btn primary">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
            <circle cx="12" cy="12" r="9"/><path d="M12 3v9h9"/>
          </svg><span>Segmenter</span>
        </button>
      </div>
    </div>

    <div class="card">
      <div class="imgbox">
        <img id="ov" src="/temp_capture/{{token}}.png" alt="Capture"/>
      </div>
    </div>
  </div>

<script>
const token = "{{token}}";
const $ = (q)=>document.querySelector(q);

async function postJSON(url, data){
  const r = await fetch(url, {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify(data||{})
  });
  let j=null; try{ j = await r.json(); }catch(_){}
  return { ok:r.ok && j && j.ok!==false, status:r.status, j };
}

$("#btnSeg")?.addEventListener('click', async ()=>{
  $("#busy").style.display = "";
  try{
    const r = await postJSON('/segment_last', { token });
    if(!r.ok){ alert('Segmentation échouée'); return; }
    const url = r.j && (r.j.view_url || r.j.overlay_url || r.j.cutout_url || r.j.mask_url || r.j.url);
    if(url) window.location.href = url;
  }catch(e){
    console.error(e);
    alert('Erreur');
  }finally{
    $("#busy").style.display = "none";
  }
});
</script>
</body>
</html>"""
    html = render_template_string(tmpl, token=token, ts=ts)
    return HTMLResponse(html)

@app.get("/segment_view", response_class=HTMLResponse)
def segment_view(
    overlay: Optional[str] = None,
    cutout: Optional[str] = None,
    mask: Optional[str] = None,
    ov_token: Optional[str] = None,
    ct_token: Optional[str] = None,
    mk_token: Optional[str] = None,
):
    overlay_url = overlay
    cutout_url = cutout
    mask_url = mask

    if not overlay_url and ov_token:
        overlay_url = f"/seg_temp/overlay/{ov_token}.png"
    if not cutout_url and ct_token:
        cutout_url = f"/seg_temp/cutout/{ct_token}.png"
    if not mask_url and mk_token:
        mask_url = f"/seg_temp/mask/{mk_token}.png"

    ts = _now_str()

    tmpl = r"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8" />
<title>Segmentation — Aperçu</title>
<meta name="viewport" content="width=device-width, initial-scale=1" />
<style>
  :root{ --bg:#0b0b10; --card:#12121a; --muted:#a9b0bf; --text:#eef1f8;
    --primary1:#082696; --primary2:#085996; --border:rgba(255,255,255,.08);
    --glow: 0 10px 30px rgba(104,128,255,.25), 0 2px 10px rgba(0,0,0,.4);}

  *{ box-sizing:border-box; } html,body{ height:100%; margin:0; }
  body{ font:14px/1.5 system-ui,Segoe UI,Roboto,Arial; color:var(--text);
    background: radial-gradient(60rem 40rem at 10% 10%, rgba(80,120,255,.10), transparent 60%),
               radial-gradient(50rem 30rem at 90% 20%, rgba(150,80,255,.10), transparent 60%), var(--bg); }

  .wrap{ max-width:min(1100px,92vw); margin:20px auto; }

  .bar{ display:flex; flex-wrap:wrap; gap:10px; align-items:center;
    background:linear-gradient(180deg, rgba(255,255,255,.06), rgba(255,255,255,.03));
    border:1px solid var(--border); border-radius:14px; padding:.6rem; backdrop-filter: blur(6px); box-shadow: var(--glow); }

  .spacer{ flex:1 1 auto; }

  .btn{ --pad:.55rem .8rem; display:inline-flex; align-items:center; gap:.5rem;
    padding:var(--pad); border-radius:12px; border:1px solid var(--border);
    color:var(--text); background:rgba(255,255,255,.05); cursor:pointer; text-decoration:none; user-select:none;
    transition:transform .08s ease, background .12s ease, filter .12s ease; }
  .btn svg{ width:18px; height:18px; display:block; }
  .btn:hover{ background:rgba(255,255,255,.08); }
  .btn:active{ transform:translateY(1px); }
  .btn.ghost{ background:transparent; border-color:var(--border); color:#b8c3ff; }
  .btn.primary{ background:linear-gradient(135deg, var(--primary1), var(--primary2)); border-color:transparent; box-shadow: var(--glow); }
  .btn:disabled{ opacity:.6; cursor:not-allowed; filter:saturate(.8); }

  .grid{ margin-top:14px; display:grid; grid-template-columns:1fr; gap:14px; }
  @media (min-width: 920px){ .grid{ grid-template-columns:1fr 1fr; } }

  .card{ background:var(--card); border:1px solid var(--border); border-radius:14px; padding:8px; box-shadow: 0 8px 22px rgba(0,0,0,.35); }
  .card h3{ margin:8px 10px 10px; font-size:13px; font-weight:650; color:#dce2ff; letter-spacing:.2px; }

  .imgbox{ display:grid; place-items:center; background:#000; border-radius:10px; overflow:hidden; }
  .imgbox img{ max-width:100%; height:auto; display:block; }

  .imgbox.checker{
    background-color:#111;
    background-image:
      linear-gradient(45deg, rgba(255,255,255,.06) 25%, transparent 25%),
      linear-gradient(-45deg, rgba(255,255,255,.06) 25%, transparent 25%),
      linear-gradient(45deg, transparent 75%, rgba(255,255,255,.06) 75%),
      linear-gradient(-45deg, transparent 75%, rgba(255,255,255,.06) 75%);
    background-size: 20px 20px;
    background-position: 0 0, 0 10px, 10px -10px, -10px 0px;
  }

  .note{ color:var(--muted); font-size:12px; margin-top:10px; display:flex; gap:.5rem; align-items:center; flex-wrap:wrap; padding:0 4px 4px; }

  /* ====== Prediction UI ====== */
  .pred{
    margin-top:12px;
    background:linear-gradient(180deg, rgba(255,255,255,.05), rgba(255,255,255,.03));
    border:1px solid var(--border);
    border-radius:14px;
    padding:12px;
    box-shadow: 0 8px 22px rgba(0,0,0,.30);
  }
  .pred-head{
    display:flex;
    gap:10px;
    align-items:center;
    justify-content:space-between;
    flex-wrap:wrap;
    margin-bottom:10px;
  }
  .pred-title{
    display:flex;
    gap:10px;
    align-items:center;
    font-weight:700;
    color:#dce2ff;
    letter-spacing:.2px;
  }
  .pill{
    display:inline-flex;
    align-items:center;
    gap:8px;
    padding:.32rem .6rem;
    border-radius:999px;
    border:1px solid var(--border);
    background:rgba(255,255,255,.04);
    color:#cdd6ff;
    font-size:12px;
  }
  .spinner{
    width:12px; height:12px;
    border-radius:50%;
    border:2px solid rgba(205,214,255,.35);
    border-top-color:#cdd6ff;
    animation: spin .8s linear infinite;
    display:none;
  }
  @keyframes spin{ to{ transform:rotate(360deg); } }

  .pred-top1{
    display:flex;
    gap:10px;
    align-items:baseline;
    padding:10px;
    border-radius:12px;
    border:1px solid rgba(255,255,255,.10);
    background:rgba(0,0,0,.25);
  }
  .pred-label{
    font-size:14px;
    font-weight:800;
    color:#eef1f8;
  }
  .pred-score{
    font-size:12px;
    color:var(--muted);
  }

  .pred-list{
    margin-top:10px;
    display:flex;
    flex-direction:column;
    gap:8px;
  }
  .row{
    display:grid;
    grid-template-columns: 26px 1fr 64px;
    gap:10px;
    align-items:center;
    padding:8px 10px;
    border:1px solid rgba(255,255,255,.08);
    background:rgba(255,255,255,.03);
    border-radius:12px;
  }
  .rank{
    width:22px; height:22px;
    display:grid; place-items:center;
    border-radius:8px;
    border:1px solid rgba(255,255,255,.10);
    background:rgba(0,0,0,.25);
    color:#cdd6ff;
    font-weight:800;
    font-size:12px;
  }
  .lbl{
    overflow:hidden;
    text-overflow:ellipsis;
    white-space:nowrap;
    font-weight:650;
    color:#e8ecff;
  }
  .pct{
    text-align:right;
    font-variant-numeric: tabular-nums;
    color:#cdd6ff;
    font-weight:750;
  }
  .err{
    color:#ffb4b4;
    border:1px solid rgba(255,180,180,.25);
    background:rgba(255,180,180,.06);
    padding:10px 12px;
    border-radius:12px;
  }
</style>
</head>
<body>
  <div class="wrap">
    <div class="bar">
      <a href="/" class="btn ghost" title="Ouvrir le viewer">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
          <path d="M15 3h6v6"/><path d="M10 14L21 3"/><path d="M21 14v7H3V3h7"/>
        </svg><span>Viewer</span>
      </a>

      <div class="spacer"></div>

      {% if overlay_url %}
      <a class="btn" href="{{overlay_url}}" download="overlay-{{ts}}.png" title="Enregistrer l’overlay">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><path d="M7 10l5 5 5-5"/><path d="M12 15V3"/>
        </svg><span>Enregistrer l’overlay</span>
      </a>
      {% endif %}

      {% if cutout_url %}
      <a class="btn" href="{{cutout_url}}" download="cutout-{{ts}}.png" title="Enregistrer l’image découpée">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><path d="M7 10l5 5 5-5"/><path d="M12 15V3"/>
        </svg><span>Enregistrer l’image découpée</span>
      </a>
      {% endif %}

      {% if mask_url %}
      <button id="btnCls" class="btn primary" title="Classifier">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
          <path d="M4 19V5"/><path d="M4 19h16"/><path d="M7 15l3-3 3 2 4-6"/>
        </svg><span>Classifier</span>
      </button>
      {% endif %}
    </div>

    <!-- ===== Prediction panel (clean UI) ===== -->
    {% if mask_url %}
    <div id="predPanel" class="pred" style="display:none;">
      <div class="pred-head">
        <div class="pred-title">
          <span>Résultat du classifieur</span>
          <span class="pill">
            <span id="predSpin" class="spinner"></span>
            <span id="predStatus">prêt</span>
          </span>
        </div>
        <div class="note" style="margin:0; padding:0;">
          <span style="color:#8fa0ff;">source :</span>
          <span id="predSrc" style="color:var(--muted);">cutout</span>
        </div>
      </div>

      <div id="predError" class="err" style="display:none;"></div>

      <div id="predOk" style="display:none;">
        <div class="pred-top1">
          <div class="pred-label" id="predTop1Lbl">—</div>
          <div class="pred-score" id="predTop1Pct">—</div>
        </div>

        <div class="pred-list" id="predList"></div>
      </div>
    </div>
    {% endif %}

    <div class="grid">
      <div class="card">
        <h3>Overlay (image + masque)</h3>
        <div class="imgbox">
          {% if overlay_url %}
            <img src="{{overlay_url}}" alt="Overlay segmentation"/>
          {% else %}
            <div style="padding:22px; color:#bbb">Aucun overlay fourni.</div>
          {% endif %}
        </div>
      </div>

      <div class="card">
        <h3>Image découpée (cutout)</h3>
        <div class="imgbox checker">
          {% if cutout_url %}
            <img src="{{cutout_url}}" alt="Image découpée (cutout)"/>
          {% else %}
            <div style="padding:22px; color:#bbb">Aucun cutout fourni.</div>
          {% endif %}
        </div>
      </div>
    </div>

    {% if mask_url %}
    <div class="note" style="margin-top:10px;">
      <span>Masque :</span>
      <a class="btn" href="{{mask_url}}" target="_blank" rel="noopener">ouvrir</a>
    </div>
    {% endif %}
  </div>

<script>
  const ovToken = "{{ov_token or ''}}";
  const ctToken = "{{ct_token or ''}}";
  const mkToken = "{{mk_token or ''}}";

  async function postJSON(url, data){
    const r = await fetch(url, {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify(data||{})
    });
    let j=null; try{ j = await r.json(); }catch(_){}
    return { ok:r.ok && j && j.ok!==false, status:r.status, j };
  }

  function fmtPct(p){
    const v = (Number(p) || 0) * 100;
    return `${v.toFixed(2)}%`;
  }

  const btn       = document.getElementById("btnCls");
  const panel     = document.getElementById("predPanel");
  const spin      = document.getElementById("predSpin");
  const status    = document.getElementById("predStatus");
  const src       = document.getElementById("predSrc");
  const errBox    = document.getElementById("predError");
  const okBox     = document.getElementById("predOk");
  const top1Lbl   = document.getElementById("predTop1Lbl");
  const top1Pct   = document.getElementById("predTop1Pct");
  const listBox   = document.getElementById("predList");

  btn?.addEventListener("click", async () => {
    if (!ctToken) {
      // pas d'alert ici: on affiche proprement dans le panel
      if (panel) panel.style.display = "";
      if (errBox) {
        errBox.style.display = "";
        errBox.textContent = "Pas de ct_token → impossible de classifier. (Ouvre cette page via segment_last)";
      }
      if (okBox) okBox.style.display = "none";
      return;
    }

    if (panel) panel.style.display = "";
    if (errBox) { errBox.style.display = "none"; errBox.textContent = ""; }
    if (okBox) okBox.style.display = "none";
    if (status) status.textContent = "prediction…";
    if (spin) spin.style.display = "inline-block";
    if (src) src.textContent = "cutout (mémoire)";

    if (btn) btn.disabled = true;

    try {
      const r = await postJSON("/predict_cutout", { ct_token: ctToken, topk: 5 });
      if (!r.ok) {
        console.error("predict_cutout failed", r.status, r.j);
        if (errBox) {
          errBox.style.display = "";
          errBox.textContent = `Prediction échouée (HTTP ${r.status}). Voir console.`;
        }
        if (status) status.textContent = "erreur";
        return;
      }

      const j = r.j || {};
      const top1 = j.top1;
      const topk = j.topk || [];

      if (!top1 || !topk.length) {
        if (errBox) {
          errBox.style.display = "";
          errBox.textContent = "Réponse prediction invalide (top1/topk manquants).";
        }
        if (status) status.textContent = "erreur";
        return;
      }

      // Top-1
      if (top1Lbl) top1Lbl.textContent = top1.label ?? "—";
      if (top1Pct) top1Pct.textContent = fmtPct(top1.prob);

      // Top-k list
      if (listBox) {
        listBox.innerHTML = "";
        topk.forEach((p, i) => {
          const row = document.createElement("div");
          row.className = "row";

          const rank = document.createElement("div");
          rank.className = "rank";
          rank.textContent = String(i + 1);

          const lbl = document.createElement("div");
          lbl.className = "lbl";
          lbl.title = p.label ?? "";
          lbl.textContent = p.label ?? "—";

          const pct = document.createElement("div");
          pct.className = "pct";
          pct.textContent = fmtPct(p.prob);

          row.appendChild(rank);
          row.appendChild(lbl);
          row.appendChild(pct);
          listBox.appendChild(row);
        });
      }

      if (okBox) okBox.style.display = "";
      if (status) status.textContent = "ok";

    } catch (e) {
      console.error(e);
      if (errBox) {
        errBox.style.display = "";
        errBox.textContent = "Erreur JS pendant la prediction. Voir console.";
      }
      if (status) status.textContent = "erreur";
    } finally {
      if (spin) spin.style.display = "none";
      if (btn) btn.disabled = false;
    }
  });
</script>
</body>
</html>"""

    html = render_template_string(
        tmpl,
        overlay_url=overlay_url,
        cutout_url=cutout_url,
        mask_url=mask_url,
        ts=ts,
        ov_token=ov_token,
        ct_token=ct_token,
        mk_token=mk_token,
    )
    return HTMLResponse(html)

# --------- Fichiers captures / uploads ----------
def _safe_file(base: Path, fname: str) -> Path:
    base_r = base.resolve()
    p = (base / fname).resolve()
    if base_r not in p.parents and p != base_r:
        raise HTTPException(404, "Not found")
    if not p.exists() or not p.is_file():
        raise HTTPException(404, "Not found")
    return p

@app.get("/captures/{fname:path}")
def dl_capture(fname: str):
    p = _safe_file(CAPTURE_DIR, fname)
    return FileResponse(p)

@app.get("/uploads/{fname:path}")
def dl_upload(fname: str):
    p = _safe_file(UPLOADS_DIR, fname)
    return FileResponse(p)

import traceback

@app.post("/predict_cutout")
def predict_cutout(data: Optional[Dict[str, Any]] = Body(None)):
    data = data or {}
    ct_token = data.get("ct_token") or data.get("token")
    topk = int(data.get("topk", 5))

    if not ct_token:
        raise HTTPException(400, "ct_token manquant")

    meta = TEMP_SEGS_CUTOUT.get(ct_token)
    if not meta:
        raise HTTPException(404, "Cutout token inconnu ou expiré")

    png_bytes = meta["bytes"]

    try:
        out = predict_cutout_bytes(png_bytes, topk=topk)
        return {"ok": True, **out}
    except Exception as e:
        traceback.print_exc() 
        raise HTTPException(500, f"{type(e).__name__}: {e}")

# --------- Static files ----------
# IMPORTANT: on mount à la fin pour laisser tes routes API matcher d'abord.
app.mount("/", StaticFiles(directory=str(ROOT), html=True), name="static")

# --------- Main ---------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=5000, log_level="info")
