# engine.py — moteur Open3D (Filament/Legacy), encodage MJPEG, capture forcée
# plus CAPTURE OBJET SEUL en PNG RGBA (fond transparent) + HQ SSAA

import io, time, queue
import numpy as np
from PIL import Image
import open3d as o3d
from open3d.visualization import rendering as o3dr

# --------- TurboJPEG optionnel ---------
try:
    from turbojpeg import TurboJPEG, TJPF_RGB, TJSAMP_420
    _JPEG = TurboJPEG()
except Exception:
    _JPEG = None

# --------- Constantes & helpers locaux ----------
DEBUG_ZOOM = False  # log zoom si besoin

DEFAULT_OBJ_RGBA = np.array([0.69, 0.66, 0.64, 1.0], dtype=np.float32)  # '#b0a9a2'
DEFAULT_BG_RGB   = np.array([0.0, 0.0, 0.0], dtype=np.float32)          # noir

TRANSPARENT_BG_RGBA = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)   # fond transparent

def hex_to_rgb(h):
    h = h.strip().lstrip('#')
    return [int(h[0:2],16)/255.0, int(h[2:4],16)/255.0, int(h[4:6],16)/255.0]

def hex_to_rgba(h):
    r,g,b = hex_to_rgb(h); return [r,g,b,1.0]

def make_two_sided_mesh(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    m = o3d.geometry.TriangleMesh()
    v = np.asarray(mesh.vertices)
    t = np.asarray(mesh.triangles)
    m.vertices = o3d.utility.Vector3dVector(v.copy())
    m.triangles = o3d.utility.Vector3iVector(t.copy())
    t_rev = t[:, [0, 2, 1]]
    m2 = o3d.geometry.TriangleMesh()
    m2.vertices = o3d.utility.Vector3dVector(v.copy())
    m2.triangles = o3d.utility.Vector3iVector(t_rev)
    m = m + m2
    m.remove_duplicated_vertices()
    m.remove_duplicated_triangles()
    m.remove_degenerate_triangles()
    m.compute_vertex_normals()
    return m

# --------- État rendu ----------
class RenderState:
    def __init__(self):
        self.width = 640
        self.height = 360

        # caméra orbit (Filament)
        self.target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.dist   = 3.0
        self.theta  = np.deg2rad(45.0)
        self.phi    = np.deg2rad(25.0)
        self.fov_v_deg = 60.0

        # couleurs "live"
        self.bg = np.array([0.043, 0.0, 0.156], dtype=np.float32)  # '#0b0028'
        self.obj_color_rgba = DEFAULT_OBJ_RGBA.copy()

        self.invert_zoom = False

        # géométrie
        self.mesh = None
        self.mesh_name = "mesh"

        # moteur
        self.engine = None
        self.renderer = None
        self.scene = None
        self.material = None

        self.vis = None
        self.vc = None

        self.legacy_target = np.array([0.0, 0.0, 0.0], dtype=np.float64)

        self.latest_jpeg = None
        self.cmd_q = queue.Queue()
        self.running = True

        # cadence & qualité
        self.idle_fps = 20
        self.burst_fps = 40
        self.idle_quality = 82
        self.burst_quality = 45
        self.burst_until = 0.0

        # Verrou zoom
        self.zoom_locked = True

    # ---------- encodage ----------
    def _encode_jpeg(self, arr_uint8: np.ndarray, quality: int) -> bytes:
        if _JPEG is not None:
            return _JPEG.encode(arr_uint8, quality=quality, pixel_format=TJPF_RGB, subsampling=TJSAMP_420)
        with io.BytesIO() as buf:
            Image.fromarray(arr_uint8).save(buf, format="JPEG", quality=quality, optimize=True)
            return buf.getvalue()

    # ---------- utils image ----------
    def _to_uint8_rgb(self, arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 3 and arr.shape[2] == 4:
            arr = arr[:, :, :3]
        if arr.dtype != np.uint8:
            arr = (np.clip(arr, 0, 1) * 255.0 + 0.5).astype(np.uint8)
        return arr

    def _to_uint8_rgba(self, arr: np.ndarray) -> np.ndarray:
        """
        Force ndarray -> uint8 RGBA.
        Open3D Filament peut rendre en float [0..1] ou uint8 selon versions.
        """
        if arr.ndim == 2:
            arr = np.repeat(arr[:, :, None], 3, axis=2)

        if arr.ndim == 3 and arr.shape[2] == 3:
            if arr.dtype != np.uint8:
                rgb = (np.clip(arr, 0, 1) * 255.0 + 0.5).astype(np.uint8)
            else:
                rgb = arr
            a = np.full((rgb.shape[0], rgb.shape[1], 1), 255, dtype=np.uint8)
            return np.concatenate([rgb, a], axis=2)

        # 4 canaux
        if arr.dtype != np.uint8:
            arr = (np.clip(arr, 0, 1) * 255.0 + 0.5).astype(np.uint8)
        return arr

    # ---------- Filament ----------
    def _setup_material_filament(self):
        m = o3dr.MaterialRecord()
        m.base_color = self.obj_color_rgba.copy()
        m.shader = "defaultLit"
        try:
            m.double_sided = True
        except Exception:
            pass
        self.material = m
        if self.mesh is not None:
            try:
                self.scene.remove_geometry(self.mesh_name)
            except Exception:
                pass
            self.scene.add_geometry(self.mesh_name, self.mesh, self.material)

    def _camera_eye_forward(self):
        cp = np.cos(self.phi); sp = np.sin(self.phi)
        ct = np.cos(self.theta); st = np.sin(self.theta)
        eye = self.target + np.array([self.dist * cp * st, self.dist * sp, self.dist * cp * ct], dtype=np.float32)
        forward = self.target - eye
        n = np.linalg.norm(forward)
        forward = np.array([0.0, 0.0, -1.0], dtype=np.float32) if n < 1e-8 else (forward / n).astype(np.float32)
        return eye, forward

    def _update_headlight_filament(self, _, forward: np.ndarray):
        try:
            self.scene.scene.set_sun_light(forward.tolist(), [1.0, 1.0, 1.0], 65000.0)
            self.scene.scene.enable_sun_light(True)
            try:
                self.scene.scene.set_ambient_light([1.0, 1.0, 1.0], 0.22)
            except TypeError:
                self.scene.scene.set_ambient_light([0.22, 0.22, 0.22])
        except Exception:
            pass

    def _look_at_filament(self):
        aspect = float(self.width) / float(max(1, self.height))
        self.scene.camera.set_projection(self.fov_v_deg, aspect, 0.01, 1000.0, o3dr.Camera.FovType.Vertical)
        eye, forward = self._camera_eye_forward()
        self.scene.camera.look_at(self.target, eye, np.array([0.0,1.0,0.0], dtype=np.float32))
        self._update_headlight_filament(eye, forward)

    def _try_init_filament(self):
        self.renderer = o3dr.OffscreenRenderer(self.width, self.height)
        self.scene = self.renderer.scene
        self.scene.set_background(self.bg)
        self._setup_material_filament()
        self._look_at_filament()
        self.engine = "filament"
        print("[Render] Filament OffscreenRenderer actif (double_sided)")

    def _resize_filament(self, w, h):
        if self.renderer is not None:
            self.renderer.resize(w, h)
        self.width, self.height = w, h
        self._look_at_filament()

    # ---------- Legacy ----------
    def _init_legacy(self):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(width=self.width, height=self.height, visible=False)
        self.vc = self.vis.get_view_control()
        opt = self.vis.get_render_option()
        opt.background_color = self.bg.astype(float)
        opt.mesh_show_back_face = False
        opt.light_on = True
        if self.mesh is not None:
            self.vis.add_geometry(self.mesh)
            bbox = self.mesh.get_axis_aligned_bounding_box()
            self.legacy_target = np.asarray(bbox.get_center(), dtype=np.float64)
        else:
            self.legacy_target = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.engine = "legacy"
        print("[Render] Legacy Visualizer actif (géométrie deux faces)")

    def _resize_legacy(self, w, h):
        params = self.vc.convert_to_pinhole_camera_parameters()
        self.vis.destroy_window()
        self.width, self.height = w, h
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(width=w, height=h, visible=False)
        self.vc = self.vis.get_view_control()
        if self.mesh is not None:
            self.vis.add_geometry(self.mesh)
        self.vc.convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)
        opt = self.vis.get_render_option()
        opt.background_color = self.bg.astype(float)
        opt.mesh_show_back_face = False
        opt.light_on = True

    # ---------- Legacy : zoom ----------
    def _legacy_zoom_by_factor(self, factor: float):
        try:
            params = self.vc.convert_to_pinhole_camera_parameters()
            extr = np.array(params.extrinsic, dtype=np.float64)
            R = extr[:3, :3]; t = extr[:3, 3]
            eye = -R.T @ t
            target = np.asarray(self.legacy_target, dtype=np.float64)
            v = target - eye
            dist = float(np.linalg.norm(v))
            if dist < 1e-9:
                v = np.array([0.0, 0.0, 1.0], dtype=np.float64); dist = 1.0
            dir_cam = v / dist
            new_dist = float(np.clip(dist * float(factor), 1e-4, 1e6))
            eye_new = target - dir_cam * new_dist
            t_new = -R @ eye_new
            extr[:3, 3] = t_new
            params.extrinsic = extr
            self.vc.convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)
        except Exception:
            try:
                z = float(self.vc.get_zoom())
                self.vc.set_zoom(z * (1.0 / float(factor)))
            except Exception:
                pass

    # ---------- commun ----------
    def ensure_renderer(self):
        if self.engine is not None:
            return
        try:
            self._try_init_filament()
        except Exception as e:
            print("[Render] Filament indisponible, fallback Legacy:", e)
            self._init_legacy()

    def set_burst(self, seconds=0.4):
        self.burst_until = time.monotonic() + max(0.05, seconds)

    # =========================================================================
    # CAPTURE "classique" (fond noir + mesh gris) 
    # =========================================================================
    def _capture_forced_array(self, hq_w: int = None, hq_h: int = None):
        """
        Retourne ndarray uint8 (H,W,3) avec fond noir + mesh gris.
        Si hq_w/hq_h fournis: resize temporaire du renderer, capture, puis restore.
        """
        old_w, old_h = int(self.width), int(self.height)

        if hq_w and hq_h:
            hq_w = int(max(320, min(hq_w, 4096)))
            hq_h = int(max(240, min(hq_h, 4096)))

        def _temp_resize(new_w, new_h):
            if self.engine == "filament":
                self._resize_filament(new_w, new_h)
            else:
                self._resize_legacy(new_w, new_h)

        try:
            if hq_w and hq_h and (hq_w != old_w or hq_h != old_h):
                _temp_resize(hq_w, hq_h)

            if self.engine == "filament":
                old_bg = self.bg.copy()
                old_user = self.obj_color_rgba.copy()

                self.scene.set_background(DEFAULT_BG_RGB)
                self.obj_color_rgba = DEFAULT_OBJ_RGBA.copy()
                self._setup_material_filament()

                img = self.renderer.render_to_image()
                arr = np.asarray(img)
                arr = self._to_uint8_rgb(arr)

                # restore couleurs
                self.obj_color_rgba = old_user
                self._setup_material_filament()
                self.scene.set_background(old_bg)
                return arr

            else:
                opt = self.vis.get_render_option()
                old_bg = np.array(opt.background_color, dtype=float)
                user_rgb = np.asarray(self.obj_color_rgba[:3], dtype=float)

                try:
                    opt.background_color = DEFAULT_BG_RGB.astype(float)
                    if self.mesh is not None:
                        self.mesh.paint_uniform_color(DEFAULT_OBJ_RGBA[:3])
                        self.vis.update_geometry(self.mesh)

                    self.vis.poll_events()
                    self.vis.update_renderer()

                    img = self.vis.capture_screen_float_buffer(do_render=True)
                    arr = (np.asarray(img) * 255.0 + 0.5).astype(np.uint8)
                    if arr.ndim == 2:
                        arr = np.repeat(arr[:, :, None], 3, axis=2)
                    return arr

                finally:
                    opt.background_color = old_bg.astype(float)
                    if self.mesh is not None:
                        self.mesh.paint_uniform_color(user_rgb)
                        self.vis.update_geometry(self.mesh)

        finally:
            if hq_w and hq_h and (old_w != self.width or old_h != self.height):
                _temp_resize(old_w, old_h)

    def _capture_forced_array_hq(self, out_w: int, out_h: int, ssaa: int = 2):
        """
        Capture HQ indépendante du stream, fond noir + couleur neutre (comme capture classique).
        Retour: ndarray uint8 (out_h, out_w, 3)
        """
        ssaa = int(max(1, ssaa))
        out_w = int(max(64, out_w))
        out_h = int(max(64, out_h))

        render_w = int(out_w * ssaa)
        render_h = int(out_h * ssaa)

        MAX_SIDE = 4096
        s = min(1.0, MAX_SIDE / max(render_w, render_h))
        render_w = int(max(64, round(render_w * s)))
        render_h = int(max(64, round(render_h * s)))

        if self.engine == "filament":
            old_w, old_h = self.width, self.height
            old_bg = self.bg.copy()
            old_user = self.obj_color_rgba.copy()

            try:
                if (render_w, render_h) != (self.width, self.height):
                    self._resize_filament(render_w, render_h)

                self.scene.set_background(DEFAULT_BG_RGB)
                self.obj_color_rgba = DEFAULT_OBJ_RGBA.copy()
                self._setup_material_filament()

                img = self.renderer.render_to_image()
                arr = np.asarray(img)
                arr = self._to_uint8_rgb(arr)

            finally:
                self.obj_color_rgba = old_user
                self._setup_material_filament()
                self.scene.set_background(old_bg)

                if (self.width, self.height) != (old_w, old_h):
                    self._resize_filament(old_w, old_h)

        else:
            old_w, old_h = self.width, self.height
            opt = self.vis.get_render_option()
            old_bg = np.array(opt.background_color, dtype=float)
            user_rgb = np.asarray(self.obj_color_rgba[:3], dtype=float)

            try:
                if (render_w, render_h) != (self.width, self.height):
                    self._resize_legacy(render_w, render_h)

                opt.background_color = DEFAULT_BG_RGB.astype(float)
                if self.mesh is not None:
                    self.mesh.paint_uniform_color(DEFAULT_OBJ_RGBA[:3])
                    self.vis.update_geometry(self.mesh)

                self.vis.poll_events()
                self.vis.update_renderer()
                img = self.vis.capture_screen_float_buffer(do_render=True)
                arr = (np.asarray(img) * 255.0 + 0.5).astype(np.uint8)
                if arr.ndim == 2:
                    arr = np.repeat(arr[:, :, None], 3, axis=2)

            finally:
                opt.background_color = old_bg.astype(float)
                if self.mesh is not None:
                    self.mesh.paint_uniform_color(user_rgb)
                    self.vis.update_geometry(self.mesh)

                if (self.width, self.height) != (old_w, old_h):
                    self._resize_legacy(old_w, old_h)

        if (arr.shape[1], arr.shape[0]) != (out_w, out_h):
            im = Image.fromarray(arr, mode="RGB")
            im = im.resize((out_w, out_h), resample=Image.LANCZOS)
            arr = np.asarray(im, dtype=np.uint8)

        return arr

    # =========================================================================
    # CAPTURE OBJET SEUL (RGBA, fond transparent)
    # =========================================================================
    def _capture_object_rgba_array(self) -> np.ndarray:
        """
        Retourne ndarray uint8 (H,W,4) RGBA.
        - Filament: vrai fond transparent
        - Legacy: chroma-key magenta -> alpha approx (moins parfait)
        """
        if self.engine == "filament":
            old_bg = self.bg.copy()
            try:
                # fond transparent juste pour la capture
                self.scene.set_background(TRANSPARENT_BG_RGBA)
                img = self.renderer.render_to_image()
                arr = np.asarray(img)
                rgba = self._to_uint8_rgba(arr)
                return rgba
            finally:
                # restore fond user pour le streaming
                try:
                    self.scene.set_background(old_bg)
                except Exception:
                    pass

        # -------- Legacy fallback: chroma key --------
        opt = self.vis.get_render_option()
        old_bg = np.array(opt.background_color, dtype=float)

        key_rgb_float = np.array([1.0, 0.0, 1.0], dtype=float)  # magenta
        key_u8 = np.array([255, 0, 255], dtype=np.uint8)

        try:
            opt.background_color = key_rgb_float
            self.vis.poll_events()
            self.vis.update_renderer()

            img = self.vis.capture_screen_float_buffer(do_render=True)
            rgb = (np.clip(np.asarray(img), 0, 1) * 255.0 + 0.5).astype(np.uint8)
            if rgb.ndim == 2:
                rgb = np.repeat(rgb[:, :, None], 3, axis=2)

            # diff au magenta => alpha
            diff = np.max(np.abs(rgb.astype(np.int16) - key_u8[None, None, :]), axis=2)

            # alpha "soft" (anti-halo)
            t0, t1 = 8, 36
            alpha = np.clip((diff - t0) * 255.0 / max(1, (t1 - t0)), 0, 255).astype(np.uint8)

            # Option : prémultiplication pour éviter bord sombre en viewer
            premul = (rgb.astype(np.uint16) * alpha[:, :, None].astype(np.uint16) // 255).astype(np.uint8)

            return np.dstack([premul, alpha])

        finally:
            opt.background_color = old_bg.astype(float)

    def _capture_object_rgba_array_hq(self, out_w: int, out_h: int, ssaa: int = 2) -> np.ndarray:
        """
        Capture objet seul en HQ (RGBA), indépendante du stream.
        - out_w/out_h : taille finale
        - ssaa : supersampling (2 recommandé)
        Retour: ndarray uint8 (out_h, out_w, 4)
        """
        ssaa = int(max(1, ssaa))
        out_w = int(max(64, out_w))
        out_h = int(max(64, out_h))

        render_w = int(out_w * ssaa)
        render_h = int(out_h * ssaa)

        MAX_SIDE = 4096
        s = min(1.0, MAX_SIDE / max(render_w, render_h))
        render_w = int(max(64, round(render_w * s)))
        render_h = int(max(64, round(render_h * s)))

        old_w, old_h = int(self.width), int(self.height)

        def _temp_resize(new_w, new_h):
            if self.engine == "filament":
                self._resize_filament(new_w, new_h)
            else:
                self._resize_legacy(new_w, new_h)

        try:
            if (render_w, render_h) != (self.width, self.height):
                _temp_resize(render_w, render_h)

            rgba_big = self._capture_object_rgba_array()

        finally:
            if (self.width, self.height) != (old_w, old_h):
                _temp_resize(old_w, old_h)

        if (rgba_big.shape[1], rgba_big.shape[0]) != (out_w, out_h):
            im = Image.fromarray(rgba_big, mode="RGBA")
            im = im.resize((out_w, out_h), resample=Image.LANCZOS)
            rgba_big = np.asarray(im, dtype=np.uint8)

        return rgba_big

    # ---------- commandes ----------
    def _apply_cmd(self, cmd: dict):
        typ = cmd.get("action")

        # --- capture objet seul RGBA ---
        if typ == "capture_object":
            result_q = cmd.get("result_q", None)
            try:
                arr = self._capture_object_rgba_array()
                if result_q is not None:
                    result_q.put(("ok", arr))
            except Exception as e:
                if result_q is not None:
                    result_q.put(("err", str(e)))
            return

        if typ == "capture_object_hq":
            result_q = cmd.get("result_q", None)
            try:
                out_w = int(cmd.get("out_w", self.width))
                out_h = int(cmd.get("out_h", self.height))
                ssaa  = int(cmd.get("ssaa", 2))
                arr = self._capture_object_rgba_array_hq(out_w, out_h, ssaa=ssaa)
                if result_q is not None:
                    result_q.put(("ok", arr))
            except Exception as e:
                if result_q is not None:
                    result_q.put(("err", str(e)))
            return

        # --- captures existantes ---
        if typ == "capture_hq":
            result_q = cmd.get("result_q", None)
            try:
                out_w = int(cmd.get("out_w", self.width))
                out_h = int(cmd.get("out_h", self.height))
                ssaa  = int(cmd.get("ssaa", 2))
                arr = self._capture_forced_array_hq(out_w, out_h, ssaa=ssaa)
                if result_q is not None:
                    result_q.put(("ok", arr))
            except Exception as e:
                if result_q is not None:
                    result_q.put(("err", str(e)))
            return

        if typ == "capture_forced":
            result_q = cmd.get("result_q", None)
            try:
                hq = cmd.get("hq") or {}
                hq_w = hq.get("width")
                hq_h = hq.get("height")
                arr = self._capture_forced_array(hq_w=hq_w, hq_h=hq_h)
                if result_q is not None:
                    result_q.put(("ok", arr))
            except Exception as e:
                if result_q is not None:
                    result_q.put(("err", str(e)))
            return

        if typ in ("rotate","pan","zoom","zoom_dir","zoom_sem","settings","lights","load_obj","fit","reset","remove"):
            self.set_burst()

        if typ in ("reset","fit","load_obj","remove"):
            self.zoom_locked = True

        if typ in ("reset","fit"):
            if self.engine == "filament":
                if self.mesh is not None:
                    bbox = self.mesh.get_axis_aligned_bounding_box()
                    c = np.asarray(bbox.get_center(), dtype=np.float32)
                    extent = np.asarray(bbox.get_extent(), dtype=np.float32)
                    diag = float(np.linalg.norm(extent)) or 1.0
                    self.target = c
                    self.theta = np.deg2rad(45); self.phi = np.deg2rad(25)
                    self.dist = (0.5*diag) / np.tan(np.deg2rad(self.fov_v_deg/2.0)) * 1.35
                else:
                    self.target[:] = 0; self.theta=np.deg2rad(45); self.phi=np.deg2rad(25); self.dist=3.0
                self._look_at_filament()
            else:
                if self.mesh is not None:
                    bbox = self.mesh.get_axis_aligned_bounding_box()
                    center = bbox.get_center()
                    front = np.array([-0.577,-0.577,-0.577])
                    up = np.array([0.0,1.0,0.0])
                    self.vc.set_lookat(center.tolist())
                    self.vc.set_front((front/ (np.linalg.norm(front)+1e-8)).tolist())
                    self.vc.set_up(up.tolist())
                    self.vc.set_zoom(0.8)
                    self.legacy_target = np.asarray(center, dtype=np.float64)
                else:
                    self.vc.set_lookat([0,0,0]); self.vc.set_front([-0.577,-0.577,-0.577]); self.vc.set_up([0,1,0]); self.vc.set_zoom(0.8)
                    self.legacy_target = np.array([0.0, 0.0, 0.0], dtype=np.float64)
            return

        if typ == "remove":
            if self.engine == "filament":
                try: self.scene.remove_geometry(self.mesh_name)
                except Exception: pass
            else:
                try: self.vis.remove_geometry(self.mesh, reset_bounding_box=False)
                except Exception: pass
                self.legacy_target = np.array([0.0, 0.0, 0.0], dtype=np.float64)
            self.mesh = None
            return

        if typ == "load_obj":
            from pathlib import Path
            path = Path(cmd.get("path",""))
            try:
                mesh = o3d.io.read_triangle_mesh(str(path))
                if mesh.is_empty():
                    return
                mesh.compute_vertex_normals()
                bbox = mesh.get_axis_aligned_bounding_box()
                extent = bbox.get_extent()
                max_dim = max(float(extent[0]), float(extent[1]), float(extent[2]), 1e-4)
                s = 2.5 / max_dim
                mesh.scale(s, center=bbox.get_center())
                bbox = mesh.get_axis_aligned_bounding_box()
                mesh.translate(-bbox.get_center())
                bbox = mesh.get_axis_aligned_bounding_box()
                mesh.translate((0.0, -float(bbox.min_bound[1]), 0.0))

                if self.engine == "filament":
                    if self.mesh is not None:
                        try:
                            self.scene.remove_geometry(self.mesh_name)
                        except Exception:
                            pass
                    self.mesh = mesh
                    self.scene.add_geometry(self.mesh_name, self.mesh, self.material)
                else:
                    mesh2 = make_two_sided_mesh(mesh)
                    if self.mesh is not None:
                        try:
                            self.vis.remove_geometry(self.mesh, reset_bounding_box=False)
                        except Exception:
                            pass
                    self.mesh = mesh2
                    self.vis.add_geometry(self.mesh)

                bbox = self.mesh.get_axis_aligned_bounding_box()
                center = np.asarray(bbox.get_center(), dtype=np.float64)
                if self.engine == "filament":
                    self.target = center.astype(np.float32)
                    self._look_at_filament()
                else:
                    self.legacy_target = center
                    self.vc.set_lookat(center.tolist())

                self._apply_cmd({"action": "fit"})

            except Exception as e:
                print("[load_obj] erreur:", e)
            return

        if typ == "settings":
            if "invert_zoom" in cmd:
                self.invert_zoom = bool(cmd.get("invert_zoom"))
            if "obj_color" in cmd:
                self.obj_color_rgba = np.array(hex_to_rgba(cmd["obj_color"]), dtype=np.float32)
                if self.engine == "filament":
                    self._setup_material_filament()
                else:
                    if self.mesh is not None:
                        self.mesh.paint_uniform_color(self.obj_color_rgba[:3])
                        self.vis.update_geometry(self.mesh)
            if "bg" in cmd:
                if self.engine == "filament":
                    self.bg = np.array(hex_to_rgb(cmd["bg"]), dtype=np.float32)
                    self.scene.set_background(self.bg)
                else:
                    self.bg = np.array(hex_to_rgb(cmd["bg"]), dtype=np.float32)
                    opt = self.vis.get_render_option()
                    opt.background_color = self.bg.astype(float)
            return

        if typ == "lights":
            return

        if typ == "zoom_sem":
            if self.zoom_locked:
                if DEBUG_ZOOM: print("[zoom_sem] bloqué: aucune rotation encore effectuée")
                return
            sem = str(cmd.get("sem", "in")).lower()
            step = float(cmd.get("step", 1.12))
            step = min(max(step, 1.02), 1.5)
            if self.engine == "filament":
                factor = (1.0/step) if sem == "in" else step
                self.dist = float(np.clip(self.dist * factor, 0.2, 1000.0))
                self._look_at_filament()
            else:
                factor = (1.0/step) if sem == "in" else step
                self._legacy_zoom_by_factor(factor)
            return

        if typ in ("rotate","pan","zoom","zoom_dir"):
            if DEBUG_ZOOM and typ.startswith("zoom"):
                print(f"[zoom] engine={self.engine} cmd={cmd}")

            if self.engine == "filament":
                if typ == "rotate":
                    dx = float(cmd.get("dx",0)); dy = float(cmd.get("dy",0)); s = 0.005
                    self.theta -= dx*s; self.phi -= dy*s
                    self.phi = float(np.clip(self.phi, np.deg2rad(-85), np.deg2rad(85)))
                    self._look_at_filament()
                    if abs(dx) + abs(dy) > 1e-9:
                        self.zoom_locked = False
                elif typ == "pan":
                    dx = float(cmd.get("dx",0)); dy = float(cmd.get("dy",0))
                    cp = np.cos(self.phi); sp = np.sin(self.phi); ct = np.cos(self.theta); st = np.sin(self.theta)
                    forward = -np.array([cp*st, sp, cp*ct], dtype=np.float32)
                    right = np.cross(forward, np.array([0,1,0], dtype=np.float32)); right /= (np.linalg.norm(right)+1e-8)
                    up_cam = np.cross(right, forward); up_cam /= (np.linalg.norm(up_cam)+1e-8)
                    k = 0.0015 * self.dist
                    self.target += (-dx*k)*right + (dy*k)*up_cam
                    self._look_at_filament()
                elif typ == "zoom":
                    if self.zoom_locked:
                        if DEBUG_ZOOM: print("[zoom] bloqué: aucune rotation encore effectuée")
                        return
                    scale = float(cmd.get("scale",1.0))
                    if self.invert_zoom: scale = 1.0 / max(1e-6, scale)
                    self.dist = float(np.clip(self.dist * scale, 0.2, 1000.0))
                    self._look_at_filament()
                else:
                    if self.zoom_locked:
                        if DEBUG_ZOOM: print("[zoom_dir] bloqué: aucune rotation encore effectuée")
                        return
                    dirv = int(cmd.get("dir", 1))
                    if self.invert_zoom: dirv = -dirv
                    step = float(cmd.get("step", 1.12))
                    step = min(max(step, 1.02), 1.5)
                    factor = step if dirv > 0 else (1.0/step)
                    self.dist = float(np.clip(self.dist * factor, 0.2, 1000.0))
                    self._look_at_filament()

            else:
                if typ == "rotate":
                    dx = float(cmd.get("dx",0)); dy = float(cmd.get("dy",0))
                    self.vc.rotate(dx, dy)
                    if abs(dx) + abs(dy) > 1e-9:
                        self.zoom_locked = False
                elif typ == "pan":
                    self.vc.translate(float(cmd.get("dx",0)), float(cmd.get("dy",0)))
                elif typ == "zoom":
                    if self.zoom_locked:
                        if DEBUG_ZOOM: print("[zoom legacy] bloqué: aucune rotation encore effectuée")
                        return
                    s = float(cmd.get("scale",1.0))
                    if self.invert_zoom:
                        s = 1.0 / max(1e-6, s)
                    self._legacy_zoom_by_factor(s)
                else:
                    if self.zoom_locked:
                        if DEBUG_ZOOM: print("[zoom_dir legacy] bloqué: aucune rotation encore effectuée")
                        return
                    dirv = int(cmd.get("dir", 1))
                    if self.invert_zoom:
                        dirv = -dirv
                    step = float(cmd.get("step", 1.12))
                    step = min(max(step, 1.02), 1.5)
                    factor = step if dirv > 0 else (1.0/step)
                    self._legacy_zoom_by_factor(factor)
            return

        if typ == "resize":
            w_req = int(cmd.get("width", self.width))
            h_req = int(cmd.get("height", self.height))

            w_req = max(320, w_req)
            h_req = max(240, h_req)

            MAX_W, MAX_H = 2560, 1440
            k = min(1.0, MAX_W / float(w_req), MAX_H / float(h_req))

            w = int(round(w_req * k))
            h = int(round(h_req * k))

            if self.engine == "filament":
                self._resize_filament(w, h)
            else:
                self._resize_legacy(w, h)
            return

    # ---------- boucle render ----------
    def render_once(self):
        self.ensure_renderer()

        start = time.monotonic()

        # 1) commandes
        while True:
            try:
                cmd = self.cmd_q.get_nowait()
            except queue.Empty:
                break
            self._apply_cmd(cmd)

        # 2) rendu (stream)
        if self.engine == "filament":
            img = self.renderer.render_to_image()
            arr = np.asarray(img)
            arr = self._to_uint8_rgb(arr)
        else:
            self.vis.poll_events()
            self.vis.update_renderer()
            img = self.vis.capture_screen_float_buffer(do_render=True)
            arr = (np.asarray(img) * 255.0 + 0.5).astype(np.uint8)
            if arr.ndim == 2:
                arr = np.repeat(arr[:, :, None], 3, axis=2)

        # 3) encode jpeg
        q = self.burst_quality if time.monotonic() < self.burst_until else self.idle_quality
        self.latest_jpeg = self._encode_jpeg(arr, q)

        # 4) timing
        fps = self.burst_fps if time.monotonic() < self.burst_until else self.idle_fps
        fps = max(1.0, float(fps))
        frame_dt = 1.0 / fps

        elapsed = time.monotonic() - start
        remaining = frame_dt - elapsed
        if remaining <= 0.0:
            return

        slept = 0.0
        step = 0.002 if fps >= 40 else 0.004

        while slept < remaining and self.running:
            time.sleep(step)
            slept += step
            if not self.cmd_q.empty():
                break

STATE = RenderState()

def render_loop():
    while STATE.running:
        STATE.render_once()
