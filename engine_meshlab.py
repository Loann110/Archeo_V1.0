# engine_meshlab.py — MeshLab APSS + rendu MJPEG avec fallback Filament -> Legacy (contrôles fixés)
import io, time, queue, os, uuid, hashlib
from pathlib import Path
import numpy as np
from PIL import Image

import open3d as o3d
from open3d.visualization import rendering as o3dr

# --------- PyMeshLab ----------
import pymeshlab as ml

# --------- TurboJPEG optionnel ----------
try:
    from turbojpeg import TurboJPEG, TJPF_RGB, TJSAMP_420
    _JPEG = TurboJPEG()
except Exception:
    _JPEG = None

ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "ml_cache"
CACHE_DIR.mkdir(exist_ok=True)

DEFAULT_OBJ_RGBA = np.array([0.69, 0.66, 0.64, 1.0], dtype=np.float32)
DEFAULT_BG_RGB   = np.array([0.0, 0.0, 0.0], dtype=np.float32)


def hex_to_rgb(h: str):
    h = h.strip().lstrip('#')
    return [int(h[0:2], 16)/255.0, int(h[2:4], 16)/255.0, int(h[4:6], 16)/255.0]


def hex_to_rgba(h: str):
    r, g, b = hex_to_rgb(h)
    return [r, g, b, 1.0]


def _encode_jpeg(arr_uint8: np.ndarray, quality: int) -> bytes:
    if _JPEG is not None:
        return _JPEG.encode(arr_uint8, quality=quality, pixel_format=TJPF_RGB, subsampling=TJSAMP_420)
    with io.BytesIO() as buf:
        Image.fromarray(arr_uint8).save(buf, format="JPEG", quality=quality, optimize=True)
        return buf.getvalue()


# ========= Ancien helper SHA1  =========
def file_sha1(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def apss_cache_path(
    src_path: Path,
    filterscale: float,
    sphericalparameter: float,
    curvaturetype: str,
    rotate_flag: bool
) -> Path:
    """
    Génère un chemin de fichier de cache APSS qui dépend
    du mesh normalisé (via son nom) ET des paramètres MeshLab.
    On évite de relire le fichier pour calculer un SHA1 : plus rapide.
    """
    base_id = src_path.stem  # ex: ml_src_xxx
    key = f"{base_id}|fs={filterscale:.4f}|sp={sphericalparameter:.4f}|ct={curvaturetype}|rot={int(bool(rotate_flag))}"
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return CACHE_DIR / f"{h}.ply"


# ========= Ajout : epaisseur (coquille double) =========
def _add_thickness(mesh: o3d.geometry.TriangleMesh, thickness: float = 0.0025):
    """
    Ajoute une épaisseur en dupliquant la surface
    le long des normales de vertices et en ajoutant une couche
    interne avec triangles inversés.
    """
    try:
        if mesh.is_empty():
            return mesh
        mesh.compute_vertex_normals()
        v = np.asarray(mesh.vertices)
        n = np.asarray(mesh.vertex_normals)
        if v.shape != n.shape or v.size == 0:
            return mesh

        # couche interne légèrement décalée
        v2 = v - n * float(thickness)

        verts = np.vstack([v, v2])
        tri = np.asarray(mesh.triangles)
        if tri.size == 0:
            mesh.vertices = o3d.utility.Vector3dVector(verts)
            mesh.compute_vertex_normals()
            return mesh

        # triangles internes inversés
        tri_inner = tri[:, ::-1] + len(v)
        all_tri = np.vstack([tri, tri_inner])

        thick = o3d.geometry.TriangleMesh()
        thick.vertices = o3d.utility.Vector3dVector(verts)
        thick.triangles = o3d.utility.Vector3iVector(all_tri)
        thick.compute_vertex_normals()
        return thick
    except Exception as e:
        print("[ML] _add_thickness error:", e)
        return mesh


# ==== courbure =====
# Mapping vers les enums EXACTS attendus par PyMeshLab
# (Mean, Gauss, K1, K2, ApproxMean)
_CURV_TYPE_MAP = {
    # valeurs "canoniques"
    "mean":        "Mean",
    "approxmean":  "ApproxMean",
    "gauss":       "Gauss",
    "k1":          "K1",
    "k2":          "K2",

    # alias pratico-pratiques
    "gaussian":      "Gauss",
    "principal_max": "K1",
    "principal_min": "K2",
}


class MeshLabEngineState:
    def __init__(self):
        # viewport
        self.width = 480
        self.height = 320

        # caméra orbit
        self.target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.dist   = 3.0
        self.theta  = np.deg2rad(45.0)
        self.phi    = np.deg2rad(25.0)
        self.fov_v_deg = 60.0

        # couleurs
        self.bg = np.array([0.043, 0.0, 0.156], dtype=np.float32)
        self.obj_color_rgba = DEFAULT_OBJ_RGBA.copy()
        self.invert_zoom = False

        # rendu
        self.engine = "meshlab"
        self.mode = None
        self.renderer = None
        self.scene = None
        self.material = None
        self.vis = None
        self.vc = None
        self.legacy_target = np.array([0.0, 0.0, 0.0], dtype=np.float64)

        self.mesh_name = "mesh"
        self.mesh = None
        self._mesh_in_scene = False
        self.latest_jpeg = None

        # fichiers
        self.src_path: Path | None = None           # mesh normalisé
        self.colored_path: Path | None = None       # mesh APSS caché

        # MeshLab params
        self.filterscale = 2.0
        self.sphericalparameter = 1.0
        self.curvaturetype = "mean"
        self.rotate_flag = False

        # Command queue
        self.cmd_q = queue.Queue()
        self.running = True

        # fps
        self.idle_fps = 20
        self.burst_fps = 40
        self.idle_quality = 75
        self.burst_quality = 65
        self.burst_until = 0.0

        self.zoom_locked = True

    # ---------- init ----------
    def _try_init_filament(self):
        self.renderer = o3dr.OffscreenRenderer(self.width, self.height)
        self.scene = self.renderer.scene
        self.scene.set_background(self.bg)
        m = o3dr.MaterialRecord()
        m.base_color = self.obj_color_rgba.copy()
        m.shader = "defaultLit"
        try:
            m.double_sided = True
        except Exception:
            pass
        self.material = m
        self.mode = "filament"
        print("[ML] Filament OffscreenRenderer actif")

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
        self.mode = "legacy"
        print("[ML] Legacy Visualizer actif")

    def ensure_renderer(self):
        if self.mode is not None:
            return
        try:
            self._try_init_filament()
        except Exception as e:
            print("[ML] Filament indisponible, fallback:", e)
            self._init_legacy()
        self._bind_mesh()

    # ---------- camera ----------
    def _look_at_filament(self):
        aspect = float(self.width) / float(max(1, self.height))
        self.scene.camera.set_projection(self.fov_v_deg, aspect, 0.01, 1000.0, o3dr.Camera.FovType.Vertical)
        cp = np.cos(self.phi)
        sp = np.sin(self.phi)
        ct = np.cos(self.theta)
        st = np.sin(self.theta)
        eye = self.target + np.array(
            [self.dist * cp * st, self.dist * sp, self.dist * cp * ct],
            dtype=np.float32
        )
        self.scene.camera.look_at(self.target, eye, np.array([0.0, 1.0, 0.0], dtype=np.float32))

    # ---------- legacy zoom ----------
    def _legacy_zoom_by_factor(self, factor: float):
        try:
            params = self.vc.convert_to_pinhole_camera_parameters()
            extr = np.array(params.extrinsic, dtype=np.float64)
            R = extr[:3, :3]
            t = extr[:3, 3]
            eye = -R.T @ t
            target = np.asarray(self.legacy_target, dtype=np.float64)
            v = target - eye
            dist = float(np.linalg.norm(v))
            if dist < 1e-9:
                v = np.array([0.0, 0.0, 1.0], dtype=np.float64)
                dist = 1.0
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

    # ---------- mesh binding ----------
    def _bind_mesh(self):
        if self.mesh is None:
            self._mesh_in_scene = False
            return
        if self.mode == "filament":
            try:
                self.scene.remove_geometry(self.mesh_name)
            except Exception:
                pass
            try:
                self.scene.add_geometry(self.mesh_name, self.mesh, self.material)
                self._mesh_in_scene = True
            except Exception:
                self._mesh_in_scene = False
        else:
            try:
                if self._mesh_in_scene:
                    self.vis.remove_geometry(self.mesh, reset_bounding_box=False)
            except Exception:
                pass
            try:
                self.vis.add_geometry(self.mesh)
                self._mesh_in_scene = True
            except Exception:
                self._mesh_in_scene = False

    def _refresh_material(self):
        if self.mode != "filament":
            return
        if self.material is None:
            m = o3dr.MaterialRecord()
            m.base_color = self.obj_color_rgba.copy()
            m.shader = "defaultLit"
            try:
                m.double_sided = True
            except Exception:
                pass
            self.material = m
        self._bind_mesh()

    # ---------- encode ----------
    def _encode_frame(self, quality: int):
        if self.mode == "filament":
            img = self.renderer.render_to_image()
            arr = np.asarray(img)
            if arr.ndim == 3 and arr.shape[2] == 4:
                arr = arr[:, :, :3]
            if arr.dtype != np.uint8:
                arr = (np.clip(arr, 0, 1) * 255.0 + 0.5).astype(np.uint8)
        else:
            self.vis.poll_events()
            self.vis.update_renderer()
            img = self.vis.capture_screen_float_buffer(do_render=True)
            arr = (np.asarray(img) * 255.0 + 0.5).astype(np.uint8)
            if arr.ndim == 2:
                arr = np.repeat(arr[:, :, None], 3, axis=2)
        self.latest_jpeg = _encode_jpeg(arr, quality)

    def set_burst(self, seconds=0.4):
        self.burst_until = time.monotonic() + max(0.05, seconds)

    # ============================================================
    #  _rebuild_colored_mesh PATCHÉ (décimation + APSS plus rapide)
    # ============================================================
    def _rebuild_colored_mesh(self):
        if not self.src_path or not Path(self.src_path).exists():
            return

        # ---- clé de cache basée sur ID + paramètres APSS ----
        out = apss_cache_path(
            Path(self.src_path),
            float(self.filterscale),
            float(self.sphericalparameter),
            str(self.curvaturetype),
            bool(self.rotate_flag),
        )

        # ---- 1) si cache existe : chargement instantané ----
        if out.exists():
            try:
                mesh = o3d.io.read_triangle_mesh(str(out))
                if not mesh.is_empty():
                    mesh.compute_vertex_normals()

                    # on remplace le mesh courant par le mesh APSS
                    self.mesh = mesh
                    self.colored_path = out

                    # important : on force un rebinding propre
                    self._mesh_in_scene = False
                    if self.mode == "legacy" and self.vis is not None:
                        try:
                            self.vis.clear_geometries()
                        except Exception:
                            pass

                    print("[ML] Cache APSS chargé :", out.name)

                    # petit burst pour que le flux MJPEG soit rafraîchi
                    self.set_burst(0.8)
                    return
            except Exception as e:
                print("[ML] Erreur lecture cache, recalcul APSS :", e)

        print("[ML] Aucun cache APSS -> calcul complet…")

        # ---- 2) Appliquer les filtres APSS avec MeshLab ----
        ms = ml.MeshSet()
        ms.load_new_mesh(str(self.src_path))

        # éventuelle rotation 180° Y
        if self.rotate_flag:
            try:
                ms.apply_filter("transform_rotate", rotaxis="Y axis", angle=180.0)
            except Exception:
                pass

        # ----------------------------------------------------
        #  2.a DÉCIMATION RAPIDE SI MESH TROP LOURD
        # ----------------------------------------------------
        try:
            m0 = ms.current_mesh()
            face_count = m0.face_number()
        except Exception:
            face_count = 0

        # seuils à ajuster selon tes modèles
        MAX_FACES = 200_000       # au-delà de ça, on simplifie
        TARGET_FACES = 120_000    # objectif de faces après décimation

        if face_count and face_count > MAX_FACES:
            try:
                target = min(TARGET_FACES, int(face_count * 0.6))
                t0 = time.time()
                print(f"[ML] Décimation pré-APSS : {face_count} -> {target} faces (approx)…")

                # le nom exact du filtre peut varier selon la version de PyMeshLab
                ms.apply_filter(
                    "simplify_quadric_edge_collapse",
                    targetfacenum=target,
                    preservenormal=True,
                    preserveboundary=True
                )

                t1 = time.time()
                m1 = ms.current_mesh()
                face_after = m1.face_number()
                print(f"[ML] Décimation OK : {face_count} -> {face_after} faces en {t1 - t0:.2f}s")
            except Exception as e:
                print("[ML] Décimation pré-APSS ignorée :", e)

        # ----------------------------------------------------
        #  2.b APSS avec moins d’itérations (plus rapide)
        # ----------------------------------------------------
        key = str(self.curvaturetype).lower()
        ctype = _CURV_TYPE_MAP.get(key, "Mean")

        MAX_ITERS = 12 # Max iterations pour APSS default 15 on meshlab

        t_apss0 = time.time()
        ms.compute_curvature_and_color_apss_per_vertex(
            filterscale=float(self.filterscale),
            maxprojectioniters=MAX_ITERS,
            sphericalparameter=float(self.sphericalparameter),
            curvaturetype=ctype,
            projectionaccuracy=0.0001
        )
        t_apss1 = time.time()
        print(f"[ML] APSS terminé en {t_apss1 - t_apss0:.2f}s (iters={MAX_ITERS})")

        # ---- 3) Sauvegarde APSS dans le cache ----
        ms.save_current_mesh(str(out))
        self.colored_path = out

        # ---- 4) Charger APSS pour Open3D ----
        try:
            mesh = o3d.io.read_triangle_mesh(str(out))
            if not mesh.is_empty():
                mesh.compute_vertex_normals()

                # on remplace le mesh courant par le mesh APSS
                self.mesh = mesh

                # >>> important : on force un rebinding propre
                self._mesh_in_scene = False
                if self.mode == "legacy" and self.vis is not None:
                    try:
                        self.vis.clear_geometries()
                    except Exception:
                        pass

                print("[ML] APSS recalculé et mis en cache :", out.name)

                # burst pour pousser le nouveau rendu tout de suite
                self.set_burst(0.8)
        except Exception as e:
            print("[ML] read_triangle_mesh error:", e)
            self._mesh_in_scene = False


    # ---------- capture ----------
    def _capture_forced_array(self):
        if self.mode == "filament":
            old_bg = self.bg.copy()
            old_user = self.obj_color_rgba.copy()
            self.scene.set_background(DEFAULT_BG_RGB)
            self.obj_color_rgba = DEFAULT_OBJ_RGBA.copy()
            self._refresh_material()
            img = self.renderer.render_to_image()
            arr = np.asarray(img)
            if arr.ndim == 3 and arr.shape[2] == 4:
                arr = arr[:, :, :3]
            if arr.dtype != np.uint8:
                arr = (np.clip(arr, 0, 1) * 255.0 + 0.5).astype(np.uint8)
            self.obj_color_rgba = old_user
            self._refresh_material()
            self.scene.set_background(old_bg)
            return arr
        else:
            opt = self.vis.get_render_option()
            old_bg = np.array(opt.background_color, dtype=float)
            backup = None
            try:
                if self.mesh is not None and self.mesh.has_vertex_colors():
                    backup = np.asarray(self.mesh.vertex_colors).copy()
            except Exception:
                backup = None
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
            finally:
                opt.background_color = old_bg.astype(float)
                if self.mesh is not None and backup is not None:
                    try:
                        self.mesh.vertex_colors = o3d.utility.Vector3dVector(backup)
                        self.vis.update_geometry(self.mesh)
                    except Exception:
                        pass
            return arr

    # ---------- commandes ----------
    def _apply_cmd(self, cmd: dict):
        typ = cmd.get("action")

        if typ == "capture_forced":
            result_q = cmd.get("result_q")
            try:
                arr = self._capture_forced_array()
                if result_q is not None:
                    result_q.put(("ok", arr))
            except Exception as e:
                if result_q is not None:
                    result_q.put(("err", str(e)))
            return

        if typ in (
            "reset", "fit", "load_obj", "remove",
            "rotate", "pan", "zoom", "zoom_dir", "zoom_sem",
            "settings", "resize", "ml_params"
        ):
            self.set_burst()

        if typ in ("reset", "fit", "load_obj", "remove"):
            self.zoom_locked = True

        # ---------- load_obj : affiche le mesh brut, APSS seulement sur Apply ----------
        if typ == "load_obj":
            path = Path(cmd.get("path", ""))
            if not path.exists():
                return

            try:
                # normalisation comme avant
                m = o3d.io.read_triangle_mesh(str(path))
                if m.is_empty():
                    return

                bbox = m.get_axis_aligned_bounding_box()
                extent = bbox.get_extent()
                max_dim = max(
                    float(extent[0]),
                    float(extent[1]),
                    float(extent[2]),
                    1e-4
                )

                s = 2.5 / max_dim
                m.scale(s, center=bbox.get_center())

                bbox = m.get_axis_aligned_bounding_box()
                m.translate(-bbox.get_center())

                bbox = m.get_axis_aligned_bounding_box()
                m.translate((0.0, -float(bbox.min_bound[1]), 0.0))

                # ---- Ajout : épaisseur comme dans engine ----
                m = _add_thickness(m, thickness=0.0025)

                # on écrit la version normalisée pour MeshLab/APSS
                tmp_src = CACHE_DIR / f"ml_src_{uuid.uuid4().hex}.ply"
                o3d.io.write_triangle_mesh(str(tmp_src), m, write_ascii=False)
                self.src_path = tmp_src

                # on affiche directement le mesh brut (pas d'APSS ici)
                self.mesh = m
                self._mesh_in_scene = False
                if self.mode is not None:
                    self._bind_mesh()

            except Exception as e:
                print("[ML] load_obj normalize error:", e)
                self.src_path = path
                # fallback : tenter d'afficher le mesh original
                try:
                    m = o3d.io.read_triangle_mesh(str(path))
                    if not m.is_empty():
                        self.mesh = m
                        self._mesh_in_scene = False
                        if self.mode is not None:
                            self._bind_mesh()
                except Exception as ee:
                    print("[ML] load_obj fallback error:", ee)
                    return

            # cadrage caméra sur le mesh courant (brut ou APSS si déjà appliqué)
            if self.mesh is not None:
                bbox = self.mesh.get_axis_aligned_bounding_box()
                c = np.asarray(bbox.get_center(), dtype=np.float32)
                extent = np.asarray(bbox.get_extent(), dtype=np.float32)
                diag = float(np.linalg.norm(extent)) or 1.0

                self.target = c
                self.theta = np.deg2rad(45)
                self.phi = np.deg2rad(25)
                self.dist = (0.5 * diag) / np.tan(np.deg2rad(self.fov_v_deg/2.0)) * 1.35

                if self.mode == "legacy":
                    center = np.asarray(c, dtype=np.float64)
                    self.legacy_target = center
                    self.vc.set_lookat(center.tolist())
                    self.vc.set_front((-np.array([1, 1, 1]) / np.sqrt(3)).tolist())
                    self.vc.set_up([0, 1, 0])
                    self.vc.set_zoom(0.8)

            return

        if typ == "ml_params":
            if "filterscale" in cmd:
                self.filterscale = float(cmd["filterscale"])
            if "sphericalparameter" in cmd:
                self.sphericalparameter = float(cmd["sphericalparameter"])
            if "curvaturetype" in cmd:
                v = str(cmd["curvaturetype"]).lower()
                self.curvaturetype = v if v in _CURV_TYPE_MAP else "mean"
            if "rotate" in cmd:
                self.rotate_flag = bool(cmd["rotate"])
            self._rebuild_colored_mesh()
            return

        if typ == "remove":
            # suppression robuste du mesh + reset de l'état
            if self.mode == "filament":
                try:
                    self.scene.remove_geometry(self.mesh_name)
                except Exception:
                    pass
            else:
                try:
                    if self.mesh is not None:
                        self.vis.remove_geometry(self.mesh, reset_bounding_box=False)
                except Exception:
                    pass

            self.mesh           = None
            self._mesh_in_scene = False
            self.src_path       = None
            self.colored_path   = None

            # reset caméra / cible
            self.target[:] = 0.0
            self.theta = np.deg2rad(45)
            self.phi   = np.deg2rad(25)
            self.dist  = 3.0
            self.zoom_locked = True

            if self.mode == "legacy":
                self.legacy_target = np.array([0.0, 0.0, 0.0], dtype=np.float64)
                try:
                    self.vc.set_lookat([0, 0, 0])
                    self.vc.set_front([-0.577, -0.577, -0.577])
                    self.vc.set_up([0, 1, 0])
                    self.vc.set_zoom(0.8)
                except Exception:
                    pass

            return

        if typ in ("reset", "fit"):
            if self.mesh is not None:
                bbox = self.mesh.get_axis_aligned_bounding_box()
                c = np.asarray(bbox.get_center(), dtype=np.float32)
                extent = np.asarray(bbox.get_extent(), dtype=np.float32)
                diag = float(np.linalg.norm(extent)) or 1.0
                self.target = c
                self.theta = np.deg2rad(45)
                self.phi = np.deg2rad(25)
                self.dist = (0.5 * diag) / np.tan(np.deg2rad(self.fov_v_deg/2.0)) * 1.35
                if self.mode == "legacy":
                    center = np.asarray(c, dtype=np.float64)
                    self.legacy_target = center
                    self.vc.set_lookat(center.tolist())
                    self.vc.set_front((-np.array([1, 1, 1]) / np.sqrt(3)).tolist())
                    self.vc.set_up([0, 1, 0])
                    self.vc.set_zoom(0.8)
            else:
                self.target[:] = 0
                self.theta = np.deg2rad(45)
                self.phi = np.deg2rad(25)
                self.dist = 3.0
                if self.mode == "legacy":
                    self.legacy_target = np.array([0.0, 0.0, 0.0], dtype=np.float64)
                    self.vc.set_lookat([0, 0, 0])
                    self.vc.set_front([-0.577, -0.577, -0.577])
                    self.vc.set_up([0, 1, 0])
                    self.vc.set_zoom(0.8)
            return

        if typ == "settings":
            if "invert_zoom" in cmd:
                self.invert_zoom = bool(cmd.get("invert_zoom"))
            if "obj_color" in cmd:
                self.obj_color_rgba = np.array(hex_to_rgba(cmd["obj_color"]), dtype=np.float32)
                self._refresh_material()
            if "bg" in cmd:
                self.bg = np.array(hex_to_rgb(cmd["bg"]), dtype=np.float32)
                if self.mode == "filament":
                    self.scene.set_background(self.bg)
                else:
                    opt = self.vis.get_render_option()
                    opt.background_color = self.bg.astype(float)
            return

        if typ == "resize":
            w = int(cmd.get("width", self.width))
            h = int(cmd.get("height", self.height))
            w = max(320, min(w, 640))
            h = max(240, min(h, 480))
            if self.mode == "filament":
                if self.renderer is not None:
                    self.renderer.resize(w, h)
            else:
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
            self.width, self.height = w, h
            return

        # ---------- interactions ----------
        if typ in ("rotate", "pan", "zoom", "zoom_dir", "zoom_sem"):
            if self.mode == "filament":
                if typ == "rotate":
                    dx = float(cmd.get("dx", 0))
                    dy = float(cmd.get("dy", 0))
                    s = 0.005
                    self.theta -= dx * s
                    self.phi   -= dy * s
                    self.phi = float(np.clip(self.phi, np.deg2rad(-85), np.deg2rad(85)))
                    if abs(dx) + abs(dy) > 1e-9:
                        self.zoom_locked = False
                elif typ == "pan":
                    dx = float(cmd.get("dx", 0))
                    dy = float(cmd.get("dy", 0))
                    cp = np.cos(self.phi)
                    sp = np.sin(self.phi)
                    ct = np.cos(self.theta)
                    st = np.sin(self.theta)
                    forward = -np.array([cp * st, sp, cp * ct], dtype=np.float32)
                    right = np.cross(forward, np.array([0, 1, 0], dtype=np.float32))
                    right /= (np.linalg.norm(right) + 1e-8)
                    up_cam = np.cross(right, forward)
                    up_cam /= (np.linalg.norm(up_cam) + 1e-8)
                    k = 0.0015 * self.dist
                    self.target += (-dx * k) * right + (dy * k) * up_cam
                elif typ == "zoom":
                    if self.zoom_locked:
                        return
                    scale = float(cmd.get("scale", 1.0))
                    if self.invert_zoom:
                        scale = 1.0 / max(1e-6, scale)
                    self.dist = float(np.clip(self.dist * scale, 0.2, 1000.0))
                elif typ == "zoom_dir":
                    if self.zoom_locked:
                        return
                    dirv = int(cmd.get("dir", 1))
                    if self.invert_zoom:
                        dirv = -dirv
                    step = float(cmd.get("step", 1.12))
                    step = min(max(step, 1.02), 1.5)
                    factor = step if dirv > 0 else (1.0 / step)
                    self.dist = float(np.clip(self.dist * factor, 0.2, 1000.0))
                else:  # zoom_sem
                    if self.zoom_locked:
                        return
                    sem = str(cmd.get("sem", "in")).lower()
                    step = float(cmd.get("step", 1.12))
                    step = min(max(step, 1.02), 1.5)
                    factor = (1.0 / step) if sem == "in" else step
                    self.dist = float(np.clip(self.dist * factor, 0.2, 1000.0))
                return
            else:
                if typ == "rotate":
                    dx = float(cmd.get("dx", 0))
                    dy = float(cmd.get("dy", 0))
                    self.vc.rotate(dx, dy)
                    if abs(dx) + abs(dy) > 1e-9:
                        self.zoom_locked = False
                elif typ == "pan":
                    self.vc.translate(float(cmd.get("dx", 0)), float(cmd.get("dy", 0)))
                elif typ == "zoom":
                    if self.zoom_locked:
                        return
                    s = float(cmd.get("scale", 1.0))
                    if self.invert_zoom:
                        s = 1.0 / max(1e-6, s)
                    self._legacy_zoom_by_factor(s)
                elif typ == "zoom_dir":
                    if self.zoom_locked:
                        return
                    dirv = int(cmd.get("dir", 1))
                    if self.invert_zoom:
                        dirv = -dirv
                    step = float(cmd.get("step", 1.12))
                    step = min(max(step, 1.02), 1.5)
                    factor = step if dirv > 0 else (1.0 / step)
                    self._legacy_zoom_by_factor(factor)
                else:  # zoom_sem
                    if self.zoom_locked:
                        return
                    sem = str(cmd.get("sem", "in")).lower()
                    step = float(cmd.get("step", 1.12))
                    step = min(max(step, 1.02), 1.5)
                    factor = (1.0 / step) if sem == "in" else step
                    self._legacy_zoom_by_factor(factor)
                return

    # ------------------------------
    def render_once(self):
        self.ensure_renderer()

        # commandes
        while True:
            try:
                cmd = self.cmd_q.get_nowait()
            except queue.Empty:
                break
            self._apply_cmd(cmd)

        if self.mesh is not None and not self._mesh_in_scene:
            self._bind_mesh()

        if self.mode == "filament":
            self._look_at_filament()

        q = self.burst_quality if time.monotonic() < self.burst_until else self.idle_quality
        self._encode_frame(q)

        # --- Nouvelle logique de timing ---
        fps = self.burst_fps if time.monotonic() < self.burst_until else self.idle_fps
        fps = max(1.0, float(fps))
        frame_dt = 1.0 / fps

        # On dort par petits bouts pour pouvoir réagir vite à une nouvelle commande
        slept = 0.0
        # plus le fps est élevé, plus on fait des petits pas
        step = 0.002 if fps >= 40 else 0.004   # 2–4 ms

        while slept < frame_dt and self.running:
            time.sleep(step)
            slept += step
            # si une commande arrive, on sort tout de suite -> prochaine frame directe
            if not self.cmd_q.empty():
                break



STATE = MeshLabEngineState()


def render_loop():
    while STATE.running:
        STATE.render_once()
