"""HMR pipeline: orchestrates CameraHMR inference over all views + debug output."""

import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from tqdm import tqdm

from scantosmpl.detection.image_loader import load_image

from scantosmpl.config import HMRConfig
from scantosmpl.hmr.camera_hmr import CameraHMRInference, HMROutput
from scantosmpl.hmr.orientation import check_orientation_quality
from scantosmpl.types import CameraParams, ViewResult, ViewType

# Views manually excluded from CameraHMR: filename stem → reason.
# These are still valid for Phase 1 detection and Tier 2 PnP, but CameraHMR
# produces unreliable SMPL estimates on rear/oblique views.
# ViTPose hallucinates face keypoints on rear views, so automated detection
# is unreliable — manual exclusion is the pragmatic choice.
DEFAULT_HMR_EXCLUSIONS: dict[str, str] = {
    "cam10_4": "rear view",
    "cam10_5": "rear view",
}


class HMRPipeline:
    """
    Runs CameraHMR on every non-SKIP view and populates ViewResult HMR fields.

    Usage::

        from scantosmpl.hmr.pipeline import HMRPipeline
        from scantosmpl.config import HMRConfig

        pipeline = HMRPipeline(HMRConfig(), device="cuda")
        views = pipeline.process_views(views, Path("data/t-pose/jpg"),
                                       debug_dir=Path("output/debug/hmr"))
    """

    def __init__(self, config: HMRConfig, device: str | None = None) -> None:
        self.config = config
        self.device = device or config.device
        self._inference: CameraHMRInference | None = None

    def _get_inference(self) -> CameraHMRInference:
        if self._inference is None:
            self._inference = CameraHMRInference(self.config, device=self.device)
        return self._inference

    # ------------------------------------------------------------------
    # HMR suitability filter
    # ------------------------------------------------------------------

    @staticmethod
    def _assess_hmr_suitability(
        view: ViewResult,
        exclusions: dict[str, str] | None = None,
        min_conf: float = 0.3,
        min_spread_ratio: float = 0.12,
        min_torso_ratio: float = 0.23,
    ) -> bool:
        """
        Decide whether a view is suitable for CameraHMR.

        CameraHMR is trained on frontal/near-frontal images.  Three exclusion
        criteria are checked:

        0. **Manual exclusion list** — rear views and other problematic angles
           identified by visual inspection.  ViTPose hallucinates face keypoints
           on rear views so automated nose/eye-based detection is unreliable.

        1. **Pure side view** — shoulder horizontal spread < 12 % of bbox width.
           Both shoulders are nearly stacked; the body is seen edge-on.
           (Catches cam02_4 @ 0.07, cam06_4 @ 0.02)

        2. **Floor-up / extreme elevation angle** — torso fraction
           (hip_y − shoulder_y) / bbox_height < 23 %.
           The torso appears compressed because the camera looks up at the subject.
           (Catches cam07_6 @ 0.22; all normal views ≥ 0.23)

        Returns True if the view passes all checks, False otherwise.
        Defaults to True when keypoints are unavailable (benefit of the doubt).
        """
        # Check 0: manual exclusion list
        excl = exclusions if exclusions is not None else DEFAULT_HMR_EXCLUSIONS
        if view.image_path.stem in excl:
            return False

        kps = view.keypoints_2d
        confs = view.keypoint_confs
        bbox = view.bbox

        if kps is None or confs is None or bbox is None:
            return True  # no evidence to exclude

        # COCO-17 indices
        L_SHOULDER, R_SHOULDER = 5, 6
        L_HIP, R_HIP = 11, 12

        l_sh_ok = confs[L_SHOULDER] >= min_conf
        r_sh_ok = confs[R_SHOULDER] >= min_conf
        l_hip_ok = confs[L_HIP] >= min_conf
        r_hip_ok = confs[R_HIP] >= min_conf

        # --- Check 1: shoulder spread (side-view filter) ---
        if l_sh_ok and r_sh_ok:
            bbox_w = float(bbox[2] - bbox[0])
            spread = abs(float(kps[L_SHOULDER, 0]) - float(kps[R_SHOULDER, 0]))
            if bbox_w > 0 and (spread / bbox_w) < min_spread_ratio:
                return False

        # --- Check 2: torso fraction (floor-up / elevation filter) ---
        if l_sh_ok and r_sh_ok and (l_hip_ok or r_hip_ok):
            bbox_h = float(bbox[3] - bbox[1])
            sh_y = (float(kps[L_SHOULDER, 1]) + float(kps[R_SHOULDER, 1])) / 2.0
            hip_ys = []
            if l_hip_ok:
                hip_ys.append(float(kps[L_HIP, 1]))
            if r_hip_ok:
                hip_ys.append(float(kps[R_HIP, 1]))
            hip_y = float(np.mean(hip_ys))
            if bbox_h > 0 and (hip_y - sh_y) / bbox_h < min_torso_ratio:
                return False

        return True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_views(
        self,
        views: list[ViewResult],
        image_dir: Path,
        debug_dir: Path | None = None,
    ) -> list[ViewResult]:
        """
        Run HMR on all FULL_BODY and PARTIAL views. Populates HMR fields in-place.

        Args:
            views: ViewResult list from the detection pipeline.
            image_dir: Directory containing the source images.
            debug_dir: If set, write debug JSON, wireframe overlays, and summary.

        Returns:
            Same list with HMR fields populated.
        """
        if debug_dir is not None:
            debug_dir.mkdir(parents=True, exist_ok=True)

        # Stamp suitability flag on every non-SKIP view before filtering
        for v in views:
            if v.view_type != ViewType.SKIP and v.bbox is not None:
                v.hmr_suitable = self._assess_hmr_suitability(v)

        to_process = [
            v for v in views
            if v.view_type != ViewType.SKIP and v.bbox is not None and v.hmr_suitable
        ]

        unsuitable = [v for v in views if v.view_type != ViewType.SKIP and not v.hmr_suitable]
        if unsuitable:
            names = ", ".join(v.image_path.name for v in unsuitable)
            print(f"[HMR] Skipping {len(unsuitable)} unsuitable view(s): {names}")

        if not to_process:
            print("[HMR] No processable views found.")
            return views

        inference = self._get_inference()
        hmr_outputs: dict[str, HMROutput] = {}

        t_start = time.time()

        for view in tqdm(to_process, desc="CameraHMR", unit="view"):
            image_path = image_dir / view.image_path.name
            loaded = load_image(image_path)
            image = loaded.image

            focal_length_px = (
                view.camera.focal_length if view.camera is not None else _default_focal(image)
            )

            try:
                output = inference.infer(image, view.bbox, focal_length_px)
            except Exception as exc:
                print(f"[HMR] WARNING: {view.image_path.name} failed — {exc}")
                continue

            # Populate ViewResult HMR fields
            view.betas = output.betas
            view.body_pose = output.body_pose
            view.global_orient = output.global_orient
            view.dense_keypoints_2d = output.dense_keypoints_2d
            view.dense_keypoint_confs = output.dense_keypoint_confs

            W, H = image.size
            fov_deg = output.fov_flnet if output.fov_flnet is not None else output.fov_exif
            if view.camera is None:
                view.camera = CameraParams(
                    focal_length=focal_length_px,
                    principal_point=(W / 2.0, H / 2.0),
                    fov=fov_deg,
                    hmr_translation=output.cam_translation,
                )
            else:
                view.camera.fov = fov_deg
                view.camera.hmr_translation = output.cam_translation

            hmr_outputs[view.image_path.name] = output

            if debug_dir is not None and output.vertices is not None:
                overlay_path = debug_dir / (view.image_path.stem + "_hmr_overlay.jpg")
                self._save_wireframe_overlay(
                    image,
                    output.vertices,
                    inference.smpl_faces,
                    np.array([
                        [focal_length_px, 0.0, W / 2.0],
                        [0.0, focal_length_px, H / 2.0],
                        [0.0, 0.0, 1.0],
                    ]),
                    output.cam_translation,
                    overlay_path,
                )

        elapsed = time.time() - t_start
        print(f"[HMR] Processed {len(hmr_outputs)}/{len(to_process)} views in {elapsed:.1f}s")

        if debug_dir is not None and hmr_outputs:
            self._save_debug(views, hmr_outputs, debug_dir, elapsed)

        return views

    # ------------------------------------------------------------------
    # Wireframe overlay
    # ------------------------------------------------------------------

    def _save_wireframe_overlay(
        self,
        image: Image.Image,
        vertices_3d: np.ndarray,
        faces: np.ndarray,
        K: np.ndarray,
        cam_trans: np.ndarray,
        output_path: Path,
    ) -> None:
        """
        Project SMPL mesh onto the image with grey face shading and purple edges.

        Rendering steps:
          1. Filter faces: all 3 vertices must be in front of the camera (Z > 0.01).
          2. Back-face cull: discard faces whose projected 2D winding is clockwise
             (rear-facing in SMPL's CCW-Y-down convention).
          3. Painter's algorithm: sort front faces far-to-near and draw grey filled
             polygons on a transparent RGBA overlay.
          4. Draw purple edges on top (edges_unique, subsampled to ≤ 4000).
          5. Alpha-composite the overlay onto the source image and save.
        """
        import trimesh

        v = vertices_3d.copy()  # (6890, 3), already in camera space
        v_z = v[:, 2]
        W, H = image.size

        # Project all vertices
        pts_h = (K @ v.T).T                             # (6890, 3)
        pts_2d = pts_h[:, :2] / (pts_h[:, 2:3] + 1e-9) # (6890, 2)

        # --- Face shading: grey semi-transparent filled polygons ---
        f = np.asarray(faces)
        all_front = (v_z[f[:, 0]] > 0.01) & (v_z[f[:, 1]] > 0.01) & (v_z[f[:, 2]] > 0.01)
        vis_faces = f[all_front]

        # Back-face culling via 2D cross product.
        # SMPL uses CCW winding in 3D; with Y flipped to image space the sign reverses,
        # so front-facing projected triangles have cross > 0.
        v0 = pts_2d[vis_faces[:, 0]]
        v1 = pts_2d[vis_faces[:, 1]]
        v2 = pts_2d[vis_faces[:, 2]]
        cross = (v1[:, 0] - v0[:, 0]) * (v2[:, 1] - v0[:, 1]) \
              - (v1[:, 1] - v0[:, 1]) * (v2[:, 0] - v0[:, 0])
        vis_faces = vis_faces[cross > 0]

        # Painter's algorithm: draw far faces first so near faces paint over them
        z_cent = (v_z[vis_faces[:, 0]] + v_z[vis_faces[:, 1]] + v_z[vis_faces[:, 2]]) / 3.0
        vis_faces = vis_faces[np.argsort(z_cent)[::-1]]

        # Draw on a transparent RGBA layer, then composite over the photo
        base = image.convert("RGBA")
        layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)

        GREY = (160, 160, 160, 110)   # semi-transparent grey fill
        margin = 200                   # allow slightly off-screen polys
        for face in vis_faces:
            p0 = (float(pts_2d[face[0], 0]), float(pts_2d[face[0], 1]))
            p1 = (float(pts_2d[face[1], 0]), float(pts_2d[face[1], 1]))
            p2 = (float(pts_2d[face[2], 0]), float(pts_2d[face[2], 1]))
            if all(-margin <= p[0] <= W + margin and -margin <= p[1] <= H + margin
                   for p in (p0, p1, p2)):
                draw.polygon([p0, p1, p2], fill=GREY)

        # --- Edge drawing: purple ---
        mesh = trimesh.Trimesh(vertices=v, faces=faces, process=False)
        edges = mesh.edges_unique                              # (E, 2)
        both_front = (v_z[edges[:, 0]] > 0.01) & (v_z[edges[:, 1]] > 0.01)
        edges = edges[both_front]

        max_edges = 4000
        if len(edges) > max_edges:
            edges = edges[np.linspace(0, len(edges) - 1, max_edges, dtype=int)]

        PURPLE = (148, 103, 189, 230)  # muted purple, nearly opaque
        for e0, e1 in edges:
            x0, y0 = float(pts_2d[e0, 0]), float(pts_2d[e0, 1])
            x1, y1 = float(pts_2d[e1, 0]), float(pts_2d[e1, 1])
            if 0 <= x0 <= W and 0 <= y0 <= H and 0 <= x1 <= W and 0 <= y1 <= H:
                draw.line([(x0, y0), (x1, y1)], fill=PURPLE, width=1)

        result = Image.alpha_composite(base, layer)
        result.convert("RGB").save(output_path, quality=90)

    # ------------------------------------------------------------------
    # Debug output
    # ------------------------------------------------------------------

    def _save_debug(
        self,
        views: list[ViewResult],
        hmr_outputs: dict[str, HMROutput],
        debug_dir: Path,
        elapsed: float,
    ) -> None:
        """Write JSON results, per-view stats, and a human-readable summary."""

        # JSON: per-image HMR results
        json_data: dict[str, dict] = {}
        for name, out in hmr_outputs.items():
            json_data[name] = {
                "betas": out.betas.tolist(),
                "body_pose_norm": float(np.linalg.norm(out.body_pose)),
                "global_orient": out.global_orient.tolist(),
                "cam_translation": out.cam_translation.tolist(),
                "fov_exif_deg": out.fov_exif,
                "fov_flnet_deg": out.fov_flnet,
                "dense_kp_count": int(out.dense_keypoints_2d.shape[0]),
            }

        json_path = debug_dir / "hmr_results.json"
        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=2)

        # Summary text
        lines: list[str] = [
            "=" * 72,
            "Phase 2 HMR Summary",
            "=" * 72,
            "",
            f"Total views processed : {len(hmr_outputs)}",
            f"Views skipped (unsuitable): "
            + (", ".join(v.image_path.name for v in views if not v.hmr_suitable) or "none"),
            f"Elapsed               : {elapsed:.1f}s",
            f"  ({elapsed / max(len(hmr_outputs), 1):.1f}s per image)",
            "",
        ]

        # FoV cross-check table
        lines += [
            "FoV Cross-Check (EXIF vs FLNet):",
            f"  {'Image':<25}  {'FoV EXIF':>10}  {'FoV FLNet':>10}  {'Diff':>8}  {'Pass':>6}",
            "  " + "-" * 66,
        ]
        fov_diffs: list[float] = []
        for name, out in sorted(hmr_outputs.items()):
            exif = out.fov_exif
            flnet = out.fov_flnet
            if flnet is not None:
                diff = abs(exif - flnet)
                fov_diffs.append(diff)
                ok = "OK" if diff < 10.0 else "FAIL"
                lines.append(
                    f"  {name:<25}  {exif:>10.2f}°  {flnet:>10.2f}°  {diff:>7.2f}°  {ok:>6}"
                )
            else:
                lines.append(
                    f"  {name:<25}  {exif:>10.2f}°  {'N/A':>10}  {'N/A':>8}  {'N/A':>6}"
                )

        if fov_diffs:
            lines += [
                "",
                f"  Mean FoV diff: {np.mean(fov_diffs):.2f}° | "
                f"Max: {np.max(fov_diffs):.2f}° | "
                f"Pass (<10°): {sum(d < 10.0 for d in fov_diffs)}/{len(fov_diffs)}",
            ]

        # Shape parameter statistics
        betas_all = np.stack([hmr_outputs[n].betas for n in hmr_outputs], axis=0)
        lines += [
            "",
            "Shape Parameter (β) Statistics:",
            f"  Mean   : {np.mean(betas_all, axis=0).round(3).tolist()}",
            f"  Std    : {np.std(betas_all, axis=0).round(3).tolist()}",
            f"  Max std: {float(np.std(betas_all, axis=0).max()):.3f}  "
            f"(criterion 2.5: < 1.0)",
        ]

        # Pose variance (body_pose norm)
        pose_norms = [float(np.linalg.norm(hmr_outputs[n].body_pose)) for n in hmr_outputs]
        lines += [
            "",
            "Body Pose Norms (θ):",
            f"  Mean: {np.mean(pose_norms):.3f} rad | "
            f"Std: {np.std(pose_norms):.3f} | "
            f"Min: {min(pose_norms):.3f} | Max: {max(pose_norms):.3f}",
        ]

        # Orientation quality for views with ViTPose keypoints
        lines += ["", "Orientation Quality:"]
        view_map = {v.image_path.name: v for v in views}
        for name, out in sorted(hmr_outputs.items()):
            view = view_map.get(name)
            if view is not None and view.keypoints_2d is not None:
                img_path = view.image_path
                img = load_image(img_path).image if img_path.exists() else None
                hw = (img.height, img.width) if img is not None else (1000, 1000)
                confs = view.keypoint_confs if view.keypoint_confs is not None else np.zeros(17)
                quality = check_orientation_quality(
                    out.global_orient, view.keypoints_2d, confs, hw
                )
                status = "OK" if quality.score >= 0.67 else "WARN"
                warn_str = "; ".join(quality.warnings) if quality.warnings else "none"
                lines.append(f"  {name:<25}  score={quality.score:.2f}  [{status}]  {warn_str}")

        lines += ["", "=" * 72]

        summary_path = debug_dir / "summary.txt"
        with open(summary_path, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"[HMR] Debug output written to {debug_dir}/")
        print(f"      summary.txt — FoV table, β stats, orientation flags")
        print(f"      hmr_results.json — per-image SMPL parameters")


# ------------------------------------------------------------------
# Utility
# ------------------------------------------------------------------

def _default_focal(image: Image.Image) -> float:
    """Fallback focal length: assume 50mm equivalent on 35mm sensor."""
    W, _ = image.size
    return float(W) * (50.0 / 36.0)
