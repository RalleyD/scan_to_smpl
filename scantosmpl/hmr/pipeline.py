"""HMR pipeline: orchestrates CameraHMR inference over all views + debug output."""

import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

from scantosmpl.config import HMRConfig
from scantosmpl.hmr.camera_hmr import CameraHMRInference, HMROutput
from scantosmpl.hmr.orientation import check_orientation_quality
from scantosmpl.types import CameraParams, ViewResult, ViewType


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

        to_process = [
            v for v in views
            if v.view_type != ViewType.SKIP and v.bbox is not None
        ]

        if not to_process:
            print("[HMR] No processable views found.")
            return views

        inference = self._get_inference()
        hmr_outputs: dict[str, HMROutput] = {}

        t_start = time.time()

        for view in tqdm(to_process, desc="CameraHMR", unit="view"):
            image_path = image_dir / view.image_path.name
            image = Image.open(image_path)

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
        Project SMPL mesh onto the image and draw edges with PIL.

        Only edges with both endpoints in front of the camera (Z > 0) are drawn.
        Edges are subsampled if there are more than 4000 (keeps overlay readable).
        """
        import trimesh

        # Project vertices into image space: pts_2d = K @ v / v.z
        v = vertices_3d.copy()  # (6890, 3), already in camera space
        pts_cam = v  # camera frame
        mask_front = pts_cam[:, 2] > 0.01  # front-facing vertices

        # Full projection
        pts_h = (K @ pts_cam.T).T  # (6890, 3)
        pts_2d = pts_h[:, :2] / (pts_h[:, 2:3] + 1e-9)  # (6890, 2)

        # Extract unique edges
        mesh = trimesh.Trimesh(vertices=v, faces=faces, process=False)
        edges = mesh.edges_unique  # (E, 2)

        # Keep only edges where both vertices are in front of the camera
        both_front = mask_front[edges[:, 0]] & mask_front[edges[:, 1]]
        edges = edges[both_front]

        # Subsample if dense
        max_edges = 4000
        if len(edges) > max_edges:
            idx = np.linspace(0, len(edges) - 1, max_edges, dtype=int)
            edges = edges[idx]

        # Draw on a copy of the source image
        overlay = image.convert("RGB").copy()
        draw = ImageDraw.Draw(overlay)
        W, H = overlay.size

        for e0, e1 in edges:
            x0, y0 = float(pts_2d[e0, 0]), float(pts_2d[e0, 1])
            x1, y1 = float(pts_2d[e1, 0]), float(pts_2d[e1, 1])
            # Clip to image bounds before drawing
            if (
                0 <= x0 <= W and 0 <= y0 <= H
                and 0 <= x1 <= W and 0 <= y1 <= H
            ):
                draw.line([(x0, y0), (x1, y1)], fill=(0, 255, 100), width=1)

        overlay.save(output_path, quality=90)

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
                img = Image.open(img_path) if img_path.exists() else None
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
