"""Multi-view consensus: fuse per-view HMR estimates into a single SMPL parameter set."""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from scantosmpl.types import ViewResult, ViewType
from scantosmpl.utils.geometry import (
    aa_to_rotmat,
    compute_pa_mpjpe,
    frechet_mean_so3,
    rotmat_to_aa,
)


@dataclass
class ConsensusResult:
    """Output from multi-view SMPL parameter consensus."""

    betas: np.ndarray               # (10,) consensus shape
    body_pose: np.ndarray           # (69,) consensus pose (axis-angle)
    global_orient: np.ndarray       # (3,) canonical = [0, 0, 0]

    # Mesh from consensus params
    vertices: np.ndarray            # (6890, 3) in canonical frame
    joints: np.ndarray              # (24, 3) SMPL joints in canonical frame
    faces: np.ndarray               # (13776, 3)

    # Quality metrics
    pa_mpjpe_per_view: dict[str, float]  # view_name -> PA-MPJPE in mm
    pa_mpjpe_mean: float                  # mean across views (criterion 3.6: < 50mm)
    beta_std: np.ndarray                  # (10,) per-component std before aggregation
    body_height_m: float                  # estimated height from consensus mesh
    per_view_weights: dict[str, float]    # view_name -> confidence weight used
    n_views_used: int                     # views after trimming


class ConsensusBuilder:
    """Fuses per-view HMR estimates into a single SMPL parameter set.

    Usage::

        builder = ConsensusBuilder("models/smpl/SMPL_NEUTRAL.pkl")
        result = builder.build_consensus(views)
        # result.betas, result.body_pose, result.vertices, etc.
    """

    def __init__(
        self,
        smpl_model_path: str | Path,
        gender: str = "neutral",
        device: str = "cuda",
        trim_fraction: float = 0.1,
    ) -> None:
        self.device = device
        self.trim_fraction = trim_fraction
        self._smpl = None
        self._smpl_model_path = str(smpl_model_path)
        self._gender = gender

    def _get_smpl(self):
        if self._smpl is None:
            import smplx
            import torch

            self._smpl = smplx.create(
                self._smpl_model_path,
                model_type="smpl",
                gender=self._gender,
                use_face_contour=False,
            ).to(self.device).eval()
        return self._smpl

    def build_consensus(
        self,
        views: list[ViewResult],
        debug_dir: Path | None = None,
        image_dir: Path | None = None,
    ) -> ConsensusResult:
        """
        Build consensus from HMR-processed views.

        Only uses views where betas is not None and hmr_suitable is True.

        Args:
            views: ViewResult list from the HMR pipeline.
            debug_dir: If set, write debug JSON, summary, mesh .obj, and overlays.
            image_dir: Directory containing source images (needed for overlays).

        Returns:
            ConsensusResult with consensus parameters and quality metrics.
        """
        import torch

        # Filter to views with HMR output
        valid = [
            v for v in views
            if v.betas is not None and v.hmr_suitable and v.view_type != ViewType.SKIP
        ]
        if len(valid) < 2:
            raise ValueError(
                f"Need at least 2 valid HMR views for consensus, got {len(valid)}"
            )

        view_names = [v.image_path.name for v in valid]
        betas_list = [v.betas for v in valid]
        body_pose_list = [v.body_pose for v in valid]

        # Step 1: per-view confidence weights
        weights = self._compute_view_weights(valid)
        per_view_weights = {name: float(w) for name, w in zip(view_names, weights)}

        # Step 2: aggregate betas
        beta_std = np.std(np.stack(betas_list, axis=0), axis=0)
        consensus_betas = self._aggregate_betas(betas_list, weights)

        # Step 3: aggregate body_pose (SO(3) Frechet mean per joint)
        consensus_body_pose = self._aggregate_body_pose(body_pose_list, weights)

        # Step 4: canonical global orient
        consensus_go = np.zeros(3, dtype=np.float64)

        # Step 5: SMPL forward pass -> mesh + joints
        smpl = self._get_smpl()
        with torch.no_grad():
            smpl_out = smpl(
                global_orient=torch.from_numpy(consensus_go).float().unsqueeze(0).to(self.device),
                body_pose=torch.from_numpy(consensus_body_pose).float().unsqueeze(0).to(self.device),
                betas=torch.from_numpy(consensus_betas).float().unsqueeze(0).to(self.device),
            )
        vertices = smpl_out.vertices[0].cpu().numpy()  # (6890, 3)
        joints = smpl_out.joints[0, :24].cpu().numpy()  # (24, 3)
        faces = np.asarray(smpl.faces, dtype=np.int64)

        # Step 5b: body height from mesh
        body_height_m = self._compute_body_height(vertices)

        # Step 6: cross-view PA-MPJPE
        per_view_joints = self._compute_per_view_joints(valid)
        pa_mpjpe_per_view = self._compute_pa_mpjpe_all(
            joints, per_view_joints, view_names,
        )
        pa_mpjpe_mean = float(np.mean(list(pa_mpjpe_per_view.values())))

        result = ConsensusResult(
            betas=consensus_betas.astype(np.float32),
            body_pose=consensus_body_pose.astype(np.float32),
            global_orient=consensus_go.astype(np.float32),
            vertices=vertices.astype(np.float32),
            joints=joints.astype(np.float32),
            faces=faces,
            pa_mpjpe_per_view=pa_mpjpe_per_view,
            pa_mpjpe_mean=pa_mpjpe_mean,
            beta_std=beta_std.astype(np.float32),
            body_height_m=body_height_m,
            per_view_weights=per_view_weights,
            n_views_used=len(valid),
        )

        if debug_dir is not None:
            debug_dir.mkdir(parents=True, exist_ok=True)
            self._save_debug(result, valid, debug_dir, image_dir)

        return result

    # ------------------------------------------------------------------
    # Aggregation methods
    # ------------------------------------------------------------------

    def _compute_view_weights(self, views: list[ViewResult]) -> np.ndarray:
        """
        Confidence weight per view based on dense keypoint mean confidence.

        Returns (N,) weights normalised to sum to 1.
        """
        weights = np.ones(len(views), dtype=np.float64)
        for i, v in enumerate(views):
            if v.dense_keypoint_confs is not None:
                weights[i] = float(np.mean(v.dense_keypoint_confs))
        # Normalise
        w_sum = weights.sum()
        if w_sum > 1e-12:
            weights /= w_sum
        return weights

    def _aggregate_betas(
        self,
        betas_list: list[np.ndarray],
        weights: np.ndarray,
    ) -> np.ndarray:
        """
        Confidence-weighted trimmed mean per beta component.

        Trims `trim_fraction` from each tail before computing the weighted mean.
        Returns (10,).
        """
        betas_arr = np.stack(betas_list, axis=0)  # (N, 10)
        N = betas_arr.shape[0]
        n_trim = max(1, int(round(N * self.trim_fraction)))

        consensus = np.zeros(10, dtype=np.float64)
        for c in range(10):
            vals = betas_arr[:, c]
            w = weights.copy()

            # Sort by value, trim extremes
            order = np.argsort(vals)
            if N > 2 * n_trim + 1:
                keep = order[n_trim:-n_trim]
            else:
                keep = order  # too few views to trim

            # Weighted mean of kept views
            w_kept = w[keep]
            w_sum = w_kept.sum()
            if w_sum > 1e-12:
                consensus[c] = np.sum(w_kept * vals[keep]) / w_sum
            else:
                consensus[c] = np.median(vals[keep])

        return consensus

    def _aggregate_body_pose(
        self,
        body_pose_list: list[np.ndarray],
        weights: np.ndarray,
    ) -> np.ndarray:
        """
        SO(3) Frechet mean per joint (23 joints).

        Converts each joint's axis-angle to rotation matrix, computes the
        weighted Frechet mean on SO(3), and converts back to axis-angle.
        Returns (69,) axis-angle.
        """
        N = len(body_pose_list)
        poses = np.stack(body_pose_list, axis=0)  # (N, 69)
        consensus_pose = np.zeros(69, dtype=np.float64)

        for j in range(23):
            # Extract axis-angle for joint j across all views
            joint_aa = poses[:, j * 3: (j + 1) * 3]  # (N, 3)

            # Convert to rotation matrices
            joint_rotmats = aa_to_rotmat(joint_aa)  # (N, 3, 3)

            # Frechet mean
            mean_rot = frechet_mean_so3(joint_rotmats, weights=weights)

            # Convert back to axis-angle
            mean_aa = rotmat_to_aa(mean_rot.reshape(1, 3, 3))[0]  # (3,)
            consensus_pose[j * 3: (j + 1) * 3] = mean_aa

        return consensus_pose

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def _compute_body_height(self, vertices: np.ndarray) -> float:
        """Estimate body height in metres from vertex Y-extent."""
        # SMPL canonical frame: Y is up. Height = max(Y) - min(Y).
        return float(vertices[:, 1].max() - vertices[:, 1].min())

    def _compute_per_view_joints(
        self, views: list[ViewResult],
    ) -> list[np.ndarray]:
        """Run SMPL forward pass per view with canonical global_orient to get joints."""
        import torch

        smpl = self._get_smpl()
        per_view_joints = []

        canonical_go = torch.zeros(1, 3, dtype=torch.float32, device=self.device)

        for v in views:
            with torch.no_grad():
                out = smpl(
                    global_orient=canonical_go,
                    body_pose=torch.from_numpy(v.body_pose).float().unsqueeze(0).to(self.device),
                    betas=torch.from_numpy(v.betas).float().unsqueeze(0).to(self.device),
                )
            per_view_joints.append(out.joints[0, :24].cpu().numpy())

        return per_view_joints

    def _compute_pa_mpjpe_all(
        self,
        consensus_joints: np.ndarray,
        per_view_joints: list[np.ndarray],
        view_names: list[str],
    ) -> dict[str, float]:
        """PA-MPJPE for each view vs consensus. Returns {name: mm}."""
        result = {}
        for name, pv_joints in zip(view_names, per_view_joints):
            # PA-MPJPE in metres -> convert to mm
            pa = compute_pa_mpjpe(consensus_joints, pv_joints)
            result[name] = pa * 1000.0  # m -> mm
        return result

    # ------------------------------------------------------------------
    # Consensus overlay on source images
    # ------------------------------------------------------------------

    def _render_consensus_overlay(
        self,
        image: Image.Image,
        vertices_canonical: np.ndarray,
        faces: np.ndarray,
        global_orient: np.ndarray,
        cam_translation: np.ndarray,
        focal_length_px: float,
        consensus_body_pose: np.ndarray,
        consensus_betas: np.ndarray,
        output_path: Path,
    ) -> None:
        """
        Render the consensus mesh projected onto a source image.

        Uses the per-view global_orient and cam_translation from Phase 2
        combined with consensus betas + body_pose to produce camera-space
        vertices, then projects and renders with the same style as Phase 2.
        """
        import torch
        import trimesh

        smpl = self._get_smpl()

        # Run SMPL with consensus betas/body_pose but per-view global_orient
        with torch.no_grad():
            out = smpl(
                global_orient=torch.from_numpy(global_orient).float().unsqueeze(0).to(self.device),
                body_pose=torch.from_numpy(consensus_body_pose).float().unsqueeze(0).to(self.device),
                betas=torch.from_numpy(consensus_betas).float().unsqueeze(0).to(self.device),
            )
        verts = out.vertices[0].cpu().numpy()  # (6890, 3)

        # Translate to camera space
        v = verts + cam_translation[None, :]
        v_z = v[:, 2]
        W, H = image.size

        K = np.array([
            [focal_length_px, 0.0, W / 2.0],
            [0.0, focal_length_px, H / 2.0],
            [0.0, 0.0, 1.0],
        ])

        # Project
        pts_h = (K @ v.T).T
        pts_2d = pts_h[:, :2] / (pts_h[:, 2:3] + 1e-9)

        # --- Face shading ---
        f = np.asarray(faces)
        all_front = (v_z[f[:, 0]] > 0.01) & (v_z[f[:, 1]] > 0.01) & (v_z[f[:, 2]] > 0.01)
        vis_faces = f[all_front]

        v0 = pts_2d[vis_faces[:, 0]]
        v1 = pts_2d[vis_faces[:, 1]]
        v2 = pts_2d[vis_faces[:, 2]]
        cross = (v1[:, 0] - v0[:, 0]) * (v2[:, 1] - v0[:, 1]) \
              - (v1[:, 1] - v0[:, 1]) * (v2[:, 0] - v0[:, 0])
        vis_faces = vis_faces[cross > 0]

        z_cent = (v_z[vis_faces[:, 0]] + v_z[vis_faces[:, 1]] + v_z[vis_faces[:, 2]]) / 3.0
        vis_faces = vis_faces[np.argsort(z_cent)[::-1]]

        base = image.convert("RGBA")
        layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)

        # Teal fill to distinguish from Phase 2's grey
        TEAL = (0, 160, 160, 100)
        margin = 200
        for face in vis_faces:
            p0 = (float(pts_2d[face[0], 0]), float(pts_2d[face[0], 1]))
            p1 = (float(pts_2d[face[1], 0]), float(pts_2d[face[1], 1]))
            p2 = (float(pts_2d[face[2], 0]), float(pts_2d[face[2], 1]))
            if all(-margin <= p[0] <= W + margin and -margin <= p[1] <= H + margin
                   for p in (p0, p1, p2)):
                draw.polygon([p0, p1, p2], fill=TEAL)

        # --- Edges: dark teal ---
        mesh = trimesh.Trimesh(vertices=v, faces=faces, process=False)
        edges = mesh.edges_unique
        both_front = (v_z[edges[:, 0]] > 0.01) & (v_z[edges[:, 1]] > 0.01)
        edges = edges[both_front]

        max_edges = 4000
        if len(edges) > max_edges:
            edges = edges[np.linspace(0, len(edges) - 1, max_edges, dtype=int)]

        DARK_TEAL = (0, 120, 120, 230)
        for e0, e1 in edges:
            x0, y0 = float(pts_2d[e0, 0]), float(pts_2d[e0, 1])
            x1, y1 = float(pts_2d[e1, 0]), float(pts_2d[e1, 1])
            if 0 <= x0 <= W and 0 <= y0 <= H and 0 <= x1 <= W and 0 <= y1 <= H:
                draw.line([(x0, y0), (x1, y1)], fill=DARK_TEAL, width=1)

        result = Image.alpha_composite(base, layer)
        result.convert("RGB").save(output_path, quality=90)

    # ------------------------------------------------------------------
    # Debug output
    # ------------------------------------------------------------------

    def _select_frontal_views(
        self, views: list[ViewResult], max_views: int = 3,
    ) -> list[ViewResult]:
        """
        Select the most frontal views based on shoulder spread ratio.

        Higher spread = more frontal. Returns up to max_views sorted by frontality.
        """
        scored = []
        for v in views:
            if v.keypoints_2d is None or v.keypoint_confs is None or v.bbox is None:
                continue
            if v.betas is None:
                continue
            L_SH, R_SH = 5, 6
            if v.keypoint_confs[L_SH] < 0.3 or v.keypoint_confs[R_SH] < 0.3:
                continue
            bbox_w = float(v.bbox[2] - v.bbox[0])
            if bbox_w < 1.0:
                continue
            spread = abs(float(v.keypoints_2d[L_SH, 0]) - float(v.keypoints_2d[R_SH, 0]))
            scored.append((spread / bbox_w, v))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [v for _, v in scored[:max_views]]

    def _save_debug(
        self,
        result: ConsensusResult,
        views: list[ViewResult],
        debug_dir: Path,
        image_dir: Path | None,
    ) -> None:
        """Write JSON, summary, mesh .obj, and consensus overlays."""

        # --- JSON ---
        json_data = {
            "betas": result.betas.tolist(),
            "body_pose": result.body_pose.tolist(),
            "global_orient": result.global_orient.tolist(),
            "body_height_m": result.body_height_m,
            "pa_mpjpe_mean_mm": result.pa_mpjpe_mean,
            "pa_mpjpe_per_view": result.pa_mpjpe_per_view,
            "beta_std": result.beta_std.tolist(),
            "per_view_weights": result.per_view_weights,
            "n_views_used": result.n_views_used,
        }
        with open(debug_dir / "consensus_results.json", "w") as f:
            json.dump(json_data, f, indent=2)

        # --- Mesh .obj ---
        self._save_obj(
            result.vertices, result.faces,
            debug_dir / "consensus_mesh.obj",
        )

        # --- Summary ---
        self._save_summary(result, debug_dir / "summary.txt")

        # --- Consensus overlays on frontal views ---
        if image_dir is not None:
            from scantosmpl.detection.image_loader import load_image

            frontal = self._select_frontal_views(views, max_views=3)
            for v in frontal:
                img_path = image_dir / v.image_path.name
                if not img_path.exists():
                    continue
                loaded = load_image(img_path)
                image = loaded.image

                go = v.global_orient
                cam_t = v.camera.hmr_translation if v.camera and v.camera.hmr_translation is not None else None
                fl = v.camera.focal_length if v.camera else None
                if go is None or cam_t is None or fl is None:
                    continue

                overlay_path = debug_dir / (v.image_path.stem + "_consensus_overlay.jpg")
                self._render_consensus_overlay(
                    image, result.vertices, result.faces,
                    go, cam_t, fl,
                    result.body_pose, result.betas,
                    overlay_path,
                )

            if frontal:
                names = ", ".join(v.image_path.stem for v in frontal)
                print(f"[Consensus] Overlays: {names}")

        print(f"[Consensus] Debug output written to {debug_dir}/")

    @staticmethod
    def _save_obj(vertices: np.ndarray, faces: np.ndarray, path: Path) -> None:
        """Write a simple Wavefront .obj file."""
        with open(path, "w") as f:
            f.write("# ScanToSMPL Tier 1 consensus mesh\n")
            for v in vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            for face in faces:
                # OBJ is 1-indexed
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

    @staticmethod
    def _save_summary(result: ConsensusResult, path: Path) -> None:
        """Write a human-readable summary."""
        lines = [
            "=" * 72,
            "Phase 3 Consensus Summary — Tier 1",
            "=" * 72,
            "",
            f"Views used        : {result.n_views_used}",
            f"Body height       : {result.body_height_m:.3f} m"
            f"  {'OK' if 1.5 <= result.body_height_m <= 2.0 else 'WARNING: outside 1.5-2.0m'}",
            "",
        ]

        # Per-view weights
        lines += ["Per-View Weights:"]
        for name, w in sorted(result.per_view_weights.items()):
            lines.append(f"  {name:<25}  {w:.4f}")

        # Beta stats
        lines += [
            "",
            "Consensus Shape (beta):",
            f"  Values : {[round(float(b), 4) for b in result.betas]}",
            f"  Std    : {[round(float(s), 4) for s in result.beta_std]}",
            f"  Max std: {float(result.beta_std.max()):.4f}",
        ]

        # Body pose stats
        bp = result.body_pose.reshape(23, 3)
        joint_mags = np.linalg.norm(bp, axis=1)
        lines += [
            "",
            "Consensus Body Pose (23 joints):",
            f"  Total pose norm  : {float(np.linalg.norm(result.body_pose)):.4f} rad",
            f"  Max joint rotation: {float(joint_mags.max()):.4f} rad (joint {int(joint_mags.argmax())})",
            f"  Mean joint rotation: {float(joint_mags.mean()):.4f} rad",
        ]

        # T-pose arm check: shoulders (16,17), elbows (18,19), wrists (20,21)
        # In canonical frame, arms roughly horizontal means Y coords are similar
        if result.joints is not None:
            j = result.joints
            for side, sh, el, wr in [("Left", 16, 18, 20), ("Right", 17, 19, 21)]:
                sh_y, el_y, wr_y = float(j[sh, 1]), float(j[el, 1]), float(j[wr, 1])
                arm_drop = abs(el_y - sh_y) + abs(wr_y - sh_y)
                status = "OK" if arm_drop < 0.15 else "WARN"
                lines.append(
                    f"  {side} arm Y-drop: {arm_drop:.3f}m [{status}]"
                    f"  (shoulder={sh_y:.3f}, elbow={el_y:.3f}, wrist={wr_y:.3f})"
                )

        # PA-MPJPE
        lines += [
            "",
            "Cross-View PA-MPJPE (criterion 3.6: < 50mm):",
            f"  {'View':<25}  {'PA-MPJPE':>10}  {'Pass':>6}",
            "  " + "-" * 45,
        ]
        for name, pa in sorted(result.pa_mpjpe_per_view.items()):
            ok = "OK" if pa < 50.0 else "FAIL"
            lines.append(f"  {name:<25}  {pa:>9.2f}mm  {ok:>6}")
        lines += [
            "",
            f"  Mean PA-MPJPE: {result.pa_mpjpe_mean:.2f}mm"
            f"  {'PASS' if result.pa_mpjpe_mean < 50.0 else 'FAIL'}",
        ]

        lines += ["", "=" * 72]

        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")
