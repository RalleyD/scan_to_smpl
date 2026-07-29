"""Tier 1 (consensus) vs Tier 2 (refinement) visual comparison.

Renders each tier's SMPL mesh onto the same source photograph so the two can
be judged by eye, side by side. Reads only the debug artefacts each tier
already writes to disk (Phase 1's detections.json, Phase 2's hmr_results.json,
Tier 1's consensus_mesh.obj/consensus_results.json, Tier 2's
refined_mesh.obj/refinement_results.json) — no HMR/calibration/GPU re-run
needed.

Each tier is rendered through the camera model it was actually fit against,
not a shared approximation:
  - Tier 1 has no shared world frame — each view's body orientation lives in
    that view's own global_orient (from Phase 2's per-image HMR estimate),
    projected with an assumed image-centre principal point (mirrors
    scantosmpl.hmr.consensus._render_consensus_overlay exactly).
  - Tier 2 has one world-frame mesh, viewed through that camera's actual
    self-calibrated [R|t|K] (including Phase 4's corrected principal point).
Using a different, homogenised camera for either tier would introduce a
visual mismatch that has nothing to do with which tier fits the photo
better — so each panel intentionally uses its own tier's real camera model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageOps

from scantosmpl.detection.image_loader import DEFAULT_ORIENTATION_OVERRIDES

TIER1_FILL = (0, 160, 160, 110)
TIER1_EDGE = (0, 120, 120, 230)
TIER2_FILL = (210, 150, 0, 110)
TIER2_EDGE = (170, 110, 0, 230)

DEFAULT_IMAGE_DIR = Path("data/t-pose/jpg")
DEFAULT_DETECTIONS = Path("output/debug/detection/detections.json")
DEFAULT_CONSENSUS_DIR = Path("output/debug/consensus")
DEFAULT_REFINEMENT_DIR = Path("output/debug/refinement")
DEFAULT_HMR_RESULTS = Path("output/debug/hmr/hmr_results.json")
DEFAULT_SMPL_MODEL = Path("models/smpl/SMPL_NEUTRAL.pkl")

# Two views the Tier 1 consensus summary itself flags as its worst fits
# (cross-view PA-MPJPE 68.72mm / 63.00mm, both over the 50mm criterion,
# vs. ~20-30mm for the rest) — the most informative default comparison.
DEFAULT_VIEWS = ["cam02_5.JPG", "cam03_6.JPG"]


def _open_view_image(image_dir: Path, view_name: str) -> Image.Image:
    img_path = image_dir / view_name
    img = ImageOps.exif_transpose(Image.open(img_path))
    stem = img_path.stem
    if stem in DEFAULT_ORIENTATION_OVERRIDES:
        deg = DEFAULT_ORIENTATION_OVERRIDES[stem]
        img = img.rotate(-deg, expand=True)  # PIL rotate is CCW
    return img.convert("RGB")


def _render_mesh_overlay(
    image: Image.Image,
    verts_cam: np.ndarray,
    faces: np.ndarray,
    K: np.ndarray,
    fill_rgba: tuple[int, int, int, int],
    edge_rgba: tuple[int, int, int, int],
) -> Image.Image:
    """Project a camera-space posed mesh through K and alpha-composite onto `image`.

    `verts_cam` must already be in camera space (rotated + translated) — the
    two tiers get there differently (see callers below); rasterisation from
    here on is identical. Mirrors
    scantosmpl.hmr.consensus._render_consensus_overlay's shading/culling.
    """
    W, H = image.size
    v_z = verts_cam[:, 2]
    pts_h = (K @ verts_cam.T).T
    pts_2d = pts_h[:, :2] / (pts_h[:, 2:3] + 1e-9)

    f = np.asarray(faces)
    all_front = (v_z[f[:, 0]] > 0.01) & (
        v_z[f[:, 1]] > 0.01) & (v_z[f[:, 2]] > 0.01)
    vis_faces = f[all_front]

    v0, v1, v2 = pts_2d[vis_faces[:, 0]
                        ], pts_2d[vis_faces[:, 1]], pts_2d[vis_faces[:, 2]]
    cross = (v1[:, 0] - v0[:, 0]) * (v2[:, 1] - v0[:, 1]) - (v1[:, 1] - v0[:, 1]) * (
        v2[:, 0] - v0[:, 0]
    )
    vis_faces = vis_faces[cross > 0]

    z_cent = (v_z[vis_faces[:, 0]] + v_z[vis_faces[:, 1]] +
              v_z[vis_faces[:, 2]]) / 3.0
    vis_faces = vis_faces[np.argsort(z_cent)[::-1]]

    base = image.convert("RGBA")
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)

    margin = 200
    for face in vis_faces:
        p0 = (float(pts_2d[face[0], 0]), float(pts_2d[face[0], 1]))
        p1 = (float(pts_2d[face[1], 0]), float(pts_2d[face[1], 1]))
        p2 = (float(pts_2d[face[2], 0]), float(pts_2d[face[2], 1]))
        if all(
            -margin <= p[0] <= W + margin and -margin <= p[1] <= H + margin for p in (p0, p1, p2)
        ):
            draw.polygon([p0, p1, p2], fill=fill_rgba)

    mesh = trimesh.Trimesh(vertices=verts_cam, faces=faces, process=False)
    edges = mesh.edges_unique
    both_front = (v_z[edges[:, 0]] > 0.01) & (v_z[edges[:, 1]] > 0.01)
    edges = edges[both_front]
    max_edges = 4000
    if len(edges) > max_edges:
        edges = edges[np.linspace(0, len(edges) - 1, max_edges, dtype=int)]
    for e0, e1 in edges:
        x0, y0 = float(pts_2d[e0, 0]), float(pts_2d[e0, 1])
        x1, y1 = float(pts_2d[e1, 0]), float(pts_2d[e1, 1])
        if 0 <= x0 <= W and 0 <= y0 <= H and 0 <= x1 <= W and 0 <= y1 <= H:
            draw.line([(x0, y0), (x1, y1)], fill=edge_rgba, width=1)

    return Image.alpha_composite(base, layer).convert("RGB")


def _label(image: Image.Image, text: str) -> Image.Image:
    img = image.copy()
    draw = ImageDraw.Draw(img)
    W, _ = img.size
    draw.rectangle([0, 0, W, 34], fill=(20, 20, 20))
    draw.text((10, 8), text, fill=(255, 255, 255))
    return img


def render_tier_comparison(
    view_name: str,
    image_dir: Path = DEFAULT_IMAGE_DIR,
    detections_path: Path = DEFAULT_DETECTIONS,
    hmr_results_path: Path = DEFAULT_HMR_RESULTS,
    consensus_debug_dir: Path = DEFAULT_CONSENSUS_DIR,
    refinement_debug_dir: Path = DEFAULT_REFINEMENT_DIR,
    smpl_model_path: Path = DEFAULT_SMPL_MODEL,
    gender: str = "neutral",
) -> Image.Image:
    """Render ``[photo | Tier 1 consensus | Tier 2 refined]`` side by side for one view."""
    import smplx
    import torch

    image = _open_view_image(image_dir, view_name)

    detections = json.load(open(detections_path))
    det = next((d for d in detections if d["filename"] == view_name), None)
    if det is None:
        raise ValueError(f"{view_name} not found in {detections_path}")
    f_px = float(det["focal_length_px"])

    hmr_results = json.load(open(hmr_results_path))
    if view_name not in hmr_results:
        raise ValueError(f"{view_name} not found in {hmr_results_path}")
    view_hmr = hmr_results[view_name]

    consensus_results = json.load(
        open(consensus_debug_dir / "consensus_results.json"))
    refinement_results = json.load(
        open(refinement_debug_dir / "refinement_results.json"))

    cam2 = refinement_results["cameras"].get(view_name)
    if cam2 is None:
        raise ValueError(
            f"{view_name} has no solved camera in refinement_results.json "
            f"(excluded for insufficient extrinsics, or classified as a rear view)"
        )

    smpl = smplx.create(
        str(smpl_model_path),
        model_type="smpl",
        gender=gender,
        use_face_contour=False,
    ).eval()
    faces = smpl.faces

    # --- Tier 1: consensus body re-posed with this view's own per-image
    # global_orient (no shared world frame in Tier 1), assumed image-centre K
    # — exactly scantosmpl.hmr.consensus._render_consensus_overlay's approach.
    W, H = image.size
    K_tier1 = np.array(
        [[f_px, 0.0, W / 2.0], [0.0, f_px, H / 2.0], [0.0, 0.0, 1.0]])
    with torch.no_grad():
        out = smpl(
            global_orient=torch.tensor(
                view_hmr["global_orient"]).float().unsqueeze(0),
            body_pose=torch.tensor(
                consensus_results["body_pose"]).float().unsqueeze(0),
            betas=torch.tensor(
                consensus_results["betas"]).float().unsqueeze(0),
        )
    verts_tier1 = out.vertices[0].numpy(
    ) + np.array(view_hmr["cam_translation"])
    tier1_img = _render_mesh_overlay(
        image, verts_tier1, faces, K_tier1, TIER1_FILL, TIER1_EDGE)

    # --- Tier 2: refined mesh (already world/SMPL-frame), through the actual
    # self-calibrated camera [R|t|K] (K includes Phase 4's principal-point fix).
    R = np.array(cam2["R"])
    t = np.array(cam2["t"])
    K_tier2 = np.array(cam2["K"])
    tier2_mesh = trimesh.load(
        str(refinement_debug_dir / "refined_mesh.obj"), process=False)
    # trimesh.load()'s return type covers Scene/PointCloud too; a single-object
    # .obj with faces always loads as Trimesh, which does have `.vertices`.
    assert isinstance(tier2_mesh, trimesh.Trimesh)
    verts_world = np.asarray(tier2_mesh.vertices)
    verts_tier2 = (R @ verts_world.T).T + t
    tier2_img = _render_mesh_overlay(
        image, verts_tier2, faces, K_tier2, TIER2_FILL, TIER2_EDGE)

    # --- assemble side by side ---
    scale = 900.0 / H
    panels = [
        _label(image, "Photo"),
        _label(tier1_img, "Tier 1 - consensus"),
        _label(tier2_img, "Tier 2 - refined"),
    ]
    panels = [
        p.resize((int(p.width * scale), int(p.height * scale)),
                 Image.Resampling.LANCZOS)
        for p in panels
    ]
    combined = Image.new(
        "RGB", (sum(p.width for p in panels), panels[0].height))
    x = 0
    for p in panels:
        combined.paste(p, (x, 0))
        x += p.width
    return combined


def list_of_strs(arg: str):
    return [a.strip() for a in arg.split(',')]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--views",
        # nargs="*",
        type=list_of_strs,
        default=DEFAULT_VIEWS,
        help="View filenames to render (default: the two views the Tier 1 "
        "consensus summary itself flags as worst-fitting)",
    )
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--detections", type=Path, default=DEFAULT_DETECTIONS)
    parser.add_argument("--hmr-results", type=Path,
                        default=DEFAULT_HMR_RESULTS)
    parser.add_argument("--consensus-dir", type=Path,
                        default=DEFAULT_CONSENSUS_DIR)
    parser.add_argument("--refinement-dir", type=Path,
                        default=DEFAULT_REFINEMENT_DIR)
    parser.add_argument("--smpl-model", type=Path, default=DEFAULT_SMPL_MODEL)
    parser.add_argument("--gender", default="neutral")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("output/debug/tier_comparison"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for view in args.views:
        combined = render_tier_comparison(
            view,
            image_dir=args.image_dir,
            detections_path=args.detections,
            hmr_results_path=args.hmr_results,
            consensus_debug_dir=args.consensus_dir,
            refinement_debug_dir=args.refinement_dir,
            smpl_model_path=args.smpl_model,
            gender=args.gender,
        )
        out_path = args.output_dir / f"{Path(view).stem}_tier_comparison.jpg"
        combined.save(out_path, quality=92)
        print(f"[tier-comparison] wrote {out_path}")


if __name__ == "__main__":
    main()
