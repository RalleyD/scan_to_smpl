"""CLI entry point for ScanToSMPL."""

import json
from pathlib import Path

import click
import numpy as np
import torch

from scantosmpl.config import ModelPaths, Tier3Config
from scantosmpl.fitting.optimiser import RefinementResult
from scantosmpl.fitting.surface_pipeline import Tier3Pipeline
from scantosmpl.smpl.model import SMPLModel


@click.group()
@click.version_option()
def main():
    """ScanToSMPL: Register SMPL meshes to multi-view images and point clouds."""
    pass


@main.command()
@click.option(
    "--image-dir", required=True, type=click.Path(exists=True), help="Directory of images"
)
@click.option("--reference-pose", default="a-pose", type=click.Choice(["a-pose", "t-pose"]))
@click.option("--gender", default="neutral", type=click.Choice(["neutral", "male", "female"]))
@click.option("--output", required=True, type=click.Path(), help="Output directory")
def fit_images(image_dir, reference_pose, gender, output):
    """Fit SMPL to multi-view images (Tier 1 + Tier 2)."""
    click.echo(f"Fitting SMPL to images in {image_dir}")
    raise NotImplementedError("Phase 8: pipeline orchestrator")


@main.command()
@click.option("--pointcloud", required=True, type=click.Path(exists=True), help="PLY/OBJ file")
@click.option("--gender", default="neutral", type=click.Choice(["neutral", "male", "female"]))
@click.option("--output", required=True, type=click.Path(), help="Output directory")
def fit_pointcloud(pointcloud, gender, output):
    """Fit SMPL to a point cloud (Tier 3 only)."""
    click.echo(f"Fitting SMPL to point cloud {pointcloud}")
    raise NotImplementedError("Phase 8: pipeline orchestrator")


@main.command()
@click.option(
    "--image-dir", required=True, type=click.Path(exists=True), help="Directory of images"
)
@click.option("--pointcloud", required=True, type=click.Path(exists=True), help="PLY/OBJ file")
@click.option("--reference-pose", default="a-pose", type=click.Choice(["a-pose", "t-pose"]))
@click.option("--gender", default="neutral", type=click.Choice(["neutral", "male", "female"]))
@click.option("--output", required=True, type=click.Path(), help="Output directory")
def fit_combined(image_dir, pointcloud, reference_pose, gender, output):
    """Fit SMPL to images + point cloud (Tier 1 + 2 + 3, best accuracy)."""
    click.echo("Fitting SMPL to images + point cloud")
    raise NotImplementedError("Phase 8: pipeline orchestrator")


def _load_tier2_result(tier2_dir: Path, smpl_model: SMPLModel) -> RefinementResult:
    """Load a Phase 5 `RefinementResult` from a `Phase5Pipeline` debug directory.

    Reads `refinement_results.json` (written by
    `scantosmpl.fitting.pipeline.Phase5Pipeline._save_debug`) for the SMPL
    parameters + Tier 2 metrics, then re-runs the SMPL forward pass (D=0) to get
    `vertices`/`joints` — the same values `refined_mesh.obj` was built from, just
    without needing to re-parse a 20k-line .obj file.
    """
    results_path = Path(tier2_dir) / "refinement_results.json"
    if not results_path.exists():
        raise click.ClickException(
            f"{results_path} not found — --tier2-dir must be a Phase 5 debug directory "
            "(e.g. output/debug/refinement/) containing refinement_results.json."
        )
    with open(results_path) as f:
        data = json.load(f)

    betas = np.asarray(data["betas"], dtype=np.float32)
    body_pose = np.asarray(data["body_pose"], dtype=np.float32)
    global_orient = np.asarray(data["global_orient"], dtype=np.float32)
    translation = np.asarray(data["translation"], dtype=np.float32)
    scale = float(data["scale"])
    metrics = dict(data.get("metrics", {}))

    device = smpl_model.device
    with torch.no_grad():
        output = smpl_model.forward(
            betas=torch.as_tensor(betas, device=device).reshape(1, -1),
            body_pose=torch.as_tensor(body_pose, device=device).reshape(1, -1),
            global_orient=torch.as_tensor(global_orient, device=device).reshape(1, -1),
            translation=torch.as_tensor(translation, device=device).reshape(1, -1),
            scale=torch.tensor([scale], device=device),
            apply_displacements=False,
        )
    vertices = output.vertices.squeeze(0).cpu().numpy().astype(np.float64)
    joints = output.joints.squeeze(0).cpu().numpy().astype(np.float64)

    return RefinementResult(
        betas=betas,
        body_pose=body_pose,
        global_orient=global_orient,
        translation=translation,
        scale=scale,
        vertices=vertices,
        joints=joints,
        metrics=metrics,
    )


@main.command("fit-surface")
@click.option(
    "--tier2-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Phase 5 debug directory (contains refinement_results.json)",
)
@click.option(
    "--pointcloud",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="PLY/OBJ scan, arbitrary source frame/units",
)
@click.option("--subject", default="subject", help="Subject id — one manifest per subject")
@click.option("--pose-name", required=True, help="e.g. t-pose, a-pose, a-pose-heldout")
@click.option(
    "--lock-betas", is_flag=True, default=False, help="Freeze beta from --betas-from (7.B1)"
)
@click.option(
    "--betas-from",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Reference-pose smpl_params.npz; implies --lock-betas",
)
@click.option(
    "--oracle-only",
    is_flag=True,
    default=False,
    help="Flag this pose as an evaluation ceiling — MUST NOT enter PSD training (7.B8)",
)
@click.option(
    "--no-semantic-weighting",
    is_flag=True,
    default=False,
    help="Disable lbs_weights-derived body-part chamfer weighting (AC 7.3 A/B)",
)
@click.option("--gender", default="neutral", type=click.Choice(["neutral", "male", "female"]))
@click.option("--output", required=True, type=click.Path(path_type=Path), help="Subject output dir")
def fit_surface(
    tier2_dir: Path,
    pointcloud: Path,
    subject: str,
    pose_name: str,
    lock_betas: bool,
    betas_from: Path | None,
    oracle_only: bool,
    no_semantic_weighting: bool,
    gender: str,
    output: Path,
) -> None:
    """Fit SMPL+D surface refinement to a point cloud (Tier 3 only entry point)."""
    if betas_from is not None:
        lock_betas = True  # --betas-from implies --lock-betas
    if lock_betas and betas_from is None:
        raise click.UsageError(
            "--lock-betas requires --betas-from — there is nothing to lock beta TO."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    smpl_model = SMPLModel(model_dir=ModelPaths().smpl_dir, gender=gender, device=device)

    tier2 = _load_tier2_result(tier2_dir, smpl_model)

    cfg = Tier3Config(
        lock_betas=lock_betas,
        betas_source=betas_from,
        subject_id=subject,
        oracle_only=oracle_only,
        use_semantic_weighting=not no_semantic_weighting,
    )

    pipeline = Tier3Pipeline(smpl_model, cfg)
    result = pipeline.run(tier2, pointcloud, pose_name=pose_name, output_dir=output)

    click.echo(f"Tier 3 complete: {result.artefact_dir}")
    click.echo(
        f"cloud->mesh mean {result.report.cloud_to_mesh_mm['mean']:.2f} mm | "
        f"mesh->cloud mean {result.report.mesh_to_cloud_mm['mean']:.2f} mm | "
        f"tessellation floor mean {result.report.tessellation_floor_mm['mean']:.2f} mm"
    )
