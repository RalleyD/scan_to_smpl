"""CLI entry point for ScanToSMPL."""

import click


@click.group()
@click.version_option()
def main():
    """ScanToSMPL: Register SMPL meshes to multi-view images and point clouds."""
    pass


@main.command()
@click.option("--image-dir", required=True, type=click.Path(exists=True), help="Directory of images")
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
@click.option("--image-dir", required=True, type=click.Path(exists=True), help="Directory of images")
@click.option("--pointcloud", required=True, type=click.Path(exists=True), help="PLY/OBJ file")
@click.option("--reference-pose", default="a-pose", type=click.Choice(["a-pose", "t-pose"]))
@click.option("--gender", default="neutral", type=click.Choice(["neutral", "male", "female"]))
@click.option("--output", required=True, type=click.Path(), help="Output directory")
def fit_combined(image_dir, pointcloud, reference_pose, gender, output):
    """Fit SMPL to images + point cloud (Tier 1 + 2 + 3, best accuracy)."""
    click.echo(f"Fitting SMPL to images + point cloud")
    raise NotImplementedError("Phase 8: pipeline orchestrator")
