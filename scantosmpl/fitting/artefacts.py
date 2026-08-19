"""The 7.B PSD-boundary artefact + manifest writer (master §5.3, §7.1, §7.2).

This module is the **enforcement layer** for REVIEW.md 7.B1-7.B8: every assertion here
exists to make a corrupt artefact fail loudly, at write time, rather than silently
producing a plausible-looking but wrong `D` that a downstream PSD consumer would train
on. Read each assertion's comment for which 7.B requirement it discharges.

Two files intentionally never touch each other's concerns:

* `displacements.npz` holds ONLY `D` — the per-vertex offset field — plus the metadata
  needed to interpret it (frame, topology hash, vertex/face counts).
* `alignment.json` holds ONLY the cloud -> SMPL similarity (scale/rotation/translation).

Nothing about the similarity is folded into `D` (7.B5) — keeping them in separate files
makes the separation visible on disk, not just in code, which is the whole point of
AC18's "similarity invariance" check downstream.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from scantosmpl.fitting.surface import SurfaceFitResult
from scantosmpl.pointcloud.align import CloudAlignment
from scantosmpl.types import (
    DISPLACEMENT_FRAME,
    SMPL_NUM_FACES,
    SMPL_NUM_VERTICES,
    PoseArtefact,
    Tier3Quality,
)

__all__ = [
    "SMPL_TEMPLATE_FACES_SHA256",
    "faces_sha256",
    "write_pose_artefacts",
    "update_manifest",
    "load_locked_betas",
]

#: sha256 of `np.ascontiguousarray(faces, dtype=np.int64).tobytes()` for the canonical
#: SMPL (not SMPL-X) triangle topology. Measured directly from this repo's
#: `models/smpl/SMPL_{NEUTRAL,MALE,FEMALE}.pkl` via `smplx.create(...).faces` and
#: confirmed byte-identical across all three — SMPL's topology never varies by gender,
#: only vertex positions / shape directions do. Deliberately a hard-coded constant
#: (not read from a live model here) so 7.B4's ordering check is self-contained and
#: needs no SMPL weights loaded inside this module: any caller that hands
#: `write_pose_artefacts` a resampled, remeshed or reordered face array — even one that
#: still happens to be `(13776, 3)` — is caught here, because a reordering changes the
#: byte sequence even though it changes neither shape nor the *set* of indices used.
SMPL_TEMPLATE_FACES_SHA256 = "19710f11eb74fe51d7e33eb10372acbc86fb24d21cbc040cdf80ac793fb5d971"

_NUM_BETAS = 10


def faces_sha256(faces: np.ndarray) -> str:
    """sha256 of a `(F, 3)` face array's canonical (contiguous, int64) byte layout.

    Public so `Tier3Pipeline` can compute `smpl_meta["faces_sha256"]` for the manifest
    with the exact same byte convention `write_pose_artefacts` checks against.
    """
    arr = np.ascontiguousarray(faces, dtype=np.int64)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _assert_finite(name: str, arr: np.ndarray | float) -> None:
    values = np.asarray(arr, dtype=np.float64)
    if not np.isfinite(values).all():
        raise AssertionError(
            f"{name} contains NaN/Inf — refusing to write a corrupt Tier 3 artefact. "
            "A NaN silently propagated into D would poison every PSD training sample "
            "that reads this pose."
        )


def _save_obj(vertices: np.ndarray, faces: np.ndarray, path: Path, *, pose_name: str) -> None:
    """Write a simple Wavefront .obj — `base_vertices + D`, template face ordering."""
    with open(path, "w") as f:
        f.write(
            f"# ScanToSMPL Tier 3 registered mesh ({pose_name}): base_vertices + D, "
            "posed_world frame, metres\n"
        )
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            # OBJ is 1-indexed.
            f.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")


def write_pose_artefacts(
    out_dir: Path,
    fit: SurfaceFitResult,
    alignment: CloudAlignment,
    quality: Tier3Quality,
    faces: np.ndarray,
    *,
    pose_name: str,
) -> None:
    """Write `smpl_params.npz`, `displacements.npz`, `registered.obj`, `alignment.json`,
    `quality.json` into `out_dir` (master §7.1's five mandatory files — the sixth,
    `pointcloud_aligned.ply`, is debug-only and written by the caller when enabled).

    Asserts on write, in order (fail loudly, never silently corrupt the PSD residual):
      1. `D.shape == (6890, 3)` and `dtype == float32`                    (7.B2, 7.B4)
      2. `vertices` / `base_vertices` shape `(6890, 3)`, `faces` shape `(13776, 3)` (7.B4)
      3. `faces` byte-identical (sha256) to the SMPL template topology     (7.B4 ordering)
      4. no NaN/Inf anywhere in `D`, vertices or the SMPL parameters
      5. `allclose(base_vertices + D, vertices)`                          (D4 identity)
      6. `displacement_frame == "posed_world"`, written as an explicit field (7.B3)

    `D` is written **unconditionally** — there is no config flag that suppresses it
    (7.B2 supersedes REVIEW.md 7.6's "if enabled").

    Args:
        out_dir: Per-pose artefact directory, e.g. `output/fits/<subject>/<pose_name>/`.
            Created if missing.
        fit: The Tier 3 `SurfaceFitResult` to persist.
        alignment: The cloud -> SMPL `CloudAlignment` (kept in a separate file, 7.B5).
        quality: Per-pose `Tier3Quality` (7.B7).
        faces: `(13776, 3)` integer face indices — MUST be the SMPL template topology.
        pose_name: Recorded in `registered.obj`'s header comment only (provenance).

    Raises:
        AssertionError: On any of the six checks above.
    """
    D = np.asarray(fit.displacements)
    vertices = np.asarray(fit.vertices)
    base_vertices = np.asarray(fit.base_vertices)
    faces_arr = np.asarray(faces)

    # --- 1. D shape + dtype (7.B2 / 7.B4) -----------------------------------
    if D.shape != (SMPL_NUM_VERTICES, 3):
        raise AssertionError(
            f"D.shape must be ({SMPL_NUM_VERTICES}, 3) — got {D.shape}. A resampled or "
            "partial displacement field cannot be index-aligned with the PSD blend-shape "
            "targets (7.B2/7.B4)."
        )
    if D.dtype != np.float32:
        raise AssertionError(f"D.dtype must be float32 — got {D.dtype} (7.B2/7.B4).")

    # --- 2. vertex / face shapes (7.B4) -------------------------------------
    if vertices.shape != (SMPL_NUM_VERTICES, 3):
        raise AssertionError(
            f"vertices.shape must be ({SMPL_NUM_VERTICES}, 3) — got {vertices.shape} (7.B4)."
        )
    if base_vertices.shape != (SMPL_NUM_VERTICES, 3):
        raise AssertionError(
            f"base_vertices.shape must be ({SMPL_NUM_VERTICES}, 3) — got "
            f"{base_vertices.shape} (7.B4)."
        )
    if faces_arr.shape != (SMPL_NUM_FACES, 3):
        raise AssertionError(
            f"faces.shape must be ({SMPL_NUM_FACES}, 3) — got {faces_arr.shape} (7.B4)."
        )

    # --- 3. faces byte-identical to the SMPL template topology (7.B4) ------
    faces_hash = faces_sha256(faces_arr)
    if faces_hash != SMPL_TEMPLATE_FACES_SHA256:
        raise AssertionError(
            f"faces are not byte-identical to the SMPL template topology "
            f"(sha256 {faces_hash} != {SMPL_TEMPLATE_FACES_SHA256}). A permuted or "
            "remeshed/resampled face array is index-aligned with nothing — every PSD "
            "blend-shape target built from this pose would be silently scrambled (7.B4)."
        )

    # --- 4. no NaN/Inf anywhere ---------------------------------------------
    _assert_finite("displacements", D)
    _assert_finite("vertices", vertices)
    _assert_finite("base_vertices", base_vertices)
    _assert_finite("betas", fit.betas)
    _assert_finite("body_pose", fit.body_pose)
    _assert_finite("global_orient", fit.global_orient)
    _assert_finite("translation", fit.translation)
    _assert_finite("scale", fit.scale)

    # --- 5. the D4 identity: base_vertices + D == vertices ------------------
    if not np.allclose(
        base_vertices.astype(np.float64) + D.astype(np.float64),
        vertices.astype(np.float64),
        atol=1e-6,
    ):
        raise AssertionError(
            "base_vertices + D != vertices — the D4 identity is violated. D would not be "
            "a pure difference against the baseline PSD regenerates bit-identically, which "
            "is precisely the frame ambiguity 7.B3/7.B5 exist to prevent."
        )

    # --- 6. displacement_frame is the explicit, asserted field (7.B3) ------
    displacement_frame = DISPLACEMENT_FRAME
    if displacement_frame != "posed_world":
        raise AssertionError(  # pragma: no cover — guards a future edit to the constant
            f"DISPLACEMENT_FRAME must be 'posed_world', got {displacement_frame!r} (7.B3)."
        )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- smpl_params.npz -----------------------------------------------------
    np.savez(
        out_dir / "smpl_params.npz",
        betas=np.asarray(fit.betas, dtype=np.float64).reshape(_NUM_BETAS),
        body_pose=np.asarray(fit.body_pose, dtype=np.float64).reshape(-1),
        global_orient=np.asarray(fit.global_orient, dtype=np.float64).reshape(3),
        translation=np.asarray(fit.translation, dtype=np.float64).reshape(3),
        scale=np.float64(fit.scale),
        base_vertices=base_vertices.astype(np.float32),
        vertices=vertices.astype(np.float32),
        betas_locked=bool(fit.betas_locked),
    )

    # --- displacements.npz — D, unconditionally (7.B2) ------------------------
    np.savez(
        out_dir / "displacements.npz",
        D=D.astype(np.float32),
        displacement_frame=displacement_frame,
        faces_sha256=faces_hash,
        n_vertices=np.int64(SMPL_NUM_VERTICES),
        n_faces=np.int64(SMPL_NUM_FACES),
    )

    # --- registered.obj — base_vertices + D, template face ordering ----------
    _save_obj(vertices, faces_arr, out_dir / "registered.obj", pose_name=pose_name)

    # --- alignment.json — the cloud -> SMPL similarity, kept SEPARATE (7.B5) --
    alignment_data = {
        "scale": float(alignment.scale),
        "rotation": np.asarray(alignment.rotation, dtype=np.float64).tolist(),
        "translation": np.asarray(alignment.translation, dtype=np.float64).tolist(),
        "inlier_rmse_m": float(alignment.inlier_rmse_m),
        "fitness": float(alignment.fitness),
        "n_candidates": int(alignment.n_candidates),
        "candidate_index": int(alignment.candidate_index),
        "converged": bool(alignment.converged),
    }
    with open(out_dir / "alignment.json", "w") as f:
        json.dump(alignment_data, f, indent=2)

    # --- quality.json — Tier3Quality, flat (7.B7) -----------------------------
    with open(out_dir / "quality.json", "w") as f:
        json.dump(asdict(quality), f, indent=2)


def _assert_manifest_compatible(manifest: dict, *, subject_id: str, smpl_meta: dict) -> None:
    """Raise if `manifest` disagrees with the current run on the fields 7.B6 requires
    to stay fixed across every pose of one subject: `subject_id`, `displacement_frame`,
    `gender`, `num_betas`, `faces_sha256`."""
    existing_subject = manifest.get("subject_id")
    if existing_subject != subject_id:
        raise ValueError(
            f"manifest at this path already belongs to subject_id={existing_subject!r}; "
            f"refusing to merge subject_id={subject_id!r} into it — a corpus manifest "
            "holds exactly one subject (7.B6)."
        )

    existing_frame = manifest.get("displacement_frame")
    if existing_frame is not None and existing_frame != DISPLACEMENT_FRAME:
        raise ValueError(
            f"manifest displacement_frame={existing_frame!r} disagrees with the current "
            f"run's {DISPLACEMENT_FRAME!r} (7.B3/7.B6)."
        )

    existing_smpl = manifest.get("smpl", {})
    for key in ("gender", "num_betas", "faces_sha256"):
        existing_value = existing_smpl.get(key)
        new_value = smpl_meta.get(key)
        if existing_value is not None and existing_value != new_value:
            raise ValueError(
                f"manifest smpl.{key}={existing_value!r} disagrees with the current run's "
                f"{new_value!r} — every pose of one subject must share the same SMPL setup "
                "(7.B6)."
            )


def update_manifest(
    manifest_path: Path,
    entry: PoseArtefact,
    *,
    subject_id: str,
    smpl_meta: dict,
    beta_policy: dict,
) -> None:
    """Create-or-update `output/fits/<subject>/manifest.json` (master §7.2, 7.B6).

    A pose already present under `entry.pose_name` is replaced (idempotent re-runs);
    every other pose entry is preserved. `beta_policy` reflects the CURRENT run — the
    manifest's authoritative record of "what mode was this subject last fitted under"
    (master D10) — while each pose's own `PoseArtefact.betas_locked` records that
    specific pose's mode, so both β modes are visible: the top-level policy and every
    pose's individual flag (D10's "record both β modes").

    Args:
        manifest_path: `output/fits/<subject>/manifest.json`.
        entry: This pose's manifest row.
        subject_id: Must match every other pose already in the manifest.
        smpl_meta: `{"gender", "num_betas", "num_vertices", "num_faces", "faces_sha256"}`.
        beta_policy: `{"mode": "refined"|"locked", "source_pose": str, "betas_sha256": str}`.

    Raises:
        ValueError: If an existing manifest disagrees on `subject_id`,
            `displacement_frame`, `gender`, `num_betas` or `faces_sha256`.
    """
    manifest_path = Path(manifest_path)

    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        _assert_manifest_compatible(manifest, subject_id=subject_id, smpl_meta=smpl_meta)
    else:
        manifest = {
            "schema_version": 1,
            "subject_id": subject_id,
            "displacement_frame": DISPLACEMENT_FRAME,
            "smpl": {},
            "beta_policy": {},
            "poses": [],
        }

    manifest["subject_id"] = subject_id
    manifest["displacement_frame"] = DISPLACEMENT_FRAME
    manifest["smpl"] = dict(smpl_meta)
    manifest["beta_policy"] = dict(beta_policy)

    poses = [p for p in manifest.get("poses", []) if p.get("pose_name") != entry.pose_name]
    poses.append(asdict(entry))
    manifest["poses"] = poses

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)


def load_locked_betas(path: Path) -> np.ndarray:
    """Load `(10,)` float64 betas from a reference-pose `smpl_params.npz` (7.B1).

    Used for `--betas-from`: every subsequent pose of a subject is fitted with these
    betas frozen (master D10).

    Args:
        path: Path to a `smpl_params.npz` written by `write_pose_artefacts`.

    Returns:
        `(10,)` float64 betas.

    Raises:
        FileNotFoundError: If `path` does not exist.
        ValueError: If the file has no `betas` array, or it is not `(10,)`.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"--betas-from path not found: {path}")

    with np.load(path) as data:
        if "betas" not in data:
            raise ValueError(f"{path} has no 'betas' array — is this a smpl_params.npz?")
        betas = np.asarray(data["betas"], dtype=np.float64).reshape(-1)

    if betas.shape != (_NUM_BETAS,):
        raise ValueError(f"betas in {path} must be ({_NUM_BETAS},), got {betas.shape}")
    return betas
