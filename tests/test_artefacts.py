"""Unit tests for the 7.B PSD-boundary artefact + manifest writer
(`scantosmpl.fitting.artefacts`). Discharges AC15, AC17, AC19, AC20, AC21.

Per the brief: every assertion needs a FAILING-path test, not just a happy-path one —
most tests below construct the exact violating input and assert it raises.

Face-topology tests use a MONKEYPATCHED `SMPL_TEMPLATE_FACES_SHA256` pointed at a
deterministic dummy `(13776, 3)` face array, so the sha256-comparison LOGIC (the heart
of 7.B4's "reordering cannot slip through" guarantee) is fully testable without SMPL
model weights. Exactly one test (`test_real_smpl_faces_match_template_constant`,
`requires_smpl`) ties the hard-coded constant to genuine SMPL topology loaded from
`models/smpl/`.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from scantosmpl.fitting import artefacts
from scantosmpl.fitting.artefacts import (
    SMPL_TEMPLATE_FACES_SHA256,
    faces_sha256,
    load_locked_betas,
    update_manifest,
    write_pose_artefacts,
)
from scantosmpl.fitting.surface import SurfaceFitResult
from scantosmpl.pointcloud.align import CloudAlignment
from scantosmpl.types import (
    DISPLACEMENT_FRAME,
    SMPL_NUM_FACES,
    SMPL_NUM_VERTICES,
    PoseArtefact,
    Tier3Quality,
)

SMPL_DIR = "models/smpl"


def _smpl_available() -> bool:
    return (Path(SMPL_DIR) / "SMPL_NEUTRAL.pkl").exists()


requires_smpl = pytest.mark.skipif(
    not _smpl_available(),
    reason=f"SMPL model files not found in {SMPL_DIR}/ — see models/README.md",
)


# ---------------------------------------------------------------------------
# Deterministic fixtures — no SMPL weights required for any of these
# ---------------------------------------------------------------------------

#: A fixed, deterministic (13776, 3) dummy face array. Structurally valid (right
#: shape/dtype, indices in range) but NOT the real SMPL topology — used with a
#: monkeypatched `SMPL_TEMPLATE_FACES_SHA256` so the byte-identity CHECK is testable
#: without loading model weights.
_DUMMY_FACES = np.random.default_rng(0).integers(
    0, SMPL_NUM_VERTICES, size=(SMPL_NUM_FACES, 3), dtype=np.int64
)
_DUMMY_FACES_SHA256 = faces_sha256(_DUMMY_FACES)


@pytest.fixture
def dummy_topology(monkeypatch):
    """Point `SMPL_TEMPLATE_FACES_SHA256` at `_DUMMY_FACES`'s own hash, so tests can
    pass `_DUMMY_FACES` straight through `write_pose_artefacts`'s topology check
    without needing real SMPL weights."""
    monkeypatch.setattr(artefacts, "SMPL_TEMPLATE_FACES_SHA256", _DUMMY_FACES_SHA256)
    return _DUMMY_FACES


def _dummy_alignment(**overrides) -> CloudAlignment:
    base = dict(
        scale=0.5,
        rotation=np.eye(3),
        translation=np.array([0.1, -0.2, 0.3]),
        inlier_rmse_m=0.004,
        fitness=0.92,
        n_candidates=24,
        candidate_index=3,
        converged=True,
    )
    base.update(overrides)
    return CloudAlignment(**base)


def _dummy_quality(**overrides) -> Tier3Quality:
    base = dict(
        chamfer_cloud_to_mesh_mean_mm=5.0,
        chamfer_cloud_to_mesh_median_mm=4.5,
        chamfer_cloud_to_mesh_rms_mm=5.5,
        chamfer_mesh_to_cloud_mean_mm=4.0,
        chamfer_mesh_to_cloud_median_mm=3.8,
        chamfer_mesh_to_cloud_rms_mm=4.2,
        tessellation_floor_mean_mm=2.0,
        tessellation_floor_max_mm=6.0,
        icp_inlier_rmse_mm=4.0,
        icp_fitness=0.92,
        displacement_mean_mm=1.0,
        displacement_p95_mm=3.5,
        pa_mpjpe_mm=32.0,
        median_reproj_px=45.0,
    )
    base.update(overrides)
    return Tier3Quality(**base)


def _dummy_fit(
    *,
    seed: int = 0,
    betas_locked: bool = False,
    d_shape: tuple[int, int] = (SMPL_NUM_VERTICES, 3),
    d_dtype: type = np.float32,
    nan_in_d: bool = False,
    break_identity: bool = False,
) -> SurfaceFitResult:
    """A fully synthetic (no SMPL weights needed) `SurfaceFitResult`."""
    rng = np.random.default_rng(seed)
    base_vertices = rng.normal(scale=0.3, size=(SMPL_NUM_VERTICES, 3)).astype(np.float32)
    D = rng.normal(scale=0.001, size=d_shape).astype(d_dtype)
    if nan_in_d:
        D = D.copy()
        D[0, 0] = np.nan

    if D.shape == base_vertices.shape:
        vertices = base_vertices + D.astype(np.float32)
        if break_identity:
            vertices = vertices + np.float32(1.0)
    else:
        vertices = base_vertices.copy()  # shape mismatch — write_pose_artefacts must
        # raise on D's shape before ever reaching the identity check.

    return SurfaceFitResult(
        betas=rng.normal(size=10),
        body_pose=rng.normal(size=69),
        global_orient=rng.normal(size=3),
        translation=rng.normal(size=3),
        scale=1.0,
        displacements=D,
        vertices=vertices,
        base_vertices=base_vertices,
        betas_locked=betas_locked,
        loss_history={"model_fit": [1.0, 0.5], "displacement": [0.5, 0.1]},
        metrics={},
    )


# ---------------------------------------------------------------------------
# write_pose_artefacts — happy path
# ---------------------------------------------------------------------------


class TestWritePoseArtefactsHappyPath:
    def test_writes_all_five_files(self, tmp_path, dummy_topology):
        fit = _dummy_fit()
        alignment = _dummy_alignment()
        quality = _dummy_quality()
        out_dir = tmp_path / "t-pose"

        write_pose_artefacts(out_dir, fit, alignment, quality, dummy_topology, pose_name="t-pose")

        for name in (
            "smpl_params.npz",
            "displacements.npz",
            "registered.obj",
            "alignment.json",
            "quality.json",
        ):
            assert (out_dir / name).exists(), f"missing {name}"

    def test_smpl_params_npz_contents(self, tmp_path, dummy_topology):
        fit = _dummy_fit()
        out_dir = tmp_path / "p"
        write_pose_artefacts(
            out_dir, fit, _dummy_alignment(), _dummy_quality(), dummy_topology, pose_name="p"
        )

        with np.load(out_dir / "smpl_params.npz") as data:
            assert data["betas"].shape == (10,)
            assert data["betas"].dtype == np.float64
            assert data["body_pose"].shape == (69,)
            assert data["global_orient"].shape == (3,)
            assert data["translation"].shape == (3,)
            assert data["scale"].dtype == np.float64
            assert data["base_vertices"].shape == (SMPL_NUM_VERTICES, 3)
            assert data["base_vertices"].dtype == np.float32
            assert data["vertices"].shape == (SMPL_NUM_VERTICES, 3)
            assert bool(data["betas_locked"]) is False

    def test_displacements_npz_contents(self, tmp_path, dummy_topology):
        fit = _dummy_fit()
        out_dir = tmp_path / "p"
        write_pose_artefacts(
            out_dir, fit, _dummy_alignment(), _dummy_quality(), dummy_topology, pose_name="p"
        )

        with np.load(out_dir / "displacements.npz") as data:
            D = data["D"]
            assert D.shape == (SMPL_NUM_VERTICES, 3)  # AC15
            assert D.dtype == np.float32  # AC15
            assert str(data["displacement_frame"]) == DISPLACEMENT_FRAME == "posed_world"  # AC16
            assert str(data["faces_sha256"]) == _DUMMY_FACES_SHA256
            assert int(data["n_vertices"]) == SMPL_NUM_VERTICES
            assert int(data["n_faces"]) == SMPL_NUM_FACES

    def test_displacements_persisted_even_when_near_zero(self, tmp_path, dummy_topology):
        """AC15 — D is written unconditionally, including a near-zero field. There is
        no config flag anywhere in this function's signature that could suppress it."""
        fit = _dummy_fit(seed=1)
        zero_d = np.zeros_like(fit.displacements)
        fit = dataclasses.replace(fit, displacements=zero_d, vertices=fit.base_vertices + zero_d)
        out_dir = tmp_path / "p"
        write_pose_artefacts(
            out_dir, fit, _dummy_alignment(), _dummy_quality(), dummy_topology, pose_name="p"
        )
        with np.load(out_dir / "displacements.npz") as data:
            assert data["D"].shape == (SMPL_NUM_VERTICES, 3)
            assert np.allclose(data["D"], 0.0)

    def test_registered_obj_line_counts(self, tmp_path, dummy_topology):
        fit = _dummy_fit()
        out_dir = tmp_path / "p"
        write_pose_artefacts(
            out_dir, fit, _dummy_alignment(), _dummy_quality(), dummy_topology, pose_name="p"
        )
        lines = (out_dir / "registered.obj").read_text().splitlines()
        v_lines = [ln for ln in lines if ln.startswith("v ")]
        f_lines = [ln for ln in lines if ln.startswith("f ")]
        assert len(v_lines) == SMPL_NUM_VERTICES
        assert len(f_lines) == SMPL_NUM_FACES

    def test_alignment_json_kept_separate_from_D(self, tmp_path, dummy_topology):
        """7.B5 — alignment.json holds ONLY the similarity; nothing about D."""
        import json

        fit = _dummy_fit()
        alignment = _dummy_alignment(scale=2.5, candidate_index=7)
        out_dir = tmp_path / "p"
        write_pose_artefacts(
            out_dir, fit, alignment, _dummy_quality(), dummy_topology, pose_name="p"
        )
        data = json.loads((out_dir / "alignment.json").read_text())
        assert data["scale"] == pytest.approx(2.5)
        assert data["candidate_index"] == 7
        assert data["converged"] is True
        assert "D" not in data and "displacements" not in data

    def test_quality_json_round_trips(self, tmp_path, dummy_topology):
        import json

        fit = _dummy_fit()
        quality = _dummy_quality(chamfer_cloud_to_mesh_mean_mm=7.25)
        out_dir = tmp_path / "p"
        write_pose_artefacts(
            out_dir, fit, _dummy_alignment(), quality, dummy_topology, pose_name="p"
        )
        data = json.loads((out_dir / "quality.json").read_text())
        assert data["chamfer_cloud_to_mesh_mean_mm"] == pytest.approx(7.25)
        assert data["pa_mpjpe_mm"] == pytest.approx(32.0)


# ---------------------------------------------------------------------------
# write_pose_artefacts — failing paths (one per assertion, brief step 2)
# ---------------------------------------------------------------------------


class TestWritePoseArtefactsFailingPaths:
    def test_resampled_vertex_count_raises(self, tmp_path, dummy_topology):
        fit = _dummy_fit(d_shape=(1000, 3))
        with pytest.raises(AssertionError, match="D.shape"):
            write_pose_artefacts(
                tmp_path / "p",
                fit,
                _dummy_alignment(),
                _dummy_quality(),
                dummy_topology,
                pose_name="p",
            )

    def test_wrong_d_dtype_raises(self, tmp_path, dummy_topology):
        fit = _dummy_fit(d_dtype=np.float64)
        with pytest.raises(AssertionError, match="float32"):
            write_pose_artefacts(
                tmp_path / "p",
                fit,
                _dummy_alignment(),
                _dummy_quality(),
                dummy_topology,
                pose_name="p",
            )

    def test_permuted_face_array_raises(self, tmp_path, dummy_topology):
        """7.B4 — a row-permutation of the SAME face set, same shape, must still be
        rejected: byte order is what carries the index alignment, not the set of
        indices used."""
        permuted = np.random.default_rng(1).permutation(dummy_topology)
        assert not np.array_equal(permuted, dummy_topology)  # sanity: genuinely permuted
        assert permuted.shape == dummy_topology.shape

        fit = _dummy_fit()
        with pytest.raises(AssertionError, match="byte-identical"):
            write_pose_artefacts(
                tmp_path / "p", fit, _dummy_alignment(), _dummy_quality(), permuted, pose_name="p"
            )

    def test_wrong_faces_shape_raises(self, tmp_path, dummy_topology):
        fit = _dummy_fit()
        bad_faces = dummy_topology[:100]
        with pytest.raises(AssertionError, match="faces.shape"):
            write_pose_artefacts(
                tmp_path / "p", fit, _dummy_alignment(), _dummy_quality(), bad_faces, pose_name="p"
            )

    def test_nan_in_d_raises(self, tmp_path, dummy_topology):
        fit = _dummy_fit(nan_in_d=True)
        with pytest.raises(AssertionError, match="NaN"):
            write_pose_artefacts(
                tmp_path / "p",
                fit,
                _dummy_alignment(),
                _dummy_quality(),
                dummy_topology,
                pose_name="p",
            )

    def test_inf_in_betas_raises(self, tmp_path, dummy_topology):
        fit = _dummy_fit()
        fit = dataclasses.replace(fit, betas=np.full(10, np.inf))
        with pytest.raises(AssertionError, match="NaN"):
            write_pose_artefacts(
                tmp_path / "p",
                fit,
                _dummy_alignment(),
                _dummy_quality(),
                dummy_topology,
                pose_name="p",
            )

    def test_base_plus_d_ne_vertices_raises(self, tmp_path, dummy_topology):
        """D4 identity — `vertices` was NOT built as `base_vertices + D`."""
        fit = _dummy_fit(break_identity=True)
        with pytest.raises(AssertionError, match="D4 identity"):
            write_pose_artefacts(
                tmp_path / "p",
                fit,
                _dummy_alignment(),
                _dummy_quality(),
                dummy_topology,
                pose_name="p",
            )


def test_topology_assertions(tmp_path, dummy_topology):
    """AC17 (7.B4) — writing with a resampled vertex count, a permuted face array, or
    a NaN in D each raise; a correct write round-trips 6890/13776 and `faces_sha256`
    matches the template."""
    alignment = _dummy_alignment()
    quality = _dummy_quality()

    # 1. Resampled vertex count.
    with pytest.raises(AssertionError, match="D.shape"):
        write_pose_artefacts(
            tmp_path / "resampled",
            _dummy_fit(d_shape=(4000, 3)),
            alignment,
            quality,
            dummy_topology,
            pose_name="resampled",
        )

    # 2. Permuted face array (same rows, different byte order).
    permuted = np.random.default_rng(2).permutation(dummy_topology)
    with pytest.raises(AssertionError, match="byte-identical"):
        write_pose_artefacts(
            tmp_path / "permuted",
            _dummy_fit(),
            alignment,
            quality,
            permuted,
            pose_name="permuted",
        )

    # 3. NaN in D.
    with pytest.raises(AssertionError, match="NaN"):
        write_pose_artefacts(
            tmp_path / "nan",
            _dummy_fit(nan_in_d=True),
            alignment,
            quality,
            dummy_topology,
            pose_name="nan",
        )

    # 4. A correct write round-trips 6890/13776 and faces_sha256 matches the template.
    out_dir = tmp_path / "correct"
    write_pose_artefacts(
        out_dir, _dummy_fit(), alignment, quality, dummy_topology, pose_name="correct"
    )
    with np.load(out_dir / "displacements.npz") as data:
        assert data["D"].shape == (SMPL_NUM_VERTICES, 3)
        assert int(data["n_vertices"]) == SMPL_NUM_VERTICES
        assert int(data["n_faces"]) == SMPL_NUM_FACES
        # Matches "the template" as `write_pose_artefacts` sees it in this sandbox —
        # `dummy_topology` monkeypatches SMPL_TEMPLATE_FACES_SHA256 to this exact
        # value; the un-patched, real constant is checked separately below by
        # `test_real_smpl_faces_match_template_constant`.
        assert str(data["faces_sha256"]) == _DUMMY_FACES_SHA256


@requires_smpl
def test_real_smpl_faces_match_template_constant():
    """Ties the hard-coded `SMPL_TEMPLATE_FACES_SHA256` constant to genuine SMPL
    topology loaded from `models/smpl/`, and confirms `write_pose_artefacts` accepts
    it WITHOUT any monkeypatching (the real, un-modified constant this module ships)."""
    import smplx

    model = smplx.SMPL(model_path=SMPL_DIR, gender="neutral")
    real_faces = model.faces.astype(np.int64)
    assert real_faces.shape == (SMPL_NUM_FACES, 3)
    assert faces_sha256(real_faces) == SMPL_TEMPLATE_FACES_SHA256


# ---------------------------------------------------------------------------
# update_manifest — AC19 (7.B6), AC21 (7.B8)
# ---------------------------------------------------------------------------


def test_manifest_roundtrip(tmp_path, dummy_topology):
    """AC19 (7.B6) — two fit-surface runs for different pose names produce ONE
    manifest with two poses[] entries; each entry's directory resolves to a dir
    containing (β, θ via smpl_params.npz, D via displacements.npz, quality.json).
    A second subject writing into the same manifest raises."""
    subject_dir = tmp_path / "dan"
    manifest_path = subject_dir / "manifest.json"
    smpl_meta = {
        "gender": "neutral",
        "num_betas": 10,
        "num_vertices": SMPL_NUM_VERTICES,
        "num_faces": SMPL_NUM_FACES,
        "faces_sha256": _DUMMY_FACES_SHA256,
    }

    for pose_name, oracle_only in (("t-pose", False), ("a-pose", True)):
        out_dir = subject_dir / pose_name
        fit = _dummy_fit(betas_locked=(pose_name != "t-pose"))
        write_pose_artefacts(
            out_dir, fit, _dummy_alignment(), _dummy_quality(), dummy_topology, pose_name=pose_name
        )
        entry = PoseArtefact(
            pose_name=pose_name,
            directory=pose_name,
            oracle_only=oracle_only,
            betas_locked=fit.betas_locked,
            has_displacements=True,
            has_pointcloud=True,
            quality=_dummy_quality(),
        )
        beta_policy = (
            {"mode": "refined", "source_pose": "t-pose", "betas_sha256": "abc"}
            if pose_name == "t-pose"
            else {"mode": "locked", "source_pose": "t-pose", "betas_sha256": "abc"}
        )
        update_manifest(
            manifest_path, entry, subject_id="dan", smpl_meta=smpl_meta, beta_policy=beta_policy
        )

    import json

    manifest = json.loads(manifest_path.read_text())
    assert manifest["subject_id"] == "dan"
    assert manifest["displacement_frame"] == "posed_world"
    assert len(manifest["poses"]) == 2
    names = {p["pose_name"] for p in manifest["poses"]}
    assert names == {"t-pose", "a-pose"}

    for pose in manifest["poses"]:
        pose_dir = subject_dir / pose["directory"]
        assert (pose_dir / "smpl_params.npz").exists()  # beta, theta
        assert (pose_dir / "displacements.npz").exists()  # D
        assert (pose_dir / "quality.json").exists()  # quality

    # A second subject writing into the SAME manifest path must raise.
    other_entry = PoseArtefact(
        pose_name="t-pose",
        directory="t-pose",
        oracle_only=False,
        betas_locked=False,
        has_displacements=True,
        has_pointcloud=True,
        quality=_dummy_quality(),
    )
    with pytest.raises(ValueError, match="subject"):
        update_manifest(
            manifest_path,
            other_entry,
            subject_id="someone-else",
            smpl_meta=smpl_meta,
            beta_policy={"mode": "refined", "source_pose": "t-pose", "betas_sha256": "abc"},
        )


def test_manifest_rejects_gender_mismatch(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    smpl_meta_a = {
        "gender": "neutral",
        "num_betas": 10,
        "num_vertices": SMPL_NUM_VERTICES,
        "num_faces": SMPL_NUM_FACES,
        "faces_sha256": "aaa",
    }
    entry = PoseArtefact(
        pose_name="t-pose",
        directory="t-pose",
        oracle_only=False,
        betas_locked=False,
        has_displacements=True,
        has_pointcloud=True,
        quality=_dummy_quality(),
    )
    update_manifest(
        manifest_path,
        entry,
        subject_id="dan",
        smpl_meta=smpl_meta_a,
        beta_policy={"mode": "refined", "source_pose": "t-pose", "betas_sha256": "x"},
    )

    smpl_meta_b = dict(smpl_meta_a, gender="male")
    with pytest.raises(ValueError, match="smpl.gender"):
        update_manifest(
            manifest_path,
            entry,
            subject_id="dan",
            smpl_meta=smpl_meta_b,
            beta_policy={"mode": "refined", "source_pose": "t-pose", "betas_sha256": "x"},
        )


def test_manifest_rejects_faces_sha256_mismatch(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    smpl_meta_a = {
        "gender": "neutral",
        "num_betas": 10,
        "num_vertices": SMPL_NUM_VERTICES,
        "num_faces": SMPL_NUM_FACES,
        "faces_sha256": "aaa",
    }
    entry = PoseArtefact(
        pose_name="t-pose",
        directory="t-pose",
        oracle_only=False,
        betas_locked=False,
        has_displacements=True,
        has_pointcloud=True,
        quality=_dummy_quality(),
    )
    update_manifest(
        manifest_path,
        entry,
        subject_id="dan",
        smpl_meta=smpl_meta_a,
        beta_policy={"mode": "refined", "source_pose": "t-pose", "betas_sha256": "x"},
    )

    smpl_meta_b = dict(smpl_meta_a, faces_sha256="bbb")
    with pytest.raises(ValueError, match="faces_sha256"):
        update_manifest(
            manifest_path,
            entry,
            subject_id="dan",
            smpl_meta=smpl_meta_b,
            beta_policy={"mode": "refined", "source_pose": "t-pose", "betas_sha256": "x"},
        )


def test_oracle_flag(tmp_path):
    """AC21 (7.B8) — `--oracle-only` sets `poses[i].oracle_only == true`; the default
    is `false`; the flag survives a manifest update by a LATER run for a different
    pose (i.e. updating pose B does not clobber pose A's oracle_only)."""
    manifest_path = tmp_path / "manifest.json"
    smpl_meta = {
        "gender": "neutral",
        "num_betas": 10,
        "num_vertices": SMPL_NUM_VERTICES,
        "num_faces": SMPL_NUM_FACES,
        "faces_sha256": "aaa",
    }
    beta_policy = {"mode": "refined", "source_pose": "t-pose", "betas_sha256": "x"}

    default_entry = PoseArtefact(
        pose_name="t-pose",
        directory="t-pose",
        oracle_only=False,
        betas_locked=False,
        has_displacements=True,
        has_pointcloud=True,
        quality=_dummy_quality(),
    )
    update_manifest(
        manifest_path, default_entry, subject_id="dan", smpl_meta=smpl_meta, beta_policy=beta_policy
    )

    oracle_entry = PoseArtefact(
        pose_name="a-pose-heldout",
        directory="a-pose-heldout",
        oracle_only=True,
        betas_locked=True,
        has_displacements=True,
        has_pointcloud=True,
        quality=_dummy_quality(),
    )
    update_manifest(
        manifest_path, oracle_entry, subject_id="dan", smpl_meta=smpl_meta, beta_policy=beta_policy
    )

    import json

    manifest = json.loads(manifest_path.read_text())
    by_name = {p["pose_name"]: p for p in manifest["poses"]}
    assert by_name["t-pose"]["oracle_only"] is False
    assert by_name["a-pose-heldout"]["oracle_only"] is True

    # A THIRD run, for a different pose again, must not disturb either flag above.
    third_entry = PoseArtefact(
        pose_name="b-pose",
        directory="b-pose",
        oracle_only=False,
        betas_locked=True,
        has_displacements=True,
        has_pointcloud=True,
        quality=_dummy_quality(),
    )
    update_manifest(
        manifest_path, third_entry, subject_id="dan", smpl_meta=smpl_meta, beta_policy=beta_policy
    )
    manifest = json.loads(manifest_path.read_text())
    by_name = {p["pose_name"]: p for p in manifest["poses"]}
    assert by_name["t-pose"]["oracle_only"] is False
    assert by_name["a-pose-heldout"]["oracle_only"] is True  # survived the later update
    assert by_name["b-pose"]["oracle_only"] is False


# ---------------------------------------------------------------------------
# load_locked_betas — 7.B1
# ---------------------------------------------------------------------------


class TestLoadLockedBetas:
    def test_round_trips_from_written_smpl_params(self, tmp_path, dummy_topology):
        fit = _dummy_fit(seed=3)
        out_dir = tmp_path / "t-pose"
        write_pose_artefacts(
            out_dir, fit, _dummy_alignment(), _dummy_quality(), dummy_topology, pose_name="t-pose"
        )

        betas = load_locked_betas(out_dir / "smpl_params.npz")
        assert betas.shape == (10,)
        assert betas.dtype == np.float64
        assert np.allclose(betas, fit.betas.astype(np.float64))

    def test_missing_path_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_locked_betas(tmp_path / "nope.npz")

    def test_missing_betas_key_raises_value_error(self, tmp_path):
        path = tmp_path / "not_smpl_params.npz"
        np.savez(path, foo=np.zeros(3))
        with pytest.raises(ValueError, match="betas"):
            load_locked_betas(path)

    def test_wrong_shape_betas_raises_value_error(self, tmp_path):
        path = tmp_path / "bad.npz"
        np.savez(path, betas=np.zeros(5))
        with pytest.raises(ValueError, match=r"\(10,\)"):
            load_locked_betas(path)
