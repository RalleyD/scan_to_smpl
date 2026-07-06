"""Unit tests for the extracted rear-view classifier (scantosmpl.fitting.rear_views).

No GPU or model checkpoints required.
"""

import json

import numpy as np
import pytest

from scantosmpl.fitting.rear_views import classify_rear_views
from scantosmpl.hmr.consensus import ConsensusResult


def _make_consensus(joints: np.ndarray) -> ConsensusResult:
    """Build a minimal ConsensusResult carrying only the joints classify_rear_views reads.

    The other fields are unused by classify_rear_views; dummy values keep the
    dataclass construction valid.
    """
    return ConsensusResult(
        betas=np.zeros(10),
        body_pose=np.zeros(69),
        global_orient=np.zeros(3),
        vertices=np.zeros((6890, 3)),
        joints=joints,
        faces=np.zeros((13776, 3), dtype=np.int64),
        pa_mpjpe_per_view={},
        pa_mpjpe_mean=0.0,
        beta_std=np.zeros(10),
        body_height_m=1.7,
        per_view_weights={},
        n_views_used=0,
    )


class TestDegenerateConsensus:
    def test_degenerate_consensus_returns_empty(self):
        # Pelvis, neck, and both shoulders coincide -> up_vec and shoulder_vec are
        # both zero -> cross product is zero -> norm < 1e-6 guard fires.
        joints = np.zeros((24, 3))
        consensus = _make_consensus(joints)

        cameras = {
            "cam_a.JPG": (np.eye(3), np.array([0.0, 0.0, -3.0]), np.eye(3)),
            "cam_b.JPG": (np.eye(3), np.array([0.0, 0.0, 3.0]), np.eye(3)),
        }

        assert classify_rear_views(consensus, cameras) == set()


class TestSyntheticFrontAndBackCameras:
    def _canonical_consensus(self) -> ConsensusResult:
        joints = np.zeros((24, 3))
        joints[0] = [0.0, 0.0, 0.0]  # pelvis
        joints[12] = [0.0, 0.5, 0.0]  # neck
        joints[16] = [0.2, 0.4, 0.0]  # left_shoulder (+X, per SMPL convention)
        joints[17] = [-0.2, 0.4, 0.0]  # right_shoulder
        return _make_consensus(joints)

    def test_synthetic_front_and_back_cameras(self):
        consensus = self._canonical_consensus()

        # up_vec = [0, 0.5, 0], shoulder_vec = left - right = [0.4, 0, 0]
        # cross(up_vec, shoulder_vec) = [0, 0, -0.2] -> back-vector points -Z,
        # matching docs/phase5_spec_supplement.md #A1 (subject's back faces -Z).
        # Camera rotation is irrelevant to classify_rear_views (only camera
        # centre = -R.T @ t matters), so identity R is used with t = -centre.
        front_centre = np.array([0.0, 0.0, 3.0])
        back_centre = np.array([0.0, 0.0, -3.0])
        cameras = {
            "front_cam.JPG": (np.eye(3), -front_centre, np.eye(3)),
            "back_cam.JPG": (np.eye(3), -back_centre, np.eye(3)),
        }

        rear = classify_rear_views(consensus, cameras)

        assert "back_cam.JPG" in rear
        assert "front_cam.JPG" not in rear


# ---------------------------------------------------------------------------
# Integration-log regression fixture.
#
# Source: this data was dumped from a real Phase 5 integration run —
# output/debug/refinement/refinement_results.json["cameras"] for the 17
# SMPL-frame (R, t) camera pairs, plus consensus.joints[PELVIS/NECK/
# LEFT_SHOULDER/RIGHT_SHOULDER] recomputed via SMPLModel.forward() on the
# betas/body_pose/global_orient recorded in
# output/debug/consensus/consensus_results.json (that file does not persist
# `joints` directly, only the params needed to regenerate them). Values are
# rounded to 6 decimal places; classification was re-verified against the
# rounded fixture to confirm no sign flips versus the full-precision source.
# ---------------------------------------------------------------------------
_INTEGRATION_FIXTURE_JSON = """
{
    "joints": {
        "pelvis": [-0.002011, -0.214344, 0.01798],
        "neck": [0.002018, 0.285953, -0.018149],
        "left_shoulder": [0.15577, 0.224795, -0.022427],
        "right_shoulder": [-0.157654, 0.228894, -0.026619]
    },
    "cameras": {
        "cam01_2.JPG": {
            "R": [[-0.004378, -0.996294, 0.085901],
                  [-0.931024, -0.027287, -0.363937],
                  [0.364932, -0.081569, -0.927454]],
            "t": [-0.084532, -0.087114, 2.081896]
        },
        "cam01_6.JPG": {
            "R": [[0.312962, -0.914988, -0.25466],
                  [-0.680553, -0.029011, -0.732124],
                  [0.662497, 0.402437, -0.631777]],
            "t": [-0.480613, 0.084084, 1.863793]
        },
        "cam02_4.JPG": {
            "R": [[0.798309, 0.601987, 0.017719],
                  [0.033389, -0.073616, 0.996728],
                  [0.601322, -0.795105, -0.078867]],
            "t": [0.224925, 0.02893, 1.660082]
        },
        "cam02_5.JPG": {
            "R": [[0.136904, 0.989909, 0.036579],
                  [-0.037544, -0.031715, 0.998792],
                  [0.989873, -0.138112, 0.032824]],
            "t": [0.257001, -0.003079, 1.678515]
        },
        "cam03_5.JPG": {
            "R": [[-0.838236, 0.061959, 0.541776],
                  [-0.089745, -0.995651, -0.024988],
                  [0.537872, -0.069568, 0.840151]],
            "t": [-0.265303, -0.068794, 1.724394]
        },
        "cam03_6.JPG": {
            "R": [[0.232391, -0.929694, 0.285768],
                  [0.727598, -0.028803, -0.685399],
                  [0.645442, 0.367205, 0.66975]],
            "t": [-0.517849, -0.048062, 1.861727]
        },
        "cam04_4.JPG": {
            "R": [[0.012379, -0.662409, -0.74904],
                  [0.99969, 0.024387, -0.005044],
                  [0.021608, -0.748745, 0.662505]],
            "t": [-0.23359, -0.050816, 1.999702]
        },
        "cam04_5.JPG": {
            "R": [[-0.998384, -0.009592, -0.056014],
                  [0.01257, -0.998512, -0.053064],
                  [-0.055422, -0.053683, 0.997019]],
            "t": [0.07429, -0.062351, 1.770394]
        },
        "cam05_4.JPG": {
            "R": [[0.407139, -0.817209, -0.407931],
                  [0.613409, -0.086268, 0.78504],
                  [-0.676733, -0.569849, 0.46616]],
            "t": [-0.049932, 0.141375, 1.974262]
        },
        "cam05_5.JPG": {
            "R": [[-0.749413, -0.042722, -0.660723],
                  [0.003487, -0.998157, 0.060585],
                  [-0.662094, 0.043099, 0.748181]],
            "t": [0.240154, 0.148836, 1.790133]
        },
        "cam05_6.JPG": {
            "R": [[-0.482559, -0.825122, 0.293786],
                  [0.619282, -0.08423, 0.780637],
                  [-0.619376, 0.558641, 0.551629]],
            "t": [-0.155756, 0.051623, 1.910927]
        },
        "cam06_4.JPG": {
            "R": [[-0.697249, 0.716796, 0.006846],
                  [0.023634, 0.032534, -0.999191],
                  [-0.716439, -0.696523, -0.039625]],
            "t": [0.061209, 0.041274, 1.686478]
        },
        "cam07_4.JPG": {
            "R": [[-0.321647, 0.857623, -0.40128],
                  [0.832951, 0.054755, -0.550631],
                  [-0.450262, -0.511356, -0.731969]],
            "t": [-0.008263, -0.268724, 2.053452]
        },
        "cam07_6.JPG": {
            "R": [[-0.278442, -0.910666, -0.305216],
                  [-0.712554, -0.017213, 0.701406],
                  [-0.644001, 0.412784, -0.644106]],
            "t": [-0.465308, -0.242532, 1.933868]
        },
        "cam10_2.JPG": {
            "R": [[-0.90825, 0.014679, -0.418171],
                  [0.307631, 0.700853, -0.643559],
                  [0.283629, -0.713154, -0.641066]],
            "t": [-0.034332, -0.072642, 2.03107]
        },
        "cam10_4.JPG": {
            "R": [[0.97832, 0.142949, -0.149854],
                  [0.039065, 0.583217, 0.811377],
                  [0.203383, -0.79964, 0.564988]],
            "t": [-0.003288, -0.025142, 1.884246]
        },
        "cam10_5.JPG": {
            "R": [[0.956855, -0.184313, 0.224628],
                  [-0.077536, 0.583074, 0.808711],
                  [-0.28003, -0.791236, 0.543626]],
            "t": [-0.096874, 0.014714, 1.973291]
        }
    }
}
"""

_EXPECTED_REAR_VIEWS = {
    "cam02_5.JPG",
    "cam03_5.JPG",
    "cam03_6.JPG",
    "cam04_4.JPG",
    "cam04_5.JPG",
    "cam05_4.JPG",
    "cam05_5.JPG",
    "cam05_6.JPG",
    "cam10_4.JPG",
    "cam10_5.JPG",
}


class TestMatchesIntegrationRearSet:
    def test_matches_integration_rear_set(self):
        fixture = json.loads(_INTEGRATION_FIXTURE_JSON)

        joints = np.zeros((24, 3))
        joints[0] = fixture["joints"]["pelvis"]
        joints[12] = fixture["joints"]["neck"]
        joints[16] = fixture["joints"]["left_shoulder"]
        joints[17] = fixture["joints"]["right_shoulder"]
        consensus = _make_consensus(joints)

        cameras = {}
        for name, cam in fixture["cameras"].items():
            R = np.array(cam["R"])
            t = np.array(cam["t"])
            cameras[name] = (R, t, np.eye(3))

        rear_views = classify_rear_views(consensus, cameras)

        assert rear_views == _EXPECTED_REAR_VIEWS


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
