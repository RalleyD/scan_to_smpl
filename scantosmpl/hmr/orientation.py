"""Orientation quality checker for HMR outputs."""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class OrientationQuality:
    """Quality assessment of the predicted global orientation."""

    score: float                  # Overall quality score in [0, 1]
    is_upright: bool              # Person appears upright (nose above hips)
    rotation_magnitude_ok: bool   # ||global_orient|| < 2π
    warnings: list[str] = field(default_factory=list)


# COCO-17 keypoint indices used for upright check
_NOSE = 0
_LEFT_HIP = 11
_RIGHT_HIP = 12
_LEFT_SHOULDER = 5
_RIGHT_SHOULDER = 6
_LEFT_ELBOW = 7
_RIGHT_ELBOW = 8


def check_orientation_quality(
    global_orient: np.ndarray,
    keypoints_2d: np.ndarray,
    keypoint_confs: np.ndarray,
    image_hw: tuple[int, int],
) -> OrientationQuality:
    """
    Assess orientation quality using predicted axis-angle and 2D keypoints.

    Args:
        global_orient: (3,) global orientation in axis-angle.
        keypoints_2d:  (17, 2) COCO keypoint pixel coordinates (x, y).
        keypoint_confs: (17,) confidence scores.
        image_hw: (height, width) of the source image.

    Returns:
        OrientationQuality with score, flags, and warning strings.
    """
    warnings: list[str] = []
    checks_passed = 0
    total_checks = 0

    # --- Check 1: rotation magnitude < 2π ---
    rot_magnitude = float(np.linalg.norm(global_orient))
    rotation_magnitude_ok = rot_magnitude < 2.0 * np.pi
    total_checks += 1
    if rotation_magnitude_ok:
        checks_passed += 1
    else:
        warnings.append(
            f"Global orientation magnitude {rot_magnitude:.2f} rad exceeds 2π — "
            "possible gimbal lock or erroneous prediction"
        )

    # --- Check 2: upright (nose above hips in image space) ---
    # In image coords: smaller y = higher in image = higher on person
    is_upright = False
    nose_conf = keypoint_confs[_NOSE]
    lhip_conf = keypoint_confs[_LEFT_HIP]
    rhip_conf = keypoint_confs[_RIGHT_HIP]
    total_checks += 1

    if nose_conf > 0.3 and (lhip_conf > 0.3 or rhip_conf > 0.3):
        nose_y = float(keypoints_2d[_NOSE, 1])

        hip_ys = []
        if lhip_conf > 0.3:
            hip_ys.append(float(keypoints_2d[_LEFT_HIP, 1]))
        if rhip_conf > 0.3:
            hip_ys.append(float(keypoints_2d[_RIGHT_HIP, 1]))

        hip_y_mean = float(np.mean(hip_ys))
        # In image coords, nose_y < hip_y_mean means nose is above hips
        if nose_y < hip_y_mean:
            is_upright = True
            checks_passed += 1
        else:
            warnings.append(
                f"Person appears inverted: nose y={nose_y:.1f} is below hips y={hip_y_mean:.1f}"
            )
    else:
        # Insufficient keypoint confidence — skip upright check, don't penalise
        is_upright = True  # benefit of the doubt
        checks_passed += 1
        warnings.append("Upright check skipped: nose or hip keypoints below confidence threshold")

    # --- Check 3: T-pose elbow height (arms roughly horizontal) ---
    # For T-pose scans: elbows should be near shoulder height (within 20% of image height)
    lshoulder_conf = keypoint_confs[_LEFT_SHOULDER]
    rshoulder_conf = keypoint_confs[_RIGHT_SHOULDER]
    lelbow_conf = keypoint_confs[_LEFT_ELBOW]
    relbow_conf = keypoint_confs[_RIGHT_ELBOW]
    image_h = float(image_hw[0])

    t_pose_arm_check = True
    total_checks += 1

    arm_pairs = []
    if lshoulder_conf > 0.3 and lelbow_conf > 0.3:
        arm_pairs.append(
            (float(keypoints_2d[_LEFT_SHOULDER, 1]), float(keypoints_2d[_LEFT_ELBOW, 1]))
        )
    if rshoulder_conf > 0.3 and relbow_conf > 0.3:
        arm_pairs.append(
            (float(keypoints_2d[_RIGHT_SHOULDER, 1]), float(keypoints_2d[_RIGHT_ELBOW, 1]))
        )

    if arm_pairs:
        threshold = 0.20 * image_h  # 20% of image height
        for shoulder_y, elbow_y in arm_pairs:
            if abs(elbow_y - shoulder_y) > threshold:
                t_pose_arm_check = False
                warnings.append(
                    f"Arm may not be in T-pose: shoulder y={shoulder_y:.1f}, "
                    f"elbow y={elbow_y:.1f}, diff={abs(elbow_y - shoulder_y):.1f}px "
                    f"(threshold={threshold:.1f}px)"
                )
                break
    # If no arm keypoints, skip silently
    if t_pose_arm_check:
        checks_passed += 1

    score = float(checks_passed) / float(total_checks) if total_checks > 0 else 0.0

    return OrientationQuality(
        score=score,
        is_upright=is_upright,
        rotation_magnitude_ok=rotation_magnitude_ok,
        warnings=warnings,
    )
