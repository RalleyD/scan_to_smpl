#!/usr/bin/env python
"""Print SMPL lbs_weights.max(axis=1) — the dominant joint weight per vertex."""

import argparse

import numpy as np
import smplx

# SMPL-24 joint order (matches scantosmpl/smpl/joint_map.py docstring).
SMPL_JOINT_NAMES = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hand",
    "right_hand",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        default="models/smpl",
        help="Path to SMPL model dir or .pkl file (default: models/smpl)",
    )
    parser.add_argument("--gender", default="neutral", choices=["neutral", "male", "female"])
    args = parser.parse_args()

    model = smplx.SMPL(model_path=args.model_path, gender=args.gender)
    weights = model.lbs_weights.detach().cpu().numpy()  # [V, J]

    max_weights = weights.max(axis=1)
    max_joints = weights.argmax(axis=1)

    np.set_printoptions(precision=4, suppress=True, threshold=50, edgeitems=10)
    print(f"lbs_weights shape: {weights.shape} (V={weights.shape[0]}, J={weights.shape[1]})")
    print(f"max(axis=1) shape: {max_weights.shape}")
    print(max_weights)
    print(
        f"\nmin={max_weights.min():.4f}  max={max_weights.max():.4f}  "
        f"mean={max_weights.mean():.4f}  median={np.median(max_weights):.4f}"
    )

    top10 = np.argsort(max_weights)[::-1][:10]
    print("\nTop 10 vertices by max weight:")
    print(f"{'vertex':>8}  {'weight':>8}  {'joint#':>6}  joint_name")
    for vidx in top10:
        jidx = max_joints[vidx]
        jname = SMPL_JOINT_NAMES[jidx] if jidx < len(SMPL_JOINT_NAMES) else "?"
        print(f"{vidx:>8}  {max_weights[vidx]:>8.4f}  {jidx:>6}  {jname}")


if __name__ == "__main__":
    main()
