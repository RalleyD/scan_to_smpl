"""SO(3) operations, rotation conversions, and Procrustes alignment."""

import numpy as np
from scipy.spatial.transform import Rotation


# ---------------------------------------------------------------------------
# Rotation conversions
# ---------------------------------------------------------------------------


def aa_to_rotmat(aa: np.ndarray) -> np.ndarray:
    """
    Axis-angle to rotation matrices.

    Args:
        aa: (..., 3) axis-angle vectors.

    Returns:
        (..., 3, 3) rotation matrices.
    """
    shape = aa.shape[:-1]
    flat = aa.reshape(-1, 3)
    mats = Rotation.from_rotvec(flat).as_matrix().astype(np.float64)
    return mats.reshape(*shape, 3, 3)


def rotmat_to_aa(R: np.ndarray) -> np.ndarray:
    """
    Rotation matrices to axis-angle.

    Args:
        R: (..., 3, 3) rotation matrices.

    Returns:
        (..., 3) axis-angle vectors.
    """
    shape = R.shape[:-2]
    flat = R.reshape(-1, 3, 3)
    aa = Rotation.from_matrix(flat).as_rotvec().astype(np.float64)
    return aa.reshape(*shape, 3)


# ---------------------------------------------------------------------------
# SO(3) logarithmic and exponential maps
# ---------------------------------------------------------------------------


def so3_log(R: np.ndarray) -> np.ndarray:
    """
    Logarithmic map SO(3) -> so(3).

    Args:
        R: (3, 3) rotation matrix.

    Returns:
        (3,) tangent vector (axis-angle representation of R).
    """
    return Rotation.from_matrix(R).as_rotvec().astype(np.float64)


def so3_exp(v: np.ndarray) -> np.ndarray:
    """
    Exponential map so(3) -> SO(3).

    Args:
        v: (3,) tangent vector (axis-angle).

    Returns:
        (3, 3) rotation matrix.
    """
    return Rotation.from_rotvec(v).as_matrix().astype(np.float64)


# ---------------------------------------------------------------------------
# SO(3) Frechet mean (Karcher / geometric mean)
# ---------------------------------------------------------------------------


def frechet_mean_so3(
    rotations: np.ndarray,
    weights: np.ndarray | None = None,
    max_iter: int = 50,
    tol: float = 1e-7,
) -> np.ndarray:
    """
    Weighted Frechet mean on SO(3) via iterative tangent-space averaging.

    Algorithm:
        1. Initialise R_mean = R_0 (or weighted medoid for robustness).
        2. Compute tangent vectors: v_i = Log(R_mean^T @ R_i).
        3. Weighted average: delta = sum(w_i * v_i) / sum(w_i).
        4. Update: R_mean = R_mean @ Exp(delta).
        5. Repeat until ||delta|| < tol or max_iter reached.

    Args:
        rotations: (N, 3, 3) rotation matrices.
        weights: (N,) non-negative weights. If None, uniform weights.
        max_iter: Maximum iterations.
        tol: Convergence tolerance on ||delta||.

    Returns:
        (3, 3) mean rotation matrix.

    Raises:
        ValueError: If fewer than 1 rotation is provided.
    """
    N = rotations.shape[0]
    if N == 0:
        raise ValueError("Need at least 1 rotation for Frechet mean")
    if N == 1:
        return rotations[0].copy()

    if weights is None:
        weights = np.ones(N, dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64)

    w_sum = weights.sum()
    if w_sum < 1e-12:
        raise ValueError("Weights sum to zero")

    # Initialise with the rotation closest to all others (medoid)
    # This gives better convergence than an arbitrary choice.
    R_mean = rotations[0].copy()

    for _ in range(max_iter):
        # Compute tangent vectors at R_mean
        tangents = np.zeros((N, 3), dtype=np.float64)
        for i in range(N):
            tangents[i] = so3_log(R_mean.T @ rotations[i])

        # Weighted average in tangent space
        delta = np.sum(weights[:, None] * tangents, axis=0) / w_sum

        # Update mean
        R_mean = R_mean @ so3_exp(delta)

        if np.linalg.norm(delta) < tol:
            break

    return R_mean


# ---------------------------------------------------------------------------
# Procrustes alignment
# ---------------------------------------------------------------------------


def procrustes_align(
    source: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, float]:
    """
    Procrustes alignment with scale: find optimal s, R, t to minimise
    ||target - (s * R @ source + t)||^2.

    Uses the Umeyama method (SVD-based).

    Args:
        source: (J, 3) point set to be aligned.
        target: (J, 3) reference point set.

    Returns:
        aligned: (J, 3) source after alignment.
        scale: the recovered scale factor.
    """
    assert source.shape == target.shape
    J = source.shape[0]

    # Centre both point sets
    mu_s = source.mean(axis=0)
    mu_t = target.mean(axis=0)
    src_c = source - mu_s
    tgt_c = target - mu_t

    # Variance of source
    var_s = np.sum(src_c ** 2) / J

    # Cross-covariance
    cov = (tgt_c.T @ src_c) / J  # (3, 3)

    U, S, Vt = np.linalg.svd(cov)

    # Correct reflection
    d = np.linalg.det(U) * np.linalg.det(Vt)
    D = np.diag([1.0, 1.0, np.sign(d)])

    R = U @ D @ Vt
    scale = float(np.sum(S * np.diag(D)) / var_s) if var_s > 1e-12 else 1.0
    t = mu_t - scale * (R @ mu_s)

    aligned = scale * (source @ R.T) + t
    return aligned, scale


def compute_pa_mpjpe(
    predicted: np.ndarray,
    target: np.ndarray,
) -> float:
    """
    Procrustes-Aligned Mean Per-Joint Position Error.

    Args:
        predicted: (J, 3) predicted joint positions.
        target: (J, 3) ground-truth / reference joint positions.

    Returns:
        PA-MPJPE in the same units as the input (typically mm or m).
    """
    aligned, _ = procrustes_align(predicted, target)
    errors = np.linalg.norm(aligned - target, axis=1)  # (J,)
    return float(errors.mean())
