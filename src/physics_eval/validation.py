"""Validation and qpos normalization helpers."""

from __future__ import annotations

import numpy as np

QPOS_DIM = 36
ROOT_QUAT_SLICE = slice(3, 7)


def validate_qpos_sequence(
    qpos_seq: np.ndarray,
    *,
    name: str = "qpos_seq",
    min_frames: int = 1,
) -> np.ndarray:
    """
    Return a float64 copy of a valid MuJoCo qpos sequence.

    The original array is never modified. Root quaternions are normalized on the
    returned copy so downstream velocity/inverse-dynamics computations do not
    inherit small generator drift.
    """
    arr = np.asarray(qpos_seq)
    if arr.ndim != 2:
        raise ValueError(f"{name}: expected a 2D array of shape (T, {QPOS_DIM}), got {arr.shape}")
    if arr.shape[1] != QPOS_DIM:
        raise ValueError(f"{name}: expected shape (T, {QPOS_DIM}), got {arr.shape}")
    if arr.shape[0] < min_frames:
        raise ValueError(f"{name}: expected at least {min_frames} frames, got {arr.shape[0]}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name}: qpos contains NaN or Inf values")

    out = arr.astype(np.float64, copy=True)
    quat = out[:, ROOT_QUAT_SLICE]
    norms = np.linalg.norm(quat, axis=1)
    if np.any(norms < 1e-8):
        bad = int(np.where(norms < 1e-8)[0][0])
        raise ValueError(f"{name}: root quaternion at frame {bad} has near-zero norm")
    out[:, ROOT_QUAT_SLICE] = quat / norms[:, None]
    return out
