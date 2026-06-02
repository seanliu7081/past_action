"""Trajectory smoothness metrics for action chunks and rollouts.

Computes velocity-, acceleration-, jerk-RMS, and SPARC (spectral arc length)
separately for the position (dims pos_slice) and rotation (dims rot_slice)
components of an action vector. Gripper / extra dims are excluded.

Rotation handling: by default dims rot_slice are treated as a 3-vector and
plain finite differences are taken in rotvec-space. This is correct when the
inputs are *delta* rotations (already in tangent space) -- the case for LIBERO
OSC_POSE actions used by this repo. Pass `rot_mode="geodesic"` to instead
compose consecutive rotations on SO(3) and log-map the relative rotation
before differencing; useful if inputs are absolute orientations.

All scalar outputs are returned as 0-d torch tensors so that the values can be
fed straight into `accelerate.Accelerator.reduce` for distributed averaging.
"""
from __future__ import annotations

import math
from typing import Dict, Tuple

import torch

from oat.tokenizer.oat.augment.so3_action_chunk_aug import so3_exp_map, so3_log_map


SMOOTH_KEYS = [
    "vel_pos", "acc_pos", "jerk_pos", "sparc_pos",
    "vel_rot", "acc_rot", "jerk_rot", "sparc_rot",
]


def _diff(x: torch.Tensor, n: int) -> torch.Tensor:
    """n-th finite difference along the time axis (dim=1)."""
    if x.shape[1] <= n:
        return x.new_zeros(x.shape[0], 0, x.shape[2])
    out = x
    for _ in range(n):
        out = out[:, 1:] - out[:, :-1]
    return out


def _norm_rms(x: torch.Tensor) -> torch.Tensor:
    """Mean over (B, T) of the per-step Euclidean norm. Returns 0-d tensor."""
    if x.numel() == 0:
        return x.new_tensor(0.0)
    return x.norm(dim=-1).mean()


def velocity_rms(traj: torch.Tensor, dt: float) -> torch.Tensor:
    return _norm_rms(_diff(traj, 1) / dt)


def acceleration_rms(traj: torch.Tensor, dt: float) -> torch.Tensor:
    return _norm_rms(_diff(traj, 2) / (dt * dt))


def jerk_rms(traj: torch.Tensor, dt: float) -> torch.Tensor:
    return _norm_rms(_diff(traj, 3) / (dt ** 3))


def sparc(
    traj: torch.Tensor,
    fs: float,
    padlevel: int = 4,
    fc: float = 10.0,
    amp_th: float = 0.05,
) -> torch.Tensor:
    """Balasubramanian Spectral Arc Length (SPARC).

    More-negative values indicate jerkier / less smooth motion.
    Returns a 0-d tensor: the per-batch mean SAL.
    """
    B, T, D = traj.shape
    if T < 4:
        return traj.new_tensor(0.0)

    dt = 1.0 / float(fs)
    v = (_diff(traj, 1) / dt).norm(dim=-1)            # [B, T-1]
    if v.shape[1] < 2:
        return traj.new_tensor(0.0)

    n = int(2 ** (math.ceil(math.log2(v.shape[1])) + padlevel))
    spec = torch.fft.rfft(v, n=n, dim=-1).abs()       # [B, n//2+1]
    freqs = torch.fft.rfftfreq(n, d=dt).to(spec.device, spec.dtype)

    eps = torch.finfo(spec.dtype).eps
    spec_norm = spec / spec.amax(dim=-1, keepdim=True).clamp_min(eps)

    sals = []
    for b in range(B):
        sn = spec_norm[b]
        cut = freqs <= fc
        sn_c = sn[cut]
        fr_c = freqs[cut]
        if sn_c.numel() < 2:
            sals.append(traj.new_tensor(0.0))
            continue
        keep_idx = torch.nonzero(sn_c >= amp_th, as_tuple=False).flatten()
        if keep_idx.numel() < 2:
            sals.append(traj.new_tensor(0.0))
            continue
        last = int(keep_idx[-1].item()) + 1
        sn_u = sn_c[:last]
        fr_u = fr_c[:last]
        df = (fr_u[1:] - fr_u[:-1]) / fc
        dv = sn_u[1:] - sn_u[:-1]
        sal = -(df * df + dv * dv).sqrt().sum()
        sals.append(sal)
    return torch.stack(sals).mean()


def _rotvec_geodesic_diff(omega: torch.Tensor) -> torch.Tensor:
    """omega: [B, T, 3] axis-angle. Returns [B, T-1, 3] = log(R_{t+1} R_t^T)."""
    if omega.shape[1] < 2:
        return omega.new_zeros(omega.shape[0], 0, 3)
    R = so3_exp_map(omega)                                # [B, T, 3, 3]
    rel = torch.matmul(R[:, 1:], R[:, :-1].transpose(-1, -2))
    return so3_log_map(rel)


def _component_metrics(
    traj: torch.Tensor, fs: float, prefix: str
) -> Dict[str, torch.Tensor]:
    dt = 1.0 / float(fs)
    return {
        f"vel_{prefix}":  velocity_rms(traj, dt),
        f"acc_{prefix}":  acceleration_rms(traj, dt),
        f"jerk_{prefix}": jerk_rms(traj, dt),
        f"sparc_{prefix}": sparc(traj, fs),
    }


def compute_smoothness_metrics(
    actions: torch.Tensor,
    *,
    pos_slice: Tuple[int, int] = (0, 3),
    rot_slice: Tuple[int, int] = (3, 6),
    fs: float = 20.0,
    rot_mode: str = "euclidean",
) -> Dict[str, torch.Tensor]:
    """Compute the 8 smoothness metrics in SMOOTH_KEYS.

    Args:
        actions: [B, T, D] tensor in physical units (post-unnormalize).
        pos_slice: (start, end) for the position dims (Euclidean).
        rot_slice: (start, end) for the rotation dims (length-3 axis-angle).
        fs: control / sampling frequency in Hz. Used to scale finite differences.
        rot_mode: "euclidean" (default) treats rot dims as a 3-vector and uses
            plain finite differences (correct for delta-action data). "geodesic"
            composes consecutive rotations on SO(3) and log-maps the relative
            rotation before measuring velocity (then plain finite diffs of that
            angular-velocity sequence for acc/jerk).

    Returns:
        dict with the 8 keys in SMOOTH_KEYS, each a 0-d tensor on the same
        device/dtype as `actions`.
    """
    if actions.dim() != 3:
        raise ValueError(f"Expected [B, T, D], got {tuple(actions.shape)}")
    if rot_mode not in ("euclidean", "geodesic"):
        raise ValueError(f"rot_mode must be 'euclidean' or 'geodesic', got {rot_mode}")

    pos = actions[..., pos_slice[0]:pos_slice[1]].float()
    metrics = _component_metrics(pos, fs, prefix="pos")

    rot = actions[..., rot_slice[0]:rot_slice[1]].float()
    if rot_mode == "euclidean":
        metrics.update(_component_metrics(rot, fs, prefix="rot"))
    else:
        dt = 1.0 / float(fs)
        d_rot = _rotvec_geodesic_diff(rot)                # [B, T-1, 3] = angular displacement
        ang_vel = d_rot / dt                               # treat as the velocity series
        metrics[f"vel_rot"]  = _norm_rms(ang_vel)
        metrics[f"acc_rot"]  = _norm_rms(_diff(ang_vel, 1) / dt)
        metrics[f"jerk_rot"] = _norm_rms(_diff(ang_vel, 2) / (dt * dt))
        metrics[f"sparc_rot"] = sparc(d_rot, fs)

    return {k: metrics[k] for k in SMOOTH_KEYS}
