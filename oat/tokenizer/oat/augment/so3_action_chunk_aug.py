import math

import torch
from torch import nn


def _compute_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return dtype


def hat(v: torch.Tensor) -> torch.Tensor:
    """Return the skew-symmetric matrix for vectors shaped [..., 3]."""
    if v.shape[-1] != 3:
        raise ValueError(f"Expected last dimension to be 3, got {v.shape[-1]}")

    h = torch.zeros(*v.shape[:-1], 3, 3, device=v.device, dtype=v.dtype)
    h[..., 0, 1] = -v[..., 2]
    h[..., 0, 2] = v[..., 1]
    h[..., 1, 0] = v[..., 2]
    h[..., 1, 2] = -v[..., 0]
    h[..., 2, 0] = -v[..., 1]
    h[..., 2, 1] = v[..., 0]
    return h


def so3_exp_map(rotvec: torch.Tensor) -> torch.Tensor:
    """Convert rotation vectors shaped [..., 3] to rotation matrices."""
    if rotvec.shape[-1] != 3:
        raise ValueError(f"Expected last dimension to be 3, got {rotvec.shape[-1]}")
    if not torch.is_floating_point(rotvec):
        raise TypeError("so3_exp_map expects a floating-point tensor")

    out_dtype = rotvec.dtype
    work = rotvec.to(dtype=_compute_dtype(rotvec.dtype))
    theta2 = (work * work).sum(dim=-1, keepdim=True)
    theta = torch.sqrt(theta2.clamp_min(0.0))
    small = theta2 < 1.0e-8

    theta4 = theta2 * theta2
    sin_over_theta_small = 1.0 - theta2 / 6.0 + theta4 / 120.0
    one_minus_cos_over_theta2_small = 0.5 - theta2 / 24.0 + theta4 / 720.0

    eps = torch.finfo(work.dtype).eps
    sin_over_theta = torch.where(
        small,
        sin_over_theta_small,
        torch.sin(theta) / theta.clamp_min(eps),
    )
    one_minus_cos_over_theta2 = torch.where(
        small,
        one_minus_cos_over_theta2_small,
        (1.0 - torch.cos(theta)) / theta2.clamp_min(eps),
    )

    k = hat(work)
    k2 = torch.matmul(k, k)
    eye = torch.eye(3, device=work.device, dtype=work.dtype)
    eye = eye.expand(*work.shape[:-1], 3, 3)

    r = (
        eye
        + sin_over_theta.unsqueeze(-1) * k
        + one_minus_cos_over_theta2.unsqueeze(-1) * k2
    )
    return r.to(dtype=out_dtype)


def so3_log_map(R: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices shaped [..., 3, 3] to rotation vectors."""
    if R.shape[-2:] != (3, 3):
        raise ValueError(f"Expected trailing shape [3, 3], got {R.shape[-2:]}")
    if not torch.is_floating_point(R):
        raise TypeError("so3_log_map expects a floating-point tensor")

    out_dtype = R.dtype
    work = R.to(dtype=_compute_dtype(R.dtype))
    trace = work.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    cos_theta = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    theta = torch.acos(cos_theta)
    theta2 = theta * theta
    small = theta2 < 1.0e-8

    skew_vec = torch.stack(
        (
            work[..., 2, 1] - work[..., 1, 2],
            work[..., 0, 2] - work[..., 2, 0],
            work[..., 1, 0] - work[..., 0, 1],
        ),
        dim=-1,
    )

    theta4 = theta2 * theta2
    factor_small = 0.5 + theta2 / 12.0 + 7.0 * theta4 / 720.0

    eps = torch.finfo(work.dtype).eps
    sin_theta = torch.sin(theta)
    denom = (2.0 * sin_theta).clamp_min(eps)
    factor = torch.where(small, factor_small, theta / denom)
    rotvec = factor.unsqueeze(-1) * skew_vec
    return rotvec.to(dtype=out_dtype)


def sample_random_rotvec(
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    max_angle_rad: float,
) -> torch.Tensor:
    """Sample random rotation vectors with angle uniformly in [0, max_angle_rad]."""
    work_dtype = _compute_dtype(dtype)
    rotvec = torch.zeros(batch_size, 3, device=device, dtype=work_dtype)
    if batch_size == 0 or max_angle_rad <= 0.0:
        return rotvec.to(dtype=dtype)

    direction = torch.randn(batch_size, 3, device=device, dtype=work_dtype)
    direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(
        torch.finfo(work_dtype).eps
    )
    angle = torch.rand(batch_size, 1, device=device, dtype=work_dtype)
    rotvec = direction * (angle * float(max_angle_rad))
    return rotvec.to(dtype=dtype)


class SO3ActionChunkAug(nn.Module):
    """Apply one random SO(3) perturbation per action chunk during training."""

    _VALID_MODES = ("left_noise", "right_noise", "conjugate")

    def __init__(
        self,
        p: float = 0.3,
        max_angle_deg: float = 5.0,
        mode: str = "left_noise",
        augment_position: bool = False,
        pos_start: int = 0,
        pos_end: int = 3,
        rot_start: int = 3,
        rot_end: int = 6,
    ):
        super().__init__()
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1], got {p}")
        if mode not in self._VALID_MODES:
            raise ValueError(f"mode must be one of {self._VALID_MODES}, got {mode}")
        if rot_end - rot_start != 3:
            raise ValueError("rot_end - rot_start must be 3")
        if augment_position and pos_end - pos_start != 3:
            raise ValueError("pos_end - pos_start must be 3 when augment_position=True")
        if min(pos_start, pos_end, rot_start, rot_end) < 0:
            raise ValueError("slice indices must be non-negative")
        if pos_start > pos_end or rot_start > rot_end:
            raise ValueError("slice starts must be <= slice ends")

        self.p = float(p)
        self.max_angle_deg = float(max_angle_deg)
        self.max_angle_rad = math.radians(float(max_angle_deg))
        self.mode = mode
        self.augment_position = bool(augment_position)
        self.pos_start = int(pos_start)
        self.pos_end = int(pos_end)
        self.rot_start = int(rot_start)
        self.rot_end = int(rot_end)

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        # actions: [B, T, D]
        if actions.ndim != 3:
            raise ValueError(f"Expected actions with rank 3 [B, T, D], got {actions.shape}")
        if not torch.is_floating_point(actions):
            raise TypeError("SO3ActionChunkAug expects floating-point actions")

        batch_size, _, action_dim = actions.shape
        if self.rot_end > action_dim:
            raise ValueError(
                f"Rotation slice [{self.rot_start}:{self.rot_end}] exceeds action dim {action_dim}"
            )
        if self.augment_position and self.pos_end > action_dim:
            raise ValueError(
                f"Position slice [{self.pos_start}:{self.pos_end}] exceeds action dim {action_dim}"
            )
        if not self.training or self.p <= 0.0 or self.max_angle_rad <= 0.0:
            return actions

        mask = torch.rand(batch_size, device=actions.device) < self.p
        if not mask.any():
            return actions

        out = actions.clone()
        work_dtype = _compute_dtype(actions.dtype)

        with torch.autocast(device_type=actions.device.type, enabled=False):
            rotvec = actions[..., self.rot_start:self.rot_end].to(dtype=work_dtype)
            R = so3_exp_map(rotvec)

            epsilon = sample_random_rotvec(
                batch_size=batch_size,
                device=actions.device,
                dtype=work_dtype,
                max_angle_rad=self.max_angle_rad,
            )
            Q = so3_exp_map(epsilon)  # [B, 3, 3]
            Q_expanded = Q[:, None, :, :]  # [B, 1, 3, 3]

            if self.mode == "left_noise":
                R_aug = torch.matmul(Q_expanded, R)
            elif self.mode == "right_noise":
                R_aug = torch.matmul(R, Q_expanded)
            elif self.mode == "conjugate":
                R_aug = torch.matmul(torch.matmul(Q_expanded, R), Q_expanded.transpose(-1, -2))
            else:
                raise ValueError(f"Unsupported mode {self.mode}")

            omega_aug = so3_log_map(R_aug).to(dtype=actions.dtype)
            chunk_mask = mask[:, None, None]
            out[..., self.rot_start:self.rot_end] = torch.where(
                chunk_mask,
                omega_aug,
                actions[..., self.rot_start:self.rot_end],
            )

            if self.augment_position:
                pos = actions[..., self.pos_start:self.pos_end].to(dtype=work_dtype)
                pos_aug = torch.matmul(Q_expanded, pos.unsqueeze(-1)).squeeze(-1)
                out[..., self.pos_start:self.pos_end] = torch.where(
                    chunk_mask,
                    pos_aug.to(dtype=actions.dtype),
                    actions[..., self.pos_start:self.pos_end],
                )

        return out
