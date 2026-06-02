"""Smoke tests for oat.eval.trajectory_smoothness.

Run with: python test_trajectory_smoothness.py
"""
import math
import sys
import pathlib

ROOT = pathlib.Path(__file__).parent
sys.path.insert(0, str(ROOT))

import torch

from oat.eval.trajectory_smoothness import (
    SMOOTH_KEYS,
    _diff,
    _rotvec_geodesic_diff,
    acceleration_rms,
    compute_smoothness_metrics,
    jerk_rms,
    sparc,
    velocity_rms,
)


def test_constant_velocity_ramp():
    # x[t] = t in 16 steps, expanded over 3 dims, batch 2
    fs = 20.0
    dt = 1.0 / fs
    t = torch.linspace(0.0, 15.0, 16)
    traj = t[None, :, None].expand(2, 16, 3).contiguous()
    v = velocity_rms(traj, dt).item()
    a = acceleration_rms(traj, dt).item()
    j = jerk_rms(traj, dt).item()
    # vel: per-step delta is 1.0 along each of 3 dims -> norm = sqrt(3); /dt -> sqrt(3)*fs
    expected_v = math.sqrt(3.0) * fs
    assert abs(v - expected_v) < 1e-4, f"velocity_rms expected ~{expected_v}, got {v}"
    assert a < 1e-5, f"acceleration_rms expected ~0, got {a}"
    assert j < 1e-5, f"jerk_rms expected ~0, got {j}"
    print(f"[ok] constant-velocity ramp: v={v:.4f} a={a:.2e} j={j:.2e}")


def test_short_trajectory_no_crash():
    traj = torch.randn(2, 3, 7)
    out = compute_smoothness_metrics(traj, fs=20.0)
    for k in SMOOTH_KEYS:
        v = out[k]
        assert v.dim() == 0, f"{k} must be 0-d, got shape {v.shape}"
        assert torch.isfinite(v), f"{k} not finite: {v}"
    print(f"[ok] short trajectory T=3 returns 0-d finite tensors")


def test_zero_rotvec_geodesic_is_zero():
    omega = torch.zeros(4, 10, 3)
    d = _rotvec_geodesic_diff(omega)
    assert d.shape == (4, 9, 3)
    assert d.abs().max().item() < 1e-6
    print(f"[ok] zero rotvec geodesic diff is ~0")


def test_sparc_more_negative_with_noise():
    fs = 20.0
    dt = 1.0 / fs
    t = torch.arange(64).float() * dt
    clean = torch.sin(2 * math.pi * 2.0 * t)[None, :, None].expand(1, 64, 3).contiguous()
    torch.manual_seed(0)
    noisy = clean + 0.5 * torch.randn_like(clean)
    s_clean = sparc(clean, fs).item()
    s_noisy = sparc(noisy, fs).item()
    assert math.isfinite(s_clean) and math.isfinite(s_noisy), (s_clean, s_noisy)
    assert s_noisy < s_clean, f"noisy SPARC should be more negative: clean={s_clean} noisy={s_noisy}"
    print(f"[ok] SPARC: clean={s_clean:.3f}, noisy={s_noisy:.3f} (noisy more negative)")


def test_compute_smoothness_metrics_full_shape():
    fs = 20.0
    traj = torch.randn(4, 16, 7)  # [B, T, D] -- pos(3) + rot(3) + gripper(1)
    out = compute_smoothness_metrics(traj, fs=fs)
    assert set(out.keys()) == set(SMOOTH_KEYS)
    # Test geodesic mode also runs
    out2 = compute_smoothness_metrics(traj, fs=fs, rot_mode="geodesic")
    assert set(out2.keys()) == set(SMOOTH_KEYS)
    print(f"[ok] full keys present in both rot modes")


def test_diff_empty_on_short_input():
    x = torch.randn(2, 2, 3)
    d3 = _diff(x, 3)
    assert d3.shape == (2, 0, 3), d3.shape
    print(f"[ok] _diff returns empty tensor when T <= n")


if __name__ == "__main__":
    test_constant_velocity_ramp()
    test_short_trajectory_no_crash()
    test_zero_rotvec_geodesic_is_zero()
    test_sparc_more_negative_with_noise()
    test_compute_smoothness_metrics_full_shape()
    test_diff_empty_on_short_input()
    print("\nAll smoke tests passed.")
