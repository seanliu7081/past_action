#!/usr/bin/env python3
"""Measure the effective maximum frequency ``f_max`` of LIBERO-10 demonstration
delta-actions via a Welch power-spectral-density (PSD) estimate.

Motivation
----------
The dynamics analysis bounds the Taylor remainder of the delta-action sequence
using Bernstein's inequality for band-limited signals::

    |a^(n)(t)| <= (2*pi*f_max)^n * ||a||_inf
    R_K        = (2*pi*f_max*dt)^K / K! * ||a||_inf

Every downstream claim (super-exponential contraction of the feasible action
set, and the "saturates at K ~ 3-4 past steps" justification for using only raw
past + acceleration + jerk) hinges on::

    rho = 2*pi*f_max*dt < 1.

This script turns ``f_max`` (and therefore ``rho``) into a measured,
reproducible number with an honest sensitivity analysis over several spectral
edge thresholds.

Usage
-----
    python measure_fmax.py \
        --data /path/to/libero_10.hdf5 OR /path/to/dir_of_hdf5s \
        --dt 0.05 \
        --thresholds 0.95 0.99 0.999 \
        --out ./fmax_out

Dependencies: h5py, numpy, scipy, matplotlib. No project-specific imports.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
from scipy import signal

import matplotlib

matplotlib.use("Agg")  # headless / reproducible
import matplotlib.pyplot as plt  # noqa: E402


# OSC_POSE delta + gripper action layout:
#   [dpx, dpy, dpz, drx, dry, drz, dgrip]
DIM_LABELS = ["dpx", "dpy", "dpz", "drx", "dry", "drz", "dgrip"]
POSE_DIMS = list(range(6))   # translation (0-2) + rotation (3-5)
GRIPPER_DIM = 6

MIN_DEMO_LEN = 16  # drop demos shorter than this many steps
MAX_NPERSEG = 128  # cap on Welch segment length


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _resolve_paths(data_arg: str) -> List[str]:
    """Resolve --data into a sorted list of HDF5 file paths."""
    if os.path.isdir(data_arg):
        paths = sorted(glob.glob(os.path.join(data_arg, "*.hdf5")))
        if not paths:
            raise FileNotFoundError(f"No *.hdf5 files found in directory: {data_arg}")
        return paths
    if os.path.isfile(data_arg):
        return [data_arg]
    raise FileNotFoundError(f"--data path does not exist: {data_arg}")


def _read_control_freq(f: h5py.File) -> Optional[float]:
    """Try to read control_freq from robomimic env_args metadata. Returns None
    if absent or unparseable."""
    try:
        env_args = json.loads(f["data"].attrs["env_args"])
        return float(env_args["env_kwargs"]["control_freq"])
    except Exception:
        return None


def load_actions(paths: List[str]) -> Tuple[List[np.ndarray], Dict]:
    """Load delta-action sequences from one or more LIBERO/robomimic HDF5 files.

    Returns a list of per-demo action arrays (each shape ``(T_i, 7)``) and a
    stats dict tracking files, demos used/dropped, total steps, and the set of
    ``control_freq`` values seen across files.
    """
    demos: List[np.ndarray] = []
    control_freqs: Dict[str, float] = {}  # path -> control_freq (if found)
    stats = {
        "n_files": 0,
        "n_demos_used": 0,
        "n_demos_dropped_short": 0,
        "n_demos_dropped_badshape": 0,
        "total_steps": 0,
        "files_skipped_no_data": [],
    }

    for path in paths:
        with h5py.File(path, "r") as f:
            if "data" not in f:
                print(f"[warn] no 'data' group in {path}; skipping file.")
                stats["files_skipped_no_data"].append(path)
                continue
            stats["n_files"] += 1

            cf = _read_control_freq(f)
            if cf is not None:
                control_freqs[path] = cf

            data = f["data"]
            for demo_key in sorted(data.keys()):
                grp = data[demo_key]
                if "actions" not in grp:
                    continue
                actions = np.asarray(grp["actions"][:], dtype=np.float64)
                if actions.ndim != 2 or actions.shape[1] != 7:
                    print(
                        f"[warn] {path}:{demo_key} actions shape "
                        f"{actions.shape} != (T,7); skipping demo."
                    )
                    stats["n_demos_dropped_badshape"] += 1
                    continue
                if actions.shape[0] < MIN_DEMO_LEN:
                    stats["n_demos_dropped_short"] += 1
                    continue
                demos.append(actions)
                stats["n_demos_used"] += 1
                stats["total_steps"] += actions.shape[0]

    stats["control_freqs"] = control_freqs
    return demos, stats


def resolve_dt(stats: Dict, cli_dt: float) -> Tuple[float, str]:
    """Pick dt from metadata when available, else fall back to the CLI default.

    Hard-errors if multiple files disagree on control_freq (pooling
    incomparable sampling rates would be meaningless).
    """
    control_freqs = stats.get("control_freqs", {})
    unique = sorted(set(round(v, 9) for v in control_freqs.values()))
    if len(unique) > 1:
        detail = ", ".join(f"{p}: {v}" for p, v in control_freqs.items())
        raise ValueError(
            f"Mixed control_freq across files ({detail}); refusing to pool "
            f"incomparable sampling rates."
        )
    if len(unique) == 1:
        return 1.0 / unique[0], "metadata"
    return cli_dt, "cli_default"


# --------------------------------------------------------------------------- #
# Welch PSD
# --------------------------------------------------------------------------- #
def welch_psd(
    demos: List[np.ndarray], dt: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute a length-weighted average Welch PSD per action dimension.

    Each demo is processed separately (never concatenated across demo
    boundaries, which would inject spurious high-frequency energy). A single
    shared ``nperseg`` is used so every PSD lands on the same frequency grid;
    demos shorter than that nperseg are skipped for the PSD (they were already
    length-filtered in loading, but this guards the grid invariant).

    Returns ``(freqs, psd_per_dim, weights_per_dim)`` where ``psd_per_dim`` has
    shape ``(7, n_freqs)`` and ``weights_per_dim`` is the total weight (summed
    demo lengths) contributing to each dimension's average.
    """
    fs = 1.0 / dt
    # Shared segment length across all demos -> shared frequency grid.
    shortest = min(d.shape[0] for d in demos)
    nperseg = min(MAX_NPERSEG, shortest)
    noverlap = nperseg // 2

    n_dims = demos[0].shape[1]
    psd_accum: Optional[np.ndarray] = None  # (n_dims, n_freqs)
    weight_accum = np.zeros(n_dims, dtype=np.float64)
    freqs: Optional[np.ndarray] = None

    for demo in demos:
        T = demo.shape[0]
        if T < nperseg:
            continue
        weight = float(T)  # length-weighting
        for d in range(n_dims):
            x = demo[:, d]
            x = x - np.mean(x)  # explicit DC removal (detrend per demo/dim)
            f_d, p_d = signal.welch(
                x,
                fs=fs,
                window="hann",
                detrend="constant",
                nperseg=nperseg,
                noverlap=noverlap,
            )
            if freqs is None:
                freqs = f_d
                psd_accum = np.zeros((n_dims, f_d.shape[0]), dtype=np.float64)
            psd_accum[d] += weight * p_d
            weight_accum[d] += weight

    if psd_accum is None:
        raise RuntimeError("No demo was long enough to compute a PSD.")

    psd_per_dim = psd_accum / weight_accum[:, None]
    return freqs, psd_per_dim, weight_accum


# --------------------------------------------------------------------------- #
# Spectral edge
# --------------------------------------------------------------------------- #
def spectral_edge(freqs: np.ndarray, psd: np.ndarray, q: float) -> float:
    """Smallest frequency below which fraction ``q`` of the spectral energy
    lies, linearly interpolated between bins for a smooth value.

    Real trajectories are not strictly band-limited, so we take this
    cumulative-energy edge as the "effective" band edge.
    """
    total = np.sum(psd)
    if total <= 0:
        return float("nan")
    cum = np.cumsum(psd) / total
    # First bin index where cumulative energy reaches q.
    idx = int(np.searchsorted(cum, q, side="left"))
    if idx <= 0:
        return float(freqs[0])
    if idx >= len(freqs):
        return float(freqs[-1])
    # Linear interpolation between bin (idx-1) and idx on the cumulative curve.
    c0, c1 = cum[idx - 1], cum[idx]
    f0, f1 = freqs[idx - 1], freqs[idx]
    if c1 == c0:
        return float(f1)
    frac = (q - c0) / (c1 - c0)
    return float(f0 + frac * (f1 - f0))


def pose_aggregate_psd(psd_per_dim: np.ndarray) -> np.ndarray:
    """Average the per-dimension *normalized* PSDs over the pose dims (0-5).

    Each pose dimension's PSD is normalized to integrate to 1 first, so a
    high-variance translation axis does not dominate the edge-frequency
    estimate.
    """
    normalized = []
    for d in POSE_DIMS:
        p = psd_per_dim[d]
        s = np.sum(p)
        normalized.append(p / s if s > 0 else p)
    return np.mean(np.stack(normalized, axis=0), axis=0)


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def make_report(
    freqs: np.ndarray,
    psd_per_dim: np.ndarray,
    pose_psd: np.ndarray,
    thresholds: List[float],
    dt: float,
    dt_source: str,
    stats: Dict,
) -> Dict:
    f_nyq = 1.0 / (2.0 * dt)
    f_crit = 1.0 / (2.0 * np.pi * dt)  # rho < 1  <=>  f_max < f_crit

    per_dim: Dict[str, Dict[str, float]] = {}
    for d, label in enumerate(DIM_LABELS):
        per_dim[label] = {
            f"{q}": spectral_edge(freqs, psd_per_dim[d], q) for q in thresholds
        }

    pose_fmax = {f"{q}": spectral_edge(freqs, pose_psd, q) for q in thresholds}
    pose_rho = {f"{q}": 2.0 * np.pi * pose_fmax[f"{q}"] * dt for q in thresholds}
    pose_rho_below_1 = {f"{q}": bool(pose_rho[f"{q}"] < 1.0) for q in thresholds}

    gripper_fmax = {
        f"{q}": spectral_edge(freqs, psd_per_dim[GRIPPER_DIM], q) for q in thresholds
    }

    headline_q = 0.99 if any(abs(q - 0.99) < 1e-12 for q in thresholds) else thresholds[
        len(thresholds) // 2
    ]
    hq = f"{headline_q}"
    headline = {
        "threshold": headline_q,
        "f_max_pose": pose_fmax[hq],
        "rho": pose_rho[hq],
        "rho_below_1": pose_rho_below_1[hq],
    }

    report = {
        "dt": dt,
        "dt_source": dt_source,
        "f_nyquist": f_nyq,
        "f_crit": f_crit,
        "rho_below_1_condition": "f_max < f_crit  <=>  rho = 2*pi*f_max*dt < 1",
        "thresholds": thresholds,
        "n_files": stats["n_files"],
        "n_demos_used": stats["n_demos_used"],
        "n_demos_dropped": {
            "too_short": stats["n_demos_dropped_short"],
            "bad_shape": stats["n_demos_dropped_badshape"],
        },
        "total_steps": stats["total_steps"],
        "files_skipped_no_data": stats["files_skipped_no_data"],
        "dim_labels": DIM_LABELS,
        "per_dimension": {
            "f_max": per_dim,
            "note": "axis-angle rotation deltas treated as scalar series; "
            "wraparound negligible at per-step delta scale.",
        },
        "pose_aggregate": {
            "dims_included": [DIM_LABELS[d] for d in POSE_DIMS],
            "f_max": pose_fmax,
            "rho": pose_rho,
            "rho_below_1": pose_rho_below_1,
            "note": "per-dimension PSDs normalized to unit energy before "
            "averaging so no single axis dominates the edge estimate.",
        },
        "gripper": {
            "dim": DIM_LABELS[GRIPPER_DIM],
            "f_max": gripper_fmax,
            "band_limited": False,
            "excluded_from_headline": True,
            "note": "gripper is quasi-binary / switch-like: its spectrum is "
            "broadband and NOT band-limited; reported but excluded from the "
            "headline pose f_max.",
        },
        "headline": headline,
    }
    return report


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
def make_figure(
    freqs: np.ndarray,
    psd_per_dim: np.ndarray,
    pose_psd: np.ndarray,
    thresholds: List[float],
    dt: float,
    headline: Dict,
    out_path: str,
) -> None:
    f_crit = 1.0 / (2.0 * np.pi * dt)
    trans_color = "tab:blue"
    rot_color = "tab:green"
    grip_color = "tab:red"

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(9, 9))

    # --- Top: per-dimension PSD on log-y, grouped/colored by channel type. ---
    for d in range(3):  # translation
        ax_top.semilogy(
            freqs, psd_per_dim[d], color=trans_color, alpha=0.8,
            label="translation (dpx,dpy,dpz)" if d == 0 else None,
        )
    for d in range(3, 6):  # rotation
        ax_top.semilogy(
            freqs, psd_per_dim[d], color=rot_color, alpha=0.8,
            label="rotation (drx,dry,drz)" if d == 3 else None,
        )
    ax_top.semilogy(
        freqs, psd_per_dim[GRIPPER_DIM], color=grip_color, alpha=0.9,
        linestyle="--", label="gripper (dgrip) - NOT band-limited",
    )
    ax_top.set_xlabel("frequency (Hz)")
    ax_top.set_ylabel("PSD")
    ax_top.set_title("Per-dimension Welch PSD")
    ax_top.legend(fontsize=8, loc="upper right")
    ax_top.grid(True, which="both", alpha=0.3)

    # --- Bottom: pose-aggregate cumulative energy + threshold markers. ---
    cum = np.cumsum(pose_psd) / np.sum(pose_psd)
    ax_bot.plot(freqs, cum, color="black", label="pose-aggregate cumulative energy")
    cmap = plt.get_cmap("viridis")
    for i, q in enumerate(thresholds):
        f_q = spectral_edge(freqs, pose_psd, q)
        color = cmap(0.15 + 0.7 * i / max(1, len(thresholds) - 1))
        ax_bot.axvline(f_q, color=color, linestyle=":", alpha=0.9)
        ax_bot.annotate(
            f"q={q}\nf_max={f_q:.2f} Hz",
            xy=(f_q, q),
            xytext=(5, -10 - 12 * i),
            textcoords="offset points",
            fontsize=8,
            color=color,
        )
    ax_bot.axvline(
        f_crit, color="crimson", linestyle="-", alpha=0.7,
        label=f"f_crit = {f_crit:.2f} Hz (rho=1)",
    )
    ax_bot.set_xlabel("frequency (Hz)")
    ax_bot.set_ylabel("cumulative energy fraction")
    ax_bot.set_ylim(0, 1.02)
    ax_bot.set_title("Pose-aggregate cumulative spectral energy (gripper excluded)")
    ax_bot.legend(fontsize=8, loc="lower right")
    ax_bot.grid(True, alpha=0.3)

    fig.suptitle(
        f"LIBERO-10 delta-action f_max  |  dt = {dt:.3f} s "
        f"({1.0 / dt:.1f} Hz)  |  headline f_max = {headline['f_max_pose']:.2f} Hz, "
        f"rho = {headline['rho']:.3f} "
        f"({'< 1 OK' if headline['rho_below_1'] else '>= 1 !!'})",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_csv(report: Dict, thresholds: List[float], out_path: str) -> None:
    lines = ["dim," + ",".join(f"f_max@{q}" for q in thresholds)]
    fmax = report["per_dimension"]["f_max"]
    for label in DIM_LABELS:
        row = [label] + [f"{fmax[label][f'{q}']:.6f}" for q in thresholds]
        lines.append(",".join(row))
    # Pose aggregate row.
    pose = report["pose_aggregate"]["f_max"]
    lines.append(
        ",".join(["pose_aggregate"] + [f"{pose[f'{q}']:.6f}" for q in thresholds])
    )
    with open(out_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- #
# Console summary + paper sentence
# --------------------------------------------------------------------------- #
def print_summary(report: Dict, thresholds: List[float]) -> None:
    print("\n" + "=" * 78)
    print("f_max measurement summary")
    print("=" * 78)
    print(
        f"dt = {report['dt']:.4f} s ({1.0 / report['dt']:.2f} Hz)  "
        f"[source: {report['dt_source']}]"
    )
    print(
        f"f_nyquist = {report['f_nyquist']:.3f} Hz   "
        f"f_crit (rho=1) = {report['f_crit']:.3f} Hz"
    )
    print(
        f"files={report['n_files']}  demos_used={report['n_demos_used']}  "
        f"dropped(short)={report['n_demos_dropped']['too_short']}  "
        f"dropped(badshape)={report['n_demos_dropped']['bad_shape']}  "
        f"total_steps={report['total_steps']}"
    )

    # Per-dimension table.
    header = "dim".ljust(8) + "".join(f"f_max@{q}".rjust(14) for q in thresholds)
    print("\n" + header)
    print("-" * len(header))
    fmax = report["per_dimension"]["f_max"]
    for label in DIM_LABELS:
        tag = "  (gripper: not band-limited)" if label == "dgrip" else ""
        row = label.ljust(8) + "".join(
            f"{fmax[label][f'{q}']:.3f}".rjust(14) for q in thresholds
        )
        print(row + tag)
    pose = report["pose_aggregate"]["f_max"]
    print(
        "POSE".ljust(8)
        + "".join(f"{pose[f'{q}']:.3f}".rjust(14) for q in thresholds)
        + "  (headline channel)"
    )

    # rho row for pose aggregate.
    rho = report["pose_aggregate"]["rho"]
    below = report["pose_aggregate"]["rho_below_1"]
    print("\nPose-aggregate rho = 2*pi*f_max*dt:")
    for q in thresholds:
        flag = "rho < 1  OK" if below[f"{q}"] else "rho >= 1  !! FINDING"
        print(f"  q={q}:  f_max={pose[f'{q}']:.3f} Hz  rho={rho[f'{q}']:.3f}  [{flag}]")

    hl = report["headline"]
    print("\n" + "-" * 78)
    print(
        f"HEADLINE (q={hl['threshold']}):  f_max_pose = {hl['f_max_pose']:.3f} Hz   "
        f"rho = {hl['rho']:.3f}   rho_below_1 = {hl['rho_below_1']}"
    )
    print("-" * 78)

    # Paper-ready sentence.
    print("\nPaper-ready sentence:\n")
    dt = report["dt"]
    hz = 1.0 / dt
    if hl["rho_below_1"]:
        sentence = (
            f"For LIBERO-10 demonstrations (dt = {dt:.3f} s, {hz:.0f} Hz), we estimate "
            f"the effective band edge of the pose-channel delta actions by the "
            f"{int(round(hl['threshold'] * 100))}% spectral-energy cutoff of a Welch "
            f"PSD, giving f_max = {hl['f_max_pose']:.2f} Hz and "
            f"rho = 2*pi*f_max*dt = {hl['rho']:.2f} < 1, so the Taylor remainder "
            f"R_K = rho^K/K! contracts super-exponentially and saturates by K = 3-4 "
            f"(justifying raw past + acceleration + jerk)."
        )
    else:
        sentence = (
            f"For LIBERO-10 demonstrations (dt = {dt:.3f} s, {hz:.0f} Hz), the "
            f"{int(round(hl['threshold'] * 100))}% spectral-energy cutoff of a Welch "
            f"PSD of the pose-channel delta actions gives f_max = "
            f"{hl['f_max_pose']:.2f} Hz, so rho = 2*pi*f_max*dt = {hl['rho']:.2f} >= 1; "
            f"at this cutoff the Taylor remainder R_K = rho^K/K! does NOT contract, "
            f"and the K = 3-4 saturation claim requires a more conservative reading "
            f"(e.g. a lower energy threshold, or acknowledging residual high-frequency "
            f"content from non-band-limited motion)."
        )
    print("    " + sentence)
    print()


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Measure f_max and rho=2*pi*f_max*dt from LIBERO-10 "
        "delta-action FFT/PSD.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--data",
        required=True,
        help="Path to a single .hdf5 file OR a directory of *.hdf5 files.",
    )
    p.add_argument(
        "--dt",
        type=float,
        default=0.05,
        help="Control timestep fallback (used only if metadata is absent).",
    )
    p.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.95, 0.99, 0.999],
        help="Cumulative-energy fractions defining the spectral edge.",
    )
    p.add_argument(
        "--out",
        default="./fmax_out",
        help="Output directory (created if missing).",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    thresholds = sorted(args.thresholds)

    os.makedirs(args.out, exist_ok=True)

    paths = _resolve_paths(args.data)
    print(f"Found {len(paths)} HDF5 file(s).")

    demos, stats = load_actions(paths)
    if not demos:
        print("[error] No usable demos found. Nothing to do.", file=sys.stderr)
        return 1

    dt, dt_source = resolve_dt(stats, args.dt)
    print(f"Using dt = {dt:.4f} s ({1.0 / dt:.2f} Hz)  [source: {dt_source}]")

    freqs, psd_per_dim, _weights = welch_psd(demos, dt)
    pose_psd = pose_aggregate_psd(psd_per_dim)

    report = make_report(
        freqs, psd_per_dim, pose_psd, thresholds, dt, dt_source, stats
    )

    # Write outputs.
    json_path = os.path.join(args.out, "fmax_report.json")
    with open(json_path, "w") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)

    fig_path = os.path.join(args.out, "fmax_psd.png")
    make_figure(
        freqs, psd_per_dim, pose_psd, thresholds, dt, report["headline"], fig_path
    )

    csv_path = os.path.join(args.out, "fmax_per_dim.csv")
    write_csv(report, thresholds, csv_path)

    print_summary(report, thresholds)

    print(f"Wrote: {json_path}")
    print(f"Wrote: {fig_path}")
    print(f"Wrote: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
