"""LiberoRunner subclass that adds feasibility / legality metrics to rollouts.

This mirrors ``libero_runner_smoothness.py`` but, instead of derivatives of the
command stream (which only measure *calmness*), it measures whether the policy's
actions are *legal and physically feasible* in sim:

    L0 command legality  -> the executed (pre-clip) action stream
    L1 joint feasibility -> realized robot0_joint_pos / robot0_joint_vel
    L3 realized motion    -> realized robot0_eef_pos / robot0_eef_quat

Strategy (purely additive, no existing file touched):

  * ``policy.reset`` is monkey-patched to count chunks. The parent runner calls
    ``reset()`` exactly once per chunk of parallel envs -- the same invariant the
    smoothness runner relies on for its index math.
  * ``self.env.step`` (the AsyncVectorEnv) is monkey-patched to record, per
    macro-step, (a) the *action actually sent to the env* -- which for this repo
    is the raw PRE-CLIP policy output, because the policy does not clamp and the
    OSC controller clips internally (so ``oob_frac_*`` is meaningful) -- and
    (b) the realized obs it returns, from which qpos/qvel/eef_pos/eef_quat are
    extracted by their real key names (with alias fallback; absent keys skipped).

Why hook ``env.step`` for the action rather than ``predict_action``: it yields
the true executed stream (the ``n_action_steps`` stride) with no chunk-seam
double counting, and it is the only place the per-step realized obs is available.

Realized-obs sampling caveat (a real deviation from the naive fs=20 assumption):
``MultiStepWrapper`` keeps only the last ``n_obs_steps+1`` observations, so the
per-step realized obs stream is *not* recoverable from ``env.step`` -- only the
chunk-boundary state is. We therefore take the last obs sub-step per macro-step,
giving a realized trajectory *uniformly* sampled at the chunk rate
``fs / n_action_steps`` (e.g. 20/8 = 2.5 Hz), and feed that reduced rate to the
realized-smoothness metrics so velocities come out in correct SI units. Joint
pos/vel violations are instantaneous observables, so their correctness does not
depend on the sampling rate. The command-legality metrics use the full-rate
action stream (all ``n_action_steps`` sub-steps) captured from ``env.step``.

Singularity (needs the geometric Jacobian) and self-collision (needs sim
contacts) live inside the AsyncVectorEnv worker processes and never cross the
``env.step`` boundary, so those modules are skipped here (their keys are simply
absent). The corresponding config flags are accepted but are no-ops in this
multiprocess setup.

Opt-in via Hydra by setting ``_target_`` to this class (see
``scripts/eval_policy_sim_feasibility.py``).
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from oat.env_runner.libero_runner import LiberoRunner
from oat.policy.base_policy import BasePolicy


# Realized-obs key aliases: canonical name -> ordered candidate obs-dict keys.
_OBS_ALIASES = {
    "qpos":     ["robot0_joint_pos", "joint_pos", "robot0_joint_positions"],
    "qvel":     ["robot0_joint_vel", "joint_vel", "robot0_joint_velocities"],
    "eef_pos":  ["robot0_eef_pos", "eef_pos", "robot0_eef_position"],
    "eef_quat": ["robot0_eef_quat", "eef_quat", "robot0_eef_orientation"],
}


class LiberoRunnerFeasibility(LiberoRunner):
    """LiberoRunner that also reports rollout feasibility / legality metrics."""

    def __init__(
        self,
        *args,
        feas_pos_slice: tuple = (0, 3),
        feas_rot_slice: tuple = (3, 6),
        feas_grip_idx: int = 6,
        feas_action_low: float = -1.0,
        feas_action_high: float = 1.0,
        feas_fs: float = 20.0,
        feas_enable_singularity: bool = False,
        feas_enable_collision: bool = False,
        feas_log_exec_smoothness: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.feas_pos_slice = tuple(feas_pos_slice)
        self.feas_rot_slice = tuple(feas_rot_slice)
        self.feas_grip_idx = int(feas_grip_idx)
        self.feas_action_low = float(feas_action_low)
        self.feas_action_high = float(feas_action_high)
        self.feas_fs = float(feas_fs)
        self.feas_enable_singularity = bool(feas_enable_singularity)
        self.feas_enable_collision = bool(feas_enable_collision)
        self.feas_log_exec_smoothness = bool(feas_log_exec_smoothness)

    # ── main entry ───────────────────────────────────────────────────────────

    def run(self, policy: BasePolicy, **kwargs) -> Dict[str, Any]:
        n_inits = len(self.env_init_fn_dills)
        n_envs = len(self.env_fns)
        n_chunks = math.ceil(n_inits / n_envs)
        active_counts = [
            min(n_envs, n_inits - chunk_idx * n_envs) for chunk_idx in range(n_chunks)
        ]

        # Per chunk: list (over macro-steps) of action arrays [n_envs, T, D].
        cap_actions: List[List[np.ndarray]] = [[] for _ in range(n_chunks)]
        # Per chunk: canonical field -> list (over macro-steps) of [n_envs, feat].
        cap_obs: List[Dict[str, List[np.ndarray]]] = [
            defaultdict(list) for _ in range(n_chunks)
        ]
        current_chunk = [-1]  # mutable closure box

        orig_reset = policy.reset
        orig_env_step = self.env.step

        def tap_reset(*a, **kw):
            current_chunk[0] += 1
            return orig_reset(*a, **kw)

        def tap_env_step(action, *a, **kw):
            out = orig_env_step(action, *a, **kw)
            ci = current_chunk[0]
            if 0 <= ci < n_chunks:
                act = np.asarray(action).copy()          # [n_envs, T, D] pre-clip
                cap_actions[ci].append(act)
                obs = out[0]
                if isinstance(obs, dict):
                    for field, aliases in _OBS_ALIASES.items():
                        key = next((k for k in aliases if k in obs), None)
                        if key is None:
                            continue
                        arr = np.asarray(obs[key])       # [n_envs, n_obs_steps, feat]
                        # last obs sub-step -> one realized sample per macro-step
                        cap_obs[ci][field].append(arr[:, -1].copy())
            return out

        policy.reset = tap_reset
        self.env.step = tap_env_step
        try:
            log_data = super().run(policy, **kwargs)
        finally:
            policy.reset = orig_reset
            self.env.step = orig_env_step

        episodes = self._reassemble_episodes(
            cap_actions, cap_obs, active_counts, n_envs, n_inits
        )
        self._populate_feasibility_metrics(episodes, log_data)
        return log_data

    # ── reassembly (mirrors the smoothness runner's index math) ──────────────

    def _reassemble_episodes(
        self,
        cap_actions: List[List[np.ndarray]],
        cap_obs: List[Dict[str, List[np.ndarray]]],
        active_counts: List[int],
        n_envs: int,
        n_inits: int,
    ) -> List[Optional[Dict[str, np.ndarray]]]:
        """Return a list (length n_inits) of per-episode dicts holding the subset
        of {action, qpos, qvel, eef_pos, eef_quat} that was captured."""
        episodes: List[Optional[Dict[str, np.ndarray]]] = [None] * n_inits
        for chunk_idx, calls in enumerate(cap_actions):
            k_active = active_counts[chunk_idx]
            if k_active == 0 or len(calls) == 0:
                continue
            actions = np.stack(calls, axis=0)  # [n_calls, n_envs, T, D]
            n_calls, n_envs_seen, T_chunk, D = actions.shape

            # Stack captured obs fields for this chunk: field -> [n_calls, n_envs, feat]
            obs_stacks: Dict[str, np.ndarray] = {}
            for field, lst in cap_obs[chunk_idx].items():
                if len(lst) == n_calls:
                    obs_stacks[field] = np.stack(lst, axis=0)

            for env_local in range(min(k_active, n_envs_seen)):
                global_idx = chunk_idx * n_envs + env_local
                if global_idx >= n_inits:
                    continue
                ep: Dict[str, np.ndarray] = {
                    "action": actions[:, env_local].reshape(n_calls * T_chunk, D)
                }
                for field, st in obs_stacks.items():
                    ep[field] = st[:, env_local]  # [n_calls, feat]
                episodes[global_idx] = ep
        return episodes

    # ── scoring & aggregation (mirrors _populate_smoothness_metrics) ─────────

    def _populate_feasibility_metrics(
        self,
        episodes: List[Optional[Dict[str, np.ndarray]]],
        log_data: Dict[str, Any],
    ) -> None:
        from oat.eval.trajectory_feasibility import (
            FEAS_KEYS_MEAN, FEAS_KEYS_WORST, compute_feasibility_metrics,
        )
        from oat.eval.trajectory_smoothness import (
            SMOOTH_KEYS, compute_smoothness_metrics,
        )

        # Command control rate for the (full-rate) action stream.
        fs_command = self.feas_fs
        # Realized obs is only observable at chunk boundaries -> reduced rate.
        n_action_steps = int(getattr(self, "n_action_steps", 1)) or 1
        fs_realized = fs_command / n_action_steps

        if self.feas_enable_singularity or self.feas_enable_collision:
            print(
                "[feasibility] singularity/collision requested but not reachable "
                "through the AsyncVectorEnv env.step boundary (Jacobian/contacts "
                "live in worker processes); these modules are skipped."
            )

        # Accumulators (only append when a metric is actually present).
        mean_vals: Dict[str, List[float]] = defaultdict(list)
        worst_vals: Dict[str, List[float]] = defaultdict(list)
        exec_vals: Dict[str, List[float]] = defaultdict(list)
        task_mean: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        task_worst: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        task_exec: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

        n_scored = 0
        for i, ep in enumerate(episodes):
            if ep is None:
                continue
            act = ep.get("action")
            if act is None or act.shape[0] < 4:
                continue
            n_scored += 1
            t_name = self.env_task_names[i]

            m = compute_feasibility_metrics(
                ep,
                fs=fs_realized,
                low=self.feas_action_low,
                high=self.feas_action_high,
                pos_slice=self.feas_pos_slice,
                rot_slice=self.feas_rot_slice,
                grip_idx=self.feas_grip_idx,
            )
            for k in FEAS_KEYS_MEAN:
                if k in m:
                    mean_vals[k].append(m[k])
                    task_mean[t_name][k].append(m[k])
            for k in FEAS_KEYS_WORST:
                if k in m:
                    worst_vals[k].append(m[k])
                    task_worst[t_name][k].append(m[k])

            # Optional: truthful command smoothness on the EXECUTED action stream,
            # under a distinct `exec_` prefix so legacy smoothness keys are untouched.
            if self.feas_log_exec_smoothness:
                t_th = torch.from_numpy(act).float().unsqueeze(0)  # [1, T, D]
                sm = compute_smoothness_metrics(
                    t_th,
                    pos_slice=self.feas_pos_slice,
                    rot_slice=self.feas_rot_slice,
                    fs=fs_command,
                    rot_mode="euclidean",
                )
                for sk in SMOOTH_KEYS:
                    v = float(sm[sk].item())
                    exec_vals[sk].append(v)
                    task_exec[t_name][sk].append(v)

        # ── emit overall + per-task, mirroring rollout_* naming ──────────────
        for k, vals in mean_vals.items():
            log_data[f"rollout_{k}"] = float(np.mean(vals))
        for k, vals in worst_vals.items():
            red = FEAS_KEYS_WORST[k]
            log_data[f"rollout_{k}"] = float(np.min(vals) if red == "min" else np.max(vals))
        for sk, vals in exec_vals.items():
            log_data[f"rollout_exec_{sk}"] = float(np.mean(vals))

        for t, d in task_mean.items():
            for k, vals in d.items():
                log_data[f"{t}/rollout_{k}"] = float(np.mean(vals))
        for t, d in task_worst.items():
            for k, vals in d.items():
                red = FEAS_KEYS_WORST[k]
                log_data[f"{t}/rollout_{k}"] = float(
                    np.min(vals) if red == "min" else np.max(vals)
                )
        for t, d in task_exec.items():
            for sk, vals in d.items():
                log_data[f"{t}/rollout_exec_{sk}"] = float(np.mean(vals))

        log_data["rollout_n_episodes_scored_feas"] = int(n_scored)
