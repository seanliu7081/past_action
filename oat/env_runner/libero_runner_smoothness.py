"""LiberoRunner subclass that adds trajectory-smoothness metrics to rollouts.

Wraps `policy.predict_action` and `policy.reset` to capture executed action
chunks per episode without modifying the parent runner. After the parent
`run()` completes, reassembles per-episode action streams and computes
velocity/acceleration/jerk RMS and SPARC over them, returning the results
alongside the parent's success-rate `log_data`.

Opt-in via Hydra by setting `_target_` to this class in the policy task config
(see `oat/oat/config/task/policy/libero/libero10_smoothness.yaml`).
"""
from __future__ import annotations

import math
from typing import Any, Dict, List

import numpy as np
import torch

from oat.env_runner.libero_runner import LiberoRunner
from oat.policy.base_policy import BasePolicy


class LiberoRunnerSmoothness(LiberoRunner):
    """LiberoRunner that also reports rollout trajectory smoothness.

    Strategy: monkey-patch `policy.predict_action` and `policy.reset` for the
    duration of the run to capture every action chunk, grouped by the parent's
    chunk_idx loop (one `policy.reset()` call per chunk). After the parent
    finishes, we know each chunk's active-env count from `n_envs` and
    `n_inits`, so we can reassemble each episode's action stream.

    Caveat: an env that succeeds early continues to be stepped (the parent only
    breaks when ALL envs in a chunk are done), so the tail of an early-success
    episode includes a few "irrelevant" actions. This adds mild noise but does
    not bias smoothness systematically.
    """

    def __init__(
        self,
        *args,
        smoothness_rot_mode: str = "euclidean",
        smoothness_pos_slice: tuple = (0, 3),
        smoothness_rot_slice: tuple = (3, 6),
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.smoothness_rot_mode = smoothness_rot_mode
        self.smoothness_pos_slice = tuple(smoothness_pos_slice)
        self.smoothness_rot_slice = tuple(smoothness_rot_slice)

    def run(self, policy: BasePolicy, **kwargs) -> Dict[str, Any]:
        n_inits = len(self.env_init_fn_dills)
        n_envs = len(self.env_fns)
        n_chunks = math.ceil(n_inits / n_envs)
        active_counts = [
            min(n_envs, n_inits - chunk_idx * n_envs) for chunk_idx in range(n_chunks)
        ]

        # captured[chunk_idx] -> list of action ndarrays each [n_envs, T_chunk, D]
        captured: List[List[np.ndarray]] = [[] for _ in range(n_chunks)]
        current_chunk = [-1]  # mutable closure box

        orig_predict = policy.predict_action
        orig_reset = policy.reset

        def tap_reset(*a, **kw):
            current_chunk[0] += 1
            return orig_reset(*a, **kw)

        def tap_predict(obs, **kw):
            out = orig_predict(obs, **kw)
            ci = current_chunk[0]
            if 0 <= ci < n_chunks:
                act = out["action"].detach().cpu().numpy().copy()
                captured[ci].append(act)
            return out

        policy.predict_action = tap_predict
        policy.reset = tap_reset
        try:
            log_data = super().run(policy, **kwargs)
        finally:
            policy.predict_action = orig_predict
            policy.reset = orig_reset

        # Reassemble per-episode action trajectories.
        episodes = self._reassemble_episodes(captured, active_counts, n_envs, n_inits)
        self._populate_smoothness_metrics(episodes, log_data)
        return log_data

    def _reassemble_episodes(
        self,
        captured: List[List[np.ndarray]],
        active_counts: List[int],
        n_envs: int,
        n_inits: int,
    ) -> List[np.ndarray]:
        """Build a list of per-episode action ndarrays [T_ep, D] (length n_inits)."""
        episodes: List[np.ndarray] = [None] * n_inits  # type: ignore[list-item]
        for chunk_idx, calls in enumerate(captured):
            k_active = active_counts[chunk_idx]
            if k_active == 0 or len(calls) == 0:
                continue
            stacked = np.stack(calls, axis=0)  # [n_calls, n_envs, T_chunk, D]
            n_calls, n_envs_seen, T_chunk, D = stacked.shape
            # Take active envs only and flatten time across calls.
            for env_local in range(min(k_active, n_envs_seen)):
                traj = stacked[:, env_local].reshape(n_calls * T_chunk, D)
                global_idx = chunk_idx * n_envs + env_local
                if global_idx < n_inits:
                    episodes[global_idx] = traj
        return episodes

    def _populate_smoothness_metrics(
        self,
        episodes: List[np.ndarray],
        log_data: Dict[str, Any],
    ) -> None:
        from oat.eval.trajectory_smoothness import (
            SMOOTH_KEYS, compute_smoothness_metrics,
        )

        fs = float(self.fps) if hasattr(self, "fps") else 20.0
        per_key: Dict[str, List[float]] = {k: [] for k in SMOOTH_KEYS}
        task_names = set(self.env_task_names)
        per_task_key: Dict[str, Dict[str, List[float]]] = {
            t: {k: [] for k in SMOOTH_KEYS} for t in task_names
        }

        for i, traj in enumerate(episodes):
            if traj is None or traj.shape[0] < 4:
                continue
            t_th = torch.from_numpy(traj).float().unsqueeze(0)  # [1, T_ep, D]
            m = compute_smoothness_metrics(
                t_th,
                pos_slice=self.smoothness_pos_slice,
                rot_slice=self.smoothness_rot_slice,
                fs=fs,
                rot_mode=self.smoothness_rot_mode,
            )
            t_name = self.env_task_names[i]
            for k in SMOOTH_KEYS:
                v = float(m[k].item())
                per_key[k].append(v)
                per_task_key[t_name][k].append(v)

        for k in SMOOTH_KEYS:
            log_data[f"rollout_{k}"] = float(np.mean(per_key[k])) if per_key[k] else 0.0
        for t, d in per_task_key.items():
            for k in SMOOTH_KEYS:
                log_data[f"{t}/rollout_{k}"] = (
                    float(np.mean(d[k])) if d[k] else 0.0
                )
        log_data["rollout_n_episodes_scored"] = int(
            sum(1 for tr in episodes if tr is not None and tr.shape[0] >= 4)
        )
