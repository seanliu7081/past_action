"""Like scripts/eval_policy_sim_smoothness.py, but swaps the LIBERO env runner
for LiberoRunnerFeasibility so the eval log includes feasibility / legality
metrics (L0 command legality, L1 joint feasibility, L3 realized EE motion)
alongside the existing success-rate keys.

Usage:
  python scripts/eval_policy_sim_feasibility.py \\
      -c path/to/policy.ckpt -o path/to/output_dir

The env_runner target is overridden in-memory after the checkpoint config is
loaded; the policy checkpoint and the existing libero_runner.py / config files
are not modified. The swap also injects `robot0_joint_pos` / `robot0_joint_vel`
into the env's state_ports (in-memory only) so the L1 joint metrics have data --
the base config only emits eef/gripper state.

Aggregation across `-n/--num_exp` runs matches the smoothness wrapper for the
mean metrics (mean over runs, `_mean` suffix), but the worst-case keys
(FEAS_KEYS_WORST) are reduced with min/max across runs and kept *bare* -- the
point of a worst case is the absolute extreme, not an average of extremes.
"""
if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import sys
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import json
import os
import pathlib
from typing import List, Optional

import click
import hydra
import numpy as np
import torch
import wandb
from omegaconf import OmegaConf

from oat.env_runner.base_runner import BaseRunner
from oat.policy.base_policy import BasePolicy
from oat.eval.trajectory_feasibility import FEAS_KEYS_WORST


FEAS_TARGET = "oat.env_runner.libero_runner_feasibility.LiberoRunnerFeasibility"

# state_ports the L1 joint metrics need; merged into whatever the config already
# emits (which the policy still consumes). Extra ports are filtered out before
# they reach the policy, so this is safe.
_L1_STATE_PORTS = [
    "robot0_joint_pos", "robot0_joint_vel", "robot0_eef_pos", "robot0_eef_quat",
]


def _swap_runner_to_feasibility(
    cfg, pos_slice, rot_slice, grip_idx,
    action_low, action_high, fs,
    enable_singularity, enable_collision,
    n_test=None, n_parallel_envs=None,
):
    """Mutates cfg.task.policy.env_runner so hydra.instantiate builds the
    feasibility subclass with the right extra kwargs, and ensures the env emits
    the joint state needed for L1. In-memory only -- no config file is edited."""
    OmegaConf.set_struct(cfg, False)
    runner_cfg = cfg.task.policy.env_runner
    orig_target = runner_cfg.get("_target_", "<missing>")
    runner_cfg._target_ = FEAS_TARGET
    runner_cfg.feas_pos_slice = list(pos_slice)
    runner_cfg.feas_rot_slice = list(rot_slice)
    runner_cfg.feas_grip_idx = int(grip_idx)
    runner_cfg.feas_action_low = float(action_low)
    runner_cfg.feas_action_high = float(action_high)
    runner_cfg.feas_fs = float(fs)
    runner_cfg.feas_enable_singularity = bool(enable_singularity)
    runner_cfg.feas_enable_collision = bool(enable_collision)

    # Merge in the joint ports (preserving order + whatever the policy needs).
    merged = list(runner_cfg.get("state_ports", []) or [])
    for p in _L1_STATE_PORTS:
        if p not in merged:
            merged.append(p)
    runner_cfg.state_ports = merged

    # Optional smoke-test overrides.
    if n_test is not None:
        runner_cfg.n_test = int(n_test)
        cur_vis = int(runner_cfg.get("n_test_vis", 0) or 0)
        runner_cfg.n_test_vis = min(cur_vis, int(n_test))
    if n_parallel_envs is not None:
        runner_cfg.n_parallel_envs = int(n_parallel_envs)
    OmegaConf.set_struct(cfg, True)
    print(f"[feasibility] swapped env_runner _target_: {orig_target} -> {FEAS_TARGET}")
    print(f"[feasibility] state_ports -> {merged}")


def _worst_reduction(full_key: str) -> Optional[str]:
    """Return 'min'/'max' if `full_key` is a FEAS_KEYS_WORST metric (at overall
    or per-task scope), else None."""
    k = full_key.split("/", 1)[1] if "/" in full_key else full_key
    if k.startswith("rollout_"):
        k = k[len("rollout_"):]
    return FEAS_KEYS_WORST.get(k)


@click.command()
@click.option('-c', '--checkpoint', required=True,
              help="either a .ckpt file or a directory containing .ckpt files")
@click.option('-o', '--output_dir', required=True,
              help="output directory for eval info dump")
@click.option('-n', '--num_exp', default=1, help="num experiments to run")
@click.option('-d', '--device', default='cuda:0', help="device to run on")
@click.option('--temperature', default=None, type=float)
@click.option('--topk', default=None, type=int)
@click.option('--use_k_tokens', default=None, type=int)
@click.option('--pos-start', default=0, type=int)
@click.option('--pos-end',   default=3, type=int)
@click.option('--rot-start', default=3, type=int)
@click.option('--rot-end',   default=6, type=int)
@click.option('--grip-idx',  default=6, type=int)
@click.option('--action-low',  default=-1.0, type=float,
              help='lower bound of the (pre-clip) action box for OOB / saturation')
@click.option('--action-high', default=1.0, type=float,
              help='upper bound of the (pre-clip) action box for OOB / saturation')
@click.option('--fs', default=20.0, type=float,
              help='command control rate (Hz); realized-obs rate is fs/n_action_steps')
@click.option('--enable-singularity', is_flag=True, default=False,
              help='(no-op in multiprocess env: Jacobian is unreachable, skipped)')
@click.option('--enable-collision', is_flag=True, default=False,
              help='(no-op in multiprocess env: sim contacts are unreachable, skipped)')
@click.option('--n-test', default=None, type=int,
              help='override n_test (in-memory) -- handy for a quick smoke test')
@click.option('--n-parallel-envs', default=None, type=int,
              help='override n_parallel_envs (in-memory) -- handy for a quick smoke test')
def eval_policy_sim_feasibility(
    checkpoint: str,
    output_dir: str,
    num_exp: int = 1,
    device: str = 'cuda:0',
    temperature: Optional[float] = None,
    topk: Optional[int] = None,
    use_k_tokens: Optional[int] = None,
    pos_start: int = 0,
    pos_end: int = 3,
    rot_start: int = 3,
    rot_end: int = 6,
    grip_idx: int = 6,
    action_low: float = -1.0,
    action_high: float = 1.0,
    fs: float = 20.0,
    enable_singularity: bool = False,
    enable_collision: bool = False,
    n_test: Optional[int] = None,
    n_parallel_envs: Optional[int] = None,
):
    if os.path.exists(output_dir):
        click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
        os.system(f"rm -rf {output_dir}")
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    ckpts: List[str]
    if os.path.isdir(checkpoint):
        ckpts = [os.path.join(checkpoint, f) for f in os.listdir(checkpoint)
                 if f.endswith('.ckpt') and f != 'latest.ckpt']
    else:
        ckpts = [checkpoint]

    base_output_dir = output_dir
    for ckpt in ckpts:
        if len(ckpts) > 1:
            ckpt_name = os.path.basename(ckpt).replace('.ckpt', '')
            output_dir = os.path.join(base_output_dir, ckpt_name)
            pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
        else:
            output_dir = base_output_dir

        policy, cfg = BasePolicy.from_checkpoint(ckpt, return_configuration=True)

        _swap_runner_to_feasibility(
            cfg,
            pos_slice=(pos_start, pos_end),
            rot_slice=(rot_start, rot_end),
            grip_idx=grip_idx,
            action_low=action_low,
            action_high=action_high,
            fs=fs,
            enable_singularity=enable_singularity,
            enable_collision=enable_collision,
            n_test=n_test,
            n_parallel_envs=n_parallel_envs,
        )

        device_t = torch.device(device)
        policy.to(device_t)
        policy.eval()

        print(f"Running evaluation on {ckpt}")
        env_runner: BaseRunner = hydra.utils.instantiate(
            cfg.task.policy.env_runner,
            output_dir=output_dir,
        )

        kwargs = {}
        if temperature is not None:
            kwargs['temperature'] = temperature
        if topk is not None:
            kwargs['topk'] = topk
        if use_k_tokens is not None:
            kwargs['use_k_tokens'] = use_k_tokens

        runner_log = env_runner.run(policy, **kwargs)

        all_runs = []
        for key, value in runner_log.items():
            if isinstance(value, wandb.sdk.data_types.video.Video):
                runner_log[key] = [value]
        all_runs.append({k: v for k, v in runner_log.items() if not isinstance(v, list)})
        print(f"Exp 1: success rate = {runner_log['mean_success_rate']}")

        for i in range(num_exp - 1):
            this_log = env_runner.run(policy, **kwargs)
            print(f"Exp {i + 2}: success rate = {this_log['mean_success_rate']}")
            all_runs.append({k: v for k, v in this_log.items() if not isinstance(v, list)})
            for key, value in this_log.items():
                if isinstance(value, wandb.sdk.data_types.video.Video):
                    if key in runner_log and isinstance(runner_log[key], list):
                        runner_log[key].append(value)
                    else:
                        runner_log[key] = [value]

        env_runner.close()

        # Union of numeric keys across runs (a metric can be absent in a run if
        # no episode populated it, e.g. a modality that was skipped).
        numeric_keys = []
        seen = set()
        for run in all_runs:
            for k in run.keys():
                if k not in seen:
                    seen.add(k)
                    numeric_keys.append(k)

        json_log = {'checkpoint': ckpt, 'num_exp': num_exp}
        for key in numeric_keys:
            values = [run[key] for run in all_runs if key in run]
            if not values:
                continue
            red = _worst_reduction(key)
            if red is not None:
                # Worst-case: reduce across runs by min/max, keep the key bare.
                json_log[key] = float(np.min(values) if red == "min" else np.max(values))
            else:
                mean_v = float(np.mean(values))
                json_log[f'{key}_mean'] = mean_v
                if num_exp > 1 and len(values) > 1:
                    std_v = float(np.std(values, ddof=1))
                    json_log[f'{key}_std'] = std_v
                    json_log[f'{key}_stderr'] = float(std_v / np.sqrt(len(values)))

        for key, value in runner_log.items():
            if isinstance(value, list):
                for i, video in enumerate(value):
                    if isinstance(video, wandb.sdk.data_types.video.Video):
                        json_log[f'{key}_{i}'] = video._path

        out_path = os.path.join(output_dir, 'eval_log.json')
        json.dump(json_log, open(out_path, 'w'), indent=2, sort_keys=True)
        print(f"[feasibility] wrote {out_path}")


if __name__ == '__main__':
    eval_policy_sim_feasibility()
