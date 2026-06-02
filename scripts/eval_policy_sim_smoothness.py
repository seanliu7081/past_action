"""Like scripts/eval_policy_sim.py, but swaps the LIBERO env runner for
LiberoRunnerSmoothness so the eval log includes trajectory smoothness metrics.

Usage:
  python scripts/eval_policy_sim_smoothness.py \\
      -c path/to/policy.ckpt -o path/to/output_dir

All CLI flags from the original script are preserved. The env_runner target
is overridden in-memory after the checkpoint config is loaded; the policy
checkpoint and the existing libero_runner.py are not modified.
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


SMOOTH_TARGET = "oat.env_runner.libero_runner_smoothness.LiberoRunnerSmoothness"


def _swap_runner_to_smoothness(
    cfg, rot_mode: str, pos_slice, rot_slice,
):
    """Mutates cfg.task.policy.env_runner so hydra.instantiate builds the
    smoothness subclass with the right extra kwargs."""
    OmegaConf.set_struct(cfg, False)
    runner_cfg = cfg.task.policy.env_runner
    orig_target = runner_cfg.get("_target_", "<missing>")
    runner_cfg._target_ = SMOOTH_TARGET
    runner_cfg.smoothness_rot_mode = rot_mode
    runner_cfg.smoothness_pos_slice = list(pos_slice)
    runner_cfg.smoothness_rot_slice = list(rot_slice)
    OmegaConf.set_struct(cfg, True)
    print(f"[smoothness] swapped env_runner _target_: {orig_target} -> {SMOOTH_TARGET}")


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
@click.option('--rot-mode', default='euclidean',
              type=click.Choice(['euclidean', 'geodesic']),
              help='Rotation differencing mode for smoothness metrics')
@click.option('--pos-start', default=0, type=int)
@click.option('--pos-end',   default=3, type=int)
@click.option('--rot-start', default=3, type=int)
@click.option('--rot-end',   default=6, type=int)
def eval_policy_sim_smoothness(
    checkpoint: str,
    output_dir: str,
    num_exp: int = 1,
    device: str = 'cuda:0',
    temperature: Optional[float] = None,
    topk: Optional[int] = None,
    use_k_tokens: Optional[int] = None,
    rot_mode: str = 'euclidean',
    pos_start: int = 0,
    pos_end: int = 3,
    rot_start: int = 3,
    rot_end: int = 6,
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

        _swap_runner_to_smoothness(
            cfg,
            rot_mode=rot_mode,
            pos_slice=(pos_start, pos_end),
            rot_slice=(rot_start, rot_end),
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
                assert key in runner_log
                if isinstance(value, wandb.sdk.data_types.video.Video):
                    runner_log[key].append(value)
                else:
                    runner_log[key] += value

        numeric_keys = [k for k in all_runs[0].keys()]
        mean_log, std_log = {}, {}
        for key in numeric_keys:
            values = [run[key] for run in all_runs]
            mean_log[key] = np.mean(values)
            if num_exp > 1:
                std_log[key] = np.std(values, ddof=1)

        env_runner.close()

        json_log = {'checkpoint': ckpt, 'num_exp': num_exp,
                    'smoothness_rot_mode': rot_mode}
        for key, value in mean_log.items():
            json_log[f'{key}_mean'] = float(value)
        if num_exp > 1:
            for key, value in std_log.items():
                json_log[f'{key}_std'] = float(value)
                json_log[f'{key}_stderr'] = float(value / np.sqrt(num_exp))
        for key, value in runner_log.items():
            if isinstance(value, list):
                for i, video in enumerate(value):
                    assert isinstance(video, wandb.sdk.data_types.video.Video)
                    json_log[f'{key}_{i}'] = video._path

        out_path = os.path.join(output_dir, 'eval_log.json')
        json.dump(json_log, open(out_path, 'w'), indent=2, sort_keys=True)
        print(f"[smoothness] wrote {out_path}")


if __name__ == '__main__':
    eval_policy_sim_smoothness()
