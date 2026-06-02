"""Standalone trajectory-smoothness eval for an OAT tokenizer checkpoint.

Loads a tokenizer training run (its saved Hydra config + checkpoint) without
re-training, iterates the validation set once through `tokenizer.autoencode`,
and reports velocity/acceleration/jerk RMS and SPARC for both reconstructed
and ground-truth action chunks (plus the recon/GT ratio).

Usage:
    python scripts/eval_tokenizer_smoothness.py \\
        --ckpt-dir output/20260517/HHMMSS_train_oattok_so3aug_libero10_N500 \\
        --ckpt-name latest.ckpt \\
        --device cuda:0 \\
        --fs 20.0 \\
        --wandb

The ckpt-dir is the training run's output directory (the one that contains
`.hydra/config.yaml` and `checkpoints/`).
"""
import argparse
import json
import os
import pathlib
import sys

ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import hydra
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from oat.common.hydra_util import register_new_resolvers
from oat.common.pytorch_util import dict_apply
from oat.eval.trajectory_smoothness import SMOOTH_KEYS, compute_smoothness_metrics
from oat.workspace.train_oattok import TrainOATTokWorkspace


register_new_resolvers()
# `${now:...}` is registered by Hydra at app startup; when loading a saved cfg
# directly via OmegaConf we re-register a static stub so any references that
# happen to be touched during resolution don't crash. The actual run-time
# values aren't needed for eval.
import datetime as _dt
_NOW_STR = _dt.datetime.now().strftime("%Y%m%d.%H%M%S")
OmegaConf.register_new_resolver(
    "now", lambda pattern="%Y-%m-%d_%H-%M-%S": _dt.datetime.now().strftime(pattern),
    replace=True,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", required=True,
                        help="Training run output dir (contains .hydra/ and checkpoints/)")
    parser.add_argument("--ckpt-name", default="latest.ckpt",
                        help="Checkpoint filename under <ckpt-dir>/checkpoints/")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fs", type=float, default=20.0,
                        help="Control frequency in Hz (LIBERO=20, MimicGen=10)")
    parser.add_argument("--rot-mode", choices=["euclidean", "geodesic"], default="euclidean")
    parser.add_argument("--max-batches", type=int, default=None,
                        help="Cap on number of val batches for quick checks")
    parser.add_argument("--use-ema", default="auto",
                        help="'auto' (read from cfg), 'true', or 'false'")
    parser.add_argument("--wandb", action="store_true", help="Also log results to W&B")
    parser.add_argument("--out-json", default=None, help="Optional path to dump results JSON")
    args = parser.parse_args()

    input_dir = pathlib.Path(args.ckpt_dir).resolve()
    # Accept either the run dir (contains .hydra/ and checkpoints/) or the
    # checkpoints/ subdir directly.
    if (input_dir / ".hydra" / "config.yaml").is_file():
        run_dir = input_dir
        ckpt_path = run_dir / "checkpoints" / args.ckpt_name
    elif (input_dir.parent / ".hydra" / "config.yaml").is_file():
        run_dir = input_dir.parent
        # If the user passed the checkpoints/ dir, the ckpt file sits inside it.
        ckpt_path = input_dir / args.ckpt_name
    else:
        raise FileNotFoundError(
            f"No .hydra/config.yaml found at {input_dir} or its parent. "
            f"Pass --ckpt-dir as the run output directory (the one containing "
            f"both .hydra/ and checkpoints/)."
        )
    cfg_path = run_dir / ".hydra" / "config.yaml"
    assert ckpt_path.is_file(), f"missing checkpoint file: {ckpt_path}"
    print(f"[eval-smoothness] run_dir: {run_dir}")

    saved_cfg = OmegaConf.load(cfg_path)

    # Construct workspace with eager instantiation so load_checkpoint can populate
    # self.model / self.ema_model state dicts directly.
    workspace = TrainOATTokWorkspace(
        cfg=saved_cfg,
        output_dir=str(run_dir),
        lazy_instantiation=False,
    )
    print(f"[eval-smoothness] loading {ckpt_path}")
    workspace.load_checkpoint(path=str(ckpt_path))

    if args.use_ema == "auto":
        use_ema = bool(saved_cfg.training.use_ema) and workspace.ema_model is not None
    else:
        use_ema = args.use_ema.lower() == "true"
    tokenizer = workspace.ema_model if use_ema else workspace.model
    tokenizer.eval().to(args.device)
    print(f"[eval-smoothness] using {'EMA' if use_ema else 'online'} weights")

    # Build val dataloader from the saved training cfg
    dataset = hydra.utils.instantiate(saved_cfg.task.tokenizer.dataset)
    val_dataset = dataset.get_validation_dataset()
    dl_kwargs = OmegaConf.to_container(saved_cfg.val_dataloader, resolve=True)
    # Drop_last=True can throw away the entire small val set; force False here.
    dl_kwargs["drop_last"] = False
    # The standalone eval doesn't need worker preheating.
    dl_kwargs["persistent_workers"] = False
    val_dl = DataLoader(val_dataset, **dl_kwargs)

    print(f"[eval-smoothness] val dataset size: {len(val_dataset)} samples; "
          f"batch_size={dl_kwargs['batch_size']} -> {len(val_dl)} batches")

    sums_recon = {k: 0.0 for k in SMOOTH_KEYS}
    sums_gt    = {k: 0.0 for k in SMOOTH_KEYS}
    sum_mse = 0.0
    n = 0
    with torch.inference_mode():
        for batch_idx, batch in enumerate(val_dl):
            batch = dict_apply(batch, lambda x: x.to(args.device, non_blocking=True))
            samples = batch["action"]
            recon = tokenizer.autoencode(samples=samples)
            m_r = compute_smoothness_metrics(recon,   fs=args.fs, rot_mode=args.rot_mode)
            m_g = compute_smoothness_metrics(samples, fs=args.fs, rot_mode=args.rot_mode)
            bsz = samples.shape[0]
            for k in SMOOTH_KEYS:
                sums_recon[k] += m_r[k].item() * bsz
                sums_gt[k]    += m_g[k].item() * bsz
            sum_mse += torch.nn.functional.mse_loss(recon, samples).item() * bsz
            n += bsz
            if args.max_batches is not None and batch_idx + 1 >= args.max_batches:
                break

    assert n > 0, "no validation samples processed"
    results = {"reconst_mse": sum_mse / n, "n_samples": n}
    for k in SMOOTH_KEYS:
        r = sums_recon[k] / n
        g = sums_gt[k]    / n
        results[f"reconst_{k}"] = r
        results[f"gt_{k}"]      = g
        # ratio: only meaningful for nonneg RMS-style metrics; SPARC is negative so use diff
        if k.startswith("sparc"):
            results[f"reconst_{k}_diff"] = r - g  # positive if recon is smoother
        else:
            results[f"reconst_{k}_ratio"] = r / max(abs(g), 1e-12)

    print(json.dumps(results, indent=2))

    if args.out_json:
        out = pathlib.Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2))
        print(f"[eval-smoothness] wrote {out}")

    if args.wandb:
        import wandb
        wandb.init(
            project=saved_cfg.logging.project,
            name=f"smoothness_{run_dir.name}",
            config={"run_dir": str(run_dir), "ckpt_name": args.ckpt_name,
                    "fs": args.fs, "rot_mode": args.rot_mode, "use_ema": use_ema},
        )
        wandb.log(results)
        wandb.finish()


if __name__ == "__main__":
    main()
