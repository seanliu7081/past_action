# OAT — Ordered Action Tokenization

OAT is a research codebase for **two-stage discrete action modelling** on robot manipulation
benchmarks (primarily LIBERO-10, plus MimicGen/robomimic):

1. **Stage 1 — action tokenizer.** An action-only autoencoder with a discrete bottleneck
   (FSQ / VQ / BSQ / …) compresses a `horizon`-step chunk of continuous 7-DoF actions into a
   handful of ordered tokens and reconstructs it. It never sees observations.
2. **Stage 2 — policy.** A causal transformer conditioned on observations (and, in the
   `*_with_*_past` variants, on past actions and their derivatives) autoregressively predicts the
   tokens of the *next* action chunk, decodes them with the **frozen** Stage-1 tokenizer, and
   executes them receding-horizon.

Continuous baselines that skip Stage 1 entirely (diffusion policy, flow matching, ACT) live in the
same harness, so every method shares the dataset, observation encoder, env runner and eval code.

Background reading, in rough order of usefulness:

| Doc | Contents |
|---|---|
| [PAST2NEXT.md](PAST2NEXT.md) | End-to-end architecture of the main method (`train_oattok_so3aug` + `train_oatpolicy_with_enriched_past`) |
| [method_ours_so3aug_enriched_past.md](method_ours_so3aug_enriched_past.md) | Detailed write-up of the proposed model |
| [method_baseline_oattok_oatpolicy.md](method_baseline_oattok_oatpolicy.md) | Baseline OAT tokenizer + OAT policy |
| [method_comparison_ours_vs_baseline.md](method_comparison_ours_vs_baseline.md) | Side-by-side comparison |
| [ordered_lattice_quantization.md](ordered_lattice_quantization.md) | Quantizer background |
| [docs/](docs/) | Per-baseline plans (ACT, flow-matching-with-enriched-past, PaliGemma-2 VLA) |
| [AGENTS.md](AGENTS.md) | Contribution / style conventions |

---

## 1. Repository layout

```
oat/                      # library (namespace package, importable as `oat`)
  common/                 # replay buffer, checkpointing, hydra resolvers, video/json logging
  config/                 # ALL Hydra experiment configs (train_*.yaml)
    task/tokenizer/...    # dataset-only task groups (Stage 1)
    task/policy/...       # dataset + env_runner + shape_meta task groups (Stage 2)
  dataset/                # ZarrDataset, ZarrDatasetWithPastAction
  env/                    # LIBERO + robomimic env wrappers, LIBERO hdf5 -> zarr conversion
  env_runner/             # parallel sim rollout runners (base / smoothness / feasibility)
  eval/                   # trajectory smoothness & feasibility metric implementations
  gymnasium_util/         # async vector env, multistep + video wrappers
  model/                  # act / autoregressive / diffusion / maskgit backbones, EMA, schedulers
  perception/             # observation encoders (robomimic ResNet, DINOv2, state, fused)
  policy/                 # all policies (oatpolicy, *_with_*_past, diff, flow, act, fast, quest, bin…)
  tokenizer/              # tokenizer families: oat, bin, fast, quest, a2lex, zhill, polar, polar_v2
  workspace/              # Hydra workspaces = training loops (entry points via `_target_`)
scripts/                  # data conversion, training entry point, sim eval
analysis_scripts/         # offline diagnostics (entropy, codebook usage, reconstruction, MSE …)
slurm/                    # Slurm launcher templates per method
third_party/LIBERO/       # git submodule (branch `oat`)
data/                     # datasets (gitignored)
output/                   # Hydra run dirs: checkpoints, wandb, media (gitignored)
tok_ckpt/                 # hand-curated frozen tokenizer checkpoints
test_*.py                 # top-level pytest files
```

Two conventions worth internalising up front:

- **Everything is Hydra.** One entry point, [scripts/run_workspace.py](scripts/run_workspace.py),
  selects a config in `oat/config/`, and that config's `_target_` names the workspace class that
  actually runs training.
- **Always run from the repo root.** The entry scripts `os.chdir(ROOT_DIR)`, and all default paths
  (`data/libero/...`, `output/...`) are relative to it.

---

## 2. Environment setup

Requirements: Linux x86_64, Python 3.10, an NVIDIA GPU with a driver supporting CUDA ≥ 12.6
(the pinned stack is torch 2.10 / cu128–cu129), and ~50 GB free disk for LIBERO data plus runs.

### 2.0 On this machine (already provisioned)

A ready environment exists at `/venv/oat` (Python 3.10.19, torch 2.10.0, `oat` and `libero`
installed editable against `/workspace/oat`). There is also a repo-local `.venv/` that `uv run`
picks up automatically. Either works:

```bash
cd /workspace/oat
source /venv/oat/bin/activate        # option A: the prebuilt venv
# or
uv run python -c "import oat"        # option B: repo-local .venv, managed by uv
```

Verify:

```bash
cd /workspace/oat
python -c "import torch, oat, libero; print(torch.__version__, torch.cuda.is_available())"
```

Skip to [§3 Data setup](#3-data-setup) if this works.

### 2.1 Clone with the LIBERO submodule

LIBERO is vendored as a submodule pinned to the `oat` branch of
`github.com/Chaoqi-LIU/LIBERO`; the code will not import without it.

```bash
git clone <repo-url> oat && cd oat
git submodule update --init --recursive
```

### 2.2 Pick an install path

**Option A — `uv` (matches `uv.lock`, what the Slurm scripts assume):**

```bash
uv sync                 # creates ./.venv with oat + third_party/LIBERO as workspace members
uv run python -c "import oat, libero"
```

All later commands can then be prefixed with `uv run` instead of activating a venv.

**Option B — `setup_env.sh` (fresh GPU box; installs apt deps too):**

```bash
chmod +x setup_env.sh
./setup_env.sh --cuda 12.9            # or omit --cuda to auto-detect from nvidia-smi
./setup_env.sh --venv-path /venv/oat  # optional custom location; default ./venv/oat
source ./venv/oat/bin/activate
```

This installs system libraries (`libgl1-mesa-glx`, `libegl1-mesa`, `libglfw3`, `libglew-dev`,
`libosmesa6-dev`, `patchelf`, …), the pinned pip set, then `pip install -e third_party/LIBERO`
and `pip install -e .`, and finally prints a verification block. It uses `sudo apt-get`, so run it
where you have root.

**Option C — conda:**

```bash
conda env create -f environment.yml   # fully pinned, includes the two editable installs
conda activate oat
```

(`conda_env.yaml` is an older, looser spec — prefer `environment.yml`.)

**Option D — an environment you already prepared:**

```bash
pip install -e third_party/LIBERO
pip install -e .
```

### 2.3 LIBERO paths

On first import LIBERO writes `~/.libero/config.yaml` pointing at `assets/`, `bddl_files/`,
`init_files/` and `datasets/` inside the submodule. Override the location with the
`LIBERO_CONFIG_PATH` environment variable if you need a non-default one. Confirm it resolves:

```bash
python -c "from libero.libero import get_libero_path; print(get_libero_path('bddl_files'))"
```

If it prints `[Warning]: ... path does not exist`, the submodule was not initialised or the config
points at a stale checkout — delete `~/.libero/config.yaml` and re-import.

### 2.4 Headless rendering

Sim rollouts use MuJoCo offscreen rendering. On headless machines export EGL **before** launching
anything that instantiates an env runner:

```bash
export MUJOCO_GL=egl
```

Every policy Slurm template sets `MUJOCO_GL=egl`; tokenizer training does not need it (no env).

### 2.5 Weights & Biases

Training logs through `accelerate`'s wandb tracker (project `oat_dev` by default). Either
`wandb login` once, or disable networking per run:

```bash
export WANDB_MODE=offline            # or add the override: logging.mode=offline
```

---

## 3. Data setup

### 3.1 LIBERO-10 (the default benchmark)

The pipeline is **download hdf5 → per-task zarr → merged multi-task zarr**.

```bash
# 1. download the raw LIBERO hdf5 demos (libero_100 contains the LIBERO-10 / long suite)
python third_party/LIBERO/benchmark_scripts/download_libero_datasets.py \
    --datasets libero_100 --use-huggingface

# 2. stage the *_demo.hdf5 files where the converter looks for them
mkdir -p data/libero/hdf5_datasets
cp third_party/LIBERO/libero/datasets/libero_10/*.hdf5 data/libero/hdf5_datasets/
#    (symlinks work too, and save the raw dataset size)

# 3. convert every hdf5 into a per-task zarr replay buffer
python scripts/convert_libero_dataset.py            # all demos per task
python scripts/convert_libero_dataset.py -n 50      # or subsample N demos per task

# 4. merge the 10 task zarrs into the multi-task dataset the configs expect
python scripts/compose_libero_multitask_dataset.py -mt libero10
```

Step 3 writes `data/libero/<TASK_NAME>_N<n_demo>.zarr` (it prompts before overwriting an existing
one). Step 4 shells out to [scripts/merge_data.py](scripts/merge_data.py) and produces
`data/libero/libero10_N500.zarr` (10 tasks × 50 demos). `-mt libero90` is also available.

Each episode stores `action`, `agentview_rgb`, `robot0_eye_in_hand_rgb`, `robot0_joint_pos`,
`robot0_eef_pos`, `robot0_eef_quat` (converted from axis-angle to quaternion),
`robot0_gripper_qpos`, `prompt` and `task_uid`. Expect ~3.4 GB for `libero10_N500.zarr`.

> **`training.num_demo` is a filename token, not a subsampling knob.** Configs resolve
> `zarr_path: data/libero/libero10_N${training.num_demo}.zarr`, so `training.num_demo=500` simply
> means "load `libero10_N500.zarr`". To train on fewer demos, build a smaller zarr (step 3 `-n`)
> and pass the matching `training.num_demo`. Independently, `ZarrDataset` accepts
> `max_train_episodes` to cap training episodes at load time — it is not in the configs, so add it
> with `+task.policy.dataset.max_train_episodes=100`.

### 3.2 MimicGen / robomimic

```bash
python scripts/convert_mimicgen_to_zarr.py \
    --hdf5_path data/robomimic/datasets/stack_three_d1/stack_three_d1.hdf5 \
    --zarr_path data/mimicgen/stack_three_d1_N100.zarr \
    -n 100
```

Then use the `task/tokenizer=mimicgen/stack_three_d1` / `task/policy=mimicgen/stack_three_d1`
config groups (or `--config-name=train_oattok_mimicgen` / `train_oatpolicy_mimicgen`).

---

## 4. Running experiments

### 4.1 The entry point

```bash
python scripts/run_workspace.py --config-name=<config> [hydra overrides...]
```

`<config>` is any file in [oat/config/](oat/config/) without the `.yaml` suffix. Multi-GPU uses
`accelerate` — all workspaces build their own `Accelerator`, so a plain `accelerate launch` in
front is the only change:

```bash
HYDRA_FULL_ERROR=1 accelerate launch \
    --num_machines 1 --multi_gpu --num_processes 4 \
    scripts/run_workspace.py --config-name=train_oattok ...
```

Each run creates `output/<YYYYMMDD>/<HHMMSS>_<name>_<task_name>_N<num_demo>/` containing:

```
.hydra/config.yaml             resolved config (analysis/eval scripts read this back)
checkpoints/latest.ckpt        rolling checkpoint, written every training.checkpoint_every epochs
checkpoints/ep-XXXX_*.ckpt     top-k checkpoints selected by checkpoint.topk.monitor_key
logs.json                      per-epoch metric log (one JSON object per line)
run_workspace.log              Hydra job log
wandb/, media/                 wandb run dir and rollout videos
```

Checkpoints are self-contained: they embed the full `cfg` plus all state dicts, so
`BasePolicy.from_checkpoint(path)` / `OATTok.from_checkpoint(path)` rebuild the workspace, the
model and the EMA weights (EMA weights are returned whenever `training.use_ema=true`, the default)
without any config on the side.

**Resume** is automatic: `training.resume=true` (default) makes a workspace pick up
`<output_dir>/checkpoints/latest.ckpt` if it exists. Because the output dir is timestamped,
resuming means re-launching with `hydra.run.dir=<the original run dir>` — otherwise you get a fresh
run. If the checkpoint's epoch already reached `training.num_epochs`, the workspace prints a
message and exits immediately.

### 4.2 Stage 1 — train an action tokenizer

```bash
HYDRA_FULL_ERROR=1 accelerate launch \
    --num_machines 1 --multi_gpu --num_processes 4 \
    scripts/run_workspace.py \
    --config-name=train_oattok_so3aug \
    task/tokenizer=libero/libero10 \
    training.num_epochs=5001 \
    training.num_demo=500
```

Single GPU is just `python scripts/run_workspace.py --config-name=train_oattok_so3aug ...`.
Top-k checkpoints are selected on `test_reconst_mse` (lower is better), producing files like
`ep-0900_mse-0.003.ckpt`.

Available tokenizer configs:

| Config | Workspace | `horizon` | Notes |
|---|---|---|---|
| `train_oattok` | `train_oattok.TrainOATTokWorkspace` | 16 | Baseline: RegisterEncoder → FSQ `[8,5,5,5,5]` → SinglePassDecoder |
| `train_oattok_so3aug` | same | 16 | **Main method.** Adds `SO3ActionChunkAug` (rotation-only, `p=0.6`, `max_angle_deg=30`) |
| `train_oattok_aug` | same | 32 | Earlier, non-SO(3) augmentation |
| `train_oattok_vq` / `train_oattok_bsq` / `train_oattok_spectral` | same | 16 / 32 / 32 | Quantizer ablations |
| `train_oattok_so3aug_vq` / `train_oattok_so3aug_bsq` | same | 16 | SO(3) aug × quantizer ablations |
| `train_oattok_mimicgen` | same | 32 | Defaults to `task/tokenizer=mimicgen/stack_three_d1` |
| `train_a2lextok`, `train_zhilltok`, `train_spqtok` | same | 32 | Alternative lattice / ordering quantizers |
| `train_residual_oattok` | `train_residual_oattok.TrainResidualOATTokWorkspace` | 32 | Residual tokenizer; uses `libero10_with_past` |
| `train_bintok` | `train_bintok.TrainBinTokWorkspace` | 32 | Fit-free uniform binning (256 bins); one pass, then saves |
| `train_bintok_so3aug` | `train_bintok_so3aug.TrainBinTokSO3AugWorkspace` | 32 | Binning + SO(3) aug |
| `train_fasttok` | `train_fasttok.TrainFASTTokWorkspace` | 32 | Wraps `physical-intelligence/fast` (downloads from HF); also writes `checkpoints/my_fast/` |
| `train_questtok` | `train_questtok.TrainQueSTTokWorkspace` | 32 | QueST-style tokenizer |

### 4.3 Stage 2 — train a policy

Token-based policies declare `policy.action_tokenizer.checkpoint: ???`, i.e. Hydra *requires* you to
supply the Stage-1 checkpoint path; the run aborts with a missing-mandatory-value error otherwise.

```bash
HYDRA_FULL_ERROR=1 MUJOCO_GL=egl accelerate launch \
    --num_machines 1 --multi_gpu --num_processes 4 \
    scripts/run_workspace.py \
    --config-name=train_oatpolicy_with_enriched_past \
    task/policy=libero/libero10_with_past \
    training.num_epochs=5001 \
    training.num_demo=500 \
    task.policy.lazy_eval=false \
    policy.action_tokenizer.checkpoint=/abs/path/to/output/<run>/checkpoints/ep-0900_mse-0.003.ckpt
```

Curated frozen tokenizers ship in [tok_ckpt/](tok_ckpt/) (`oat_s`, `oat_m`, `oat_l`, `oat_xl`,
`bsq`, `spectral`, `spectral_wo_norm`) if you want to skip Stage 1 while iterating on policies.

| Config | Policy class | `horizon` / `n_action_steps` | Tokenizer ckpt |
|---|---|---|---|
| `train_oatpolicy` | `OATPolicy` | 16 / 8 | yes (`OATTok`) |
| `train_oatpolicy_with_enriched_past` | `OATPolicyWithEnrichedPast` | 16 / 8, `past_n=7` | yes (`OATTok`) |
| `train_oatpolicy_with_past` | `OATPolicyWithPast` | 32 / 16, `past_n=7` | yes |
| `train_oatpolicy_with_round_past` | `OATPolicyWithRoundPast` | 32 / 16, `past_n=7` | yes |
| `train_oatpolicy_with_kinematics` | `OATPolicyWithKinematics` | 32 / 16, `past_n=3` | yes |
| `train_oatpolicy_with_residual` | `OATPolicyWithResidual` | 32 / 16, `past_n=7` | yes (`ResidualOATTok`) |
| `train_oatpolicy_gsl` | `GaussianSoftLabelPolicy` | 32 / 16 | yes |
| `train_oatpolicy_mimicgen` | `OATPolicy` on MimicGen | 32 / 16 | yes |
| `train_maskgitpolicy` | `MaskGITPolicy` (non-AR decoding) | 32 / 16 | yes |
| `train_binpolicy` | `BinPolicy` | 32 / 16 | yes (`BinTok`) |
| `train_fastpolicy` | `FASTPolicy` | 32 / 16 | yes (`FASTTok`) |
| `train_questpolicy` | `QueSTPolicy` | 32 / 16 | yes (`QueSTTok`) |
| `train_diffpolicy` | `DiffusionTransformerPolicy` | 16 / 8 | **no** |
| `train_diffpolicy_with_enriched_past` | diffusion + enriched past | 16 / 8, `past_n=7` | no |
| `train_flowpolicy` | `FlowPolicy` | 16 / 8 | no |
| `train_flowpolicy_with_enriched_past` | flow matching + enriched past | 16 / 8, `past_n=7` | no |
| `train_actpolicy` | `ACTPolicy` | 16 / 8, `n_obs_steps=1` | no |

> **The policy's `horizon` must equal the tokenizer's `horizon`.** The tokenizer decodes a fixed
> chunk length. `train_oattok` / `train_oattok_so3aug` default to 16 (pairing with `train_oatpolicy`
> and `train_oatpolicy_with_enriched_past`), while most other tokenizers default to 32 (pairing with
> the 32-horizon policies). Mixing them mismatches shapes — override `horizon=` on the *tokenizer*
> run to match the policy you intend to train.

### 4.4 In-training sim evaluation

`task.policy.lazy_eval` controls rollouts during training:

- `lazy_eval=true` (config default) — no env is constructed; training is pure supervised learning.
  Note the consequence: the top-k monitor key is `mean_success_rate`, which then never appears, so
  **only `latest.ckpt` is written**. Evaluate offline with `scripts/eval_policy_sim.py`.
- `lazy_eval=false` — rank 0 rolls out every `training.rollout_every` epochs (default 200) with
  `n_test=500` episodes spread over the 10 LIBERO-10 tasks, `n_test_vis=20` videos,
  `n_parallel_envs=20`, `max_episode_steps=550`. This is what the Slurm templates use, and what
  populates the `ep-XXXX_sr-0.XXX.ckpt` top-k files.

Useful runner overrides: `task.policy.env_runner.n_test=100`,
`task.policy.env_runner.n_parallel_envs=16`, `task.policy.env_runner.n_test_vis=4`.

### 4.5 Offline evaluation

```bash
# standard success-rate eval (a .ckpt file, or a directory — every .ckpt in it except latest.ckpt)
MUJOCO_GL=egl python scripts/eval_policy_sim.py \
    -c output/<run>/checkpoints/ep-0800_sr-0.656.ckpt \
    -o output/eval_metrics/<name> \
    -n 3 -d cuda:0

# same, plus velocity / acceleration / jerk RMS and SPARC smoothness metrics
MUJOCO_GL=egl python scripts/eval_policy_sim_smoothness.py -c <ckpt> -o <outdir>

# same, plus command legality / joint feasibility / realized EE motion
MUJOCO_GL=egl python scripts/eval_policy_sim_feasibility.py -c <ckpt> -o <outdir> \
    --enable-singularity --enable-collision

# tokenizer-only smoothness on the validation set (no policy, no sim)
python scripts/eval_tokenizer_smoothness.py \
    --ckpt-dir output/<tokenizer-run-dir> --ckpt-name latest.ckpt --device cuda:0 --fs 20.0
```

All three policy eval scripts share `-n/--num_exp` (repeat the eval and report mean/std/stderr) and
`-d/--device`, and write `eval_log.json` plus rollout videos into `-o`. The smoothness and
feasibility variants swap the env runner **in memory** after loading the checkpoint config; nothing
on disk is modified. `eval_policy_sim.py` prompts before overwriting an existing output dir.

The inference knobs `--temperature`, `--topk` and `--use_k_tokens` (decode with only the first *k*
of the ordered tokens — the Matryoshka property) apply to the token-based policies; the continuous
baselines (diffusion / flow / ACT) do not accept them, so leave them off for those checkpoints.

### 4.6 Analysis scripts

[analysis_scripts/](analysis_scripts/) holds offline diagnostics that read run dirs / checkpoints and
write plots and JSON. Each file's module docstring has its exact usage. Highlights:

| Script | Question it answers |
|---|---|
| `analyze_reconstruction_quality.py` | How much action error comes from the tokenizer vs from token prediction |
| `compare_oat_vs_enriched.py` | Baseline OAT vs enriched-past policy under rollout-like conditions |
| `compare_past_conditions.py` | Sensitivity to which past signals are fed in |
| `compare_rollout_mse_sim.py` | Closed-loop rollout MSE, in sim |
| `eval_tokenizer_metrics.py` | Token prediction accuracy, entropy, codebook usage, adjacent-chunk overlap |
| `analyze_entropy.py`, `codebook_utilization.py`, `token_sequence_stats.py` | Codebook/entropy statistics, incl. multi-tokenizer comparison |
| `diagnose_latent_dis.py`, `diagnose_per_pos_mse.py` | Latent distribution and per-position reconstruction error |
| `measure_fmax.py`, `phase1_real_data.py` | Bandwidth (`f_max`) of LIBERO delta-actions, supporting the past-horizon choice |
| `polar_analysis.py`, `diagnose_polar_policy.py` | Polar-decomposition action-space analysis |

### 4.7 Slurm

[slurm/](slurm/) has one directory per method (`oat`, `fast`, `quest`, `dp`, `act`, `bin`), each with
`train_tok_libero10.slurm` and/or `train_ply_libero10.slurm`. They are **templates**: fill in
`--account`, `--partition`, `--output`, `--error`, `--mail-user`, the `cd /path/to/project/oat`
line, and the `policy.action_tokenizer.checkpoint=...` value before `sbatch`. Defaults are 1 node ×
4 GPUs, 48 h, 128 GB (tokenizer) / 256 GB (policy), 16 CPUs, launched via `uv run accelerate launch`.

---

## 5. Tests and smoke checks

```bash
pytest test_a2lex.py test_zhill.py test_trajectory_smoothness.py   # 25 CPU-only tests
python scripts/smoke_test_so3_action_chunk_aug.py                  # SO(3) augmentation invariants
```

Per [AGENTS.md](AGENTS.md), run the pytest set before touching quantizers or evaluation logic.

---

## 6. Common overrides

| Override | Effect |
|---|---|
| `task/tokenizer=libero/libero10` · `task/policy=libero/libero10` | Select the task group (the `_with_past` variants swap in `ZarrDatasetWithPastAction`) |
| `training.num_demo=500` | Selects `data/libero/libero10_N500.zarr` and the run name |
| `training.num_epochs=5001` | Length of training |
| `training.checkpoint_every=10` · `val_every` · `rollout_every` | Cadence of checkpointing / validation / sim rollouts |
| `training.allow_bf16=false` | Disable bf16 mixed precision (auto-detected otherwise) |
| `training.resume=false` | Ignore an existing `latest.ckpt` |
| `dataloader.batch_size=` · `dataloader.num_workers=` | Throughput knobs (tokenizers default 256, policies 64) |
| `optimizer.learning_rate=` (tokenizer) · `optimizer.policy_lr=` / `optimizer.obs_enc_lr=` (policy) | LRs; the observation encoder is deliberately trained 5× slower |
| `horizon=` | Action chunk length — **must match between tokenizer and policy** |
| `past_n=` | Number of past action steps fed to the `*_with_*_past` policies |
| `task.policy.lazy_eval=false` | Enable sim rollouts during training |
| `policy.temperature=` · `policy.topk=` | AR sampling at inference |
| `logging.mode=offline` · `logging.project=` | wandb behaviour |
| `hydra.run.dir=output/...` | Pin the run directory (required to resume an existing run) |

---

## 7. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `ModuleNotFoundError: libero` | Submodule not initialised, or LIBERO not installed editable: `git submodule update --init --recursive && pip install -e third_party/LIBERO` |
| `[Warning]: bddl_files path ... does not exist` | Stale `~/.libero/config.yaml` — delete it and re-import `libero.libero` |
| `MISSING mandatory value: policy.action_tokenizer.checkpoint` | Token-based policy launched without `policy.action_tokenizer.checkpoint=<path>` |
| `FileNotFoundError: data/libero/libero10_N500.zarr` | Dataset not built, or `training.num_demo` doesn't match the zarr filename (§3.1) |
| MuJoCo / EGL / GLFW errors during rollout | `export MUJOCO_GL=egl`; on a node without EGL, install `libosmesa6-dev` and use `MUJOCO_GL=osmesa` |
| Rollout OOM or GPU context crashes | Lower `task.policy.env_runner.n_parallel_envs` (20 → 16 or 8) and `n_test` |
| Only `latest.ckpt` appears, no `ep-*_sr-*.ckpt` | `lazy_eval=true`, so `mean_success_rate` is never logged and top-k never fires (§4.4) |
| Training exits immediately after startup | The resumed checkpoint already reached `training.num_epochs` — raise it or start a fresh run dir |
| Shape mismatch when loading a tokenizer into a policy | `horizon` (or `action_dim`) differs between the two stages (§4.3) |
| Hydra swallows the traceback | Prefix the command with `HYDRA_FULL_ERROR=1` |
| NCCL timeout during in-training eval | Sim eval runs on rank 0 only; the process-group timeout is already raised to 2 h, but a very large `n_test` on slow nodes can still exceed it — reduce `n_test` or raise `rollout_every` |
