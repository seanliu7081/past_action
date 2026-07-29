# ACT baseline (Action Chunking with Transformers) on LIBERO-10 — implementation plan

Adds ACT (Zhao et al. 2023, [tonyzhaozh/act](https://github.com/tonyzhaozh/act)) as an
additional baseline alongside diffpolicy/flowpolicy/fastpolicy. Deliverables: a train
config in `oat/config`, a policy in `oat/policy`, following oat conventions.
**Hard constraint: no existing file is modified — new files only.**

**Environment: NO new conda env, NO new packages.** The active uv venv
(`/workspace/oat/.venv`) already has everything ACT needs (torch 2.10.0+cu128,
torchvision 0.25.0, numpy, einops), and the ImageNet ResNet-18 weights are already cached
at `~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth`. Use torchvision's `weights=`
API (the `pretrained=` kwarg ACT uses is removed in modern torchvision).

**Confirmed decisions:** (1) faithful ACT vision path — vendored DETR backbone with raw
images → ResNet18 spatial tokens (not the oat `RobomimicRgbEncoder`); (2) chunking matched
to other baselines — `horizon=16` (= ACT chunk/num_queries), `n_action_steps=8` receding
horizon; (3) `n_obs_steps=1` (ACT is single-frame).

## Verified constraints that shape the design

- All policies share `oat.workspace.train_policy.TrainPolicyWorkspace`; hydra-instantiated
  from `cfg.policy._target_`; no `__init__.py` anywhere (namespace packages) — no
  registration needed.
- Required API (template: `oat/policy/flow_policy.py`): `forward(batch)` → **scalar**
  loss; `predict_action(obs_dict, **kwargs)` → `{'action': [B,n_action_steps,7],
  'action_pred': [B,horizon,7]}` un-normalized; `get_optimizer(policy_lr, obs_enc_lr,
  weight_decay, betas)`; `set_normalizer`; `get_observation_encoder/modalities/ports`;
  `get_policy_name`; `create_dummy_observation`; `reset()`.
- **DDP uses `find_unused_parameters=False`** (`oat/workspace/train_policy.py:74`).
  Upstream ACT has two sources of parameters with no gradient: the unused `is_pad_head`,
  and decoder layers 2–7 — verified in the ACT source that `detr_vae.py:131` takes
  `hs[0]`, the FIRST decoder layer's output (known upstream bug, also documented by
  lerobot). Fix: drop `is_pad_head`; default `dec_layers=1` (mathematically identical to
  upstream 7-layer + `hs[0]`, and DDP-safe); keep the `[0]` indexing with an explanatory
  comment.
- Batch: rgb ports `[B,To,128,128,3]` uint8 0-255 train / float32 0-255 eval; state ports
  `robot0_eef_pos[3] + robot0_eef_quat[4] + robot0_gripper_qpos[2] + task_uid[1]` →
  proprio dim 10 (`task_uid` int64; `LinearNormalizer._normalize` auto-casts dtype);
  action `[B,16,7]`. oat dataset windows are edge-padded to full length ⇒ `is_pad` is
  all-False.
- EMA (`use_ema: True`) deepcopies the policy and syncs **parameters only** — safe because
  ACT's only buffers (FrozenBN stats, sinusoid `pos_table`, ImageNet mean/std) are static.
- `forward` also runs under `torch.inference_mode()` (validation) — CVAE reparametrize
  path allocates only, safe.

## New files (nothing existing is touched)

| # | Path | Content |
|---|------|---------|
| 1 | `oat/model/act/transformer.py` | DETR `Transformer` + encoder/decoder/layers, near-verbatim from ACT `detr/models/transformer.py`; drop `IPython` and argparse `build_transformer`. Keep `return_intermediate_dec` and seq-first layout exactly. |
| 2 | `oat/model/act/backbone.py` | Merge of ACT `backbone.py` + `position_encoding.py` (sine only): `Backbone` (torchvision `resnet18(weights=ResNet18_Weights.IMAGENET1K_V1, norm_layer=FrozenBatchNorm2d)` — use `torchvision.ops.FrozenBatchNorm2d`, numerically identical; weights stay trainable), `IntermediateLayerGetter({'layer4': "0"})`, `PositionEmbeddingSine(hidden//2, normalize=True)`, `Joiner`, `build_backbone(hidden_dim, name)`. Drop `util.misc` (`NestedTensor` annotations, `is_main_process`), dilation/masks plumbing. |
| 3 | `oat/model/act/detr_vae.py` | `DETRVAE` (CVAE) + `get_sinusoid_encoding_table` + `kl_divergence` + `build_cvae_encoder` + `build_act_model(...)` (explicit-kwargs builder replacing the sys.argv-parsing `build_ACT_model_and_optimizer`). Parameterize all hardcoded 14s into `proprio_dim`/`action_dim`; `latent_dim` arg (default 32); delete `is_pad_head`, `env_state` branch, `CNNMLP`; `reparametrize` via `torch.randn_like` (no `Variable`); inference `z=0` tensor takes `qpos.dtype` (bf16-friendly); keep single shared backbone across cameras, width-concat, `additional_pos_embed(2, hidden)`, and the `hs[0]` quirk (loud comment). Returns `a_hat, (mu, logvar)`. |
| 4 | `oat/policy/actpolicy.py` | `ACTPolicy(BasePolicy)` — oat adapter (below). |
| 5 | `oat/config/train_actpolicy.yaml` | Train config (below). |
| 6 | optional: `slurm/act/train_ply_libero10.slurm` | Copy of `slurm/dp/train_ply_libero10.slurm` with `--config-name=train_actpolicy`, job name `actply`. |

Each model file gets a provenance docstring: "Adapted from ACT (Zhao et al. 2023, MIT;
derives from DETR, Apache-2.0). Modifications: …" — matching the repo's convention of
first-party model code under `oat/model/<family>/` (like `oat/model/diffusion/`).

## `oat/policy/actpolicy.py` design

Style-match `oat/policy/flow_policy.py` (section dividers, shape_meta parse boilerplate,
param-count report, docstring documenting training/sampling math and oat adaptations).
No `obs_encoder` constructor arg — the DETR backbone is the vision path.

- `__init__(shape_meta, horizon, n_action_steps, n_obs_steps, hidden_dim=512,
  dim_feedforward=3200, enc_layers=4, dec_layers=1, nheads=8, dropout=0.1, latent_dim=32,
  kl_weight=10.0, backbone='resnet18', pre_norm=False, temporal_ensemble=False,
  temporal_ensemble_k=0.01)`. Parse shape_meta → `rgb_ports` (2), `state_ports` (4,
  proprio dim 10), `action_dim` 7; `self.model = build_act_model(...)` with
  `num_queries=horizon`; ImageNet mean/std as non-persistent buffers;
  `self.normalizer = LinearNormalizer()`.
- Preprocessing (handles uint8-train / float32-eval uniformly):
  - images: per rgb port take `[:, -1]` (last frame), HWC→CHW, `.to(float).div(255)`,
    ImageNet normalize, stack → `[B, num_cam, 3, 128, 128]`.
  - proprio: `self.normalizer[port].normalize(obs[port][:, -1])` per state port
    (incl. task_uid), concat → `[B, 10]`.
- `forward(batch)`: normalize action with `self.normalizer['action']`; `is_pad` all-False
  (oat edge-pads windows); model forward with CVAE encoder; loss = masked-L1 (keep
  upstream masked-mean expression) + `kl_weight * kl_divergence(mu.float(),
  logvar.float())` (fp32 KL for bf16 stability) → scalar.
- `predict_action(obs_dict, **kwargs)`: model forward without actions (z=0,
  deterministic); `action_pred = normalizer['action'].unnormalize(a_hat)` `[B,16,7]`;
  `action = action_pred[:, :n_action_steps]`. Optional temporal-ensemble branch (default
  off, asserts `n_action_steps==1`): keep a history of past chunks (≤ horizon), order
  OLDEST-first, and weight chunk i's action for the current step by `exp(-k*i)` — i.e. the
  OLDEST surviving prediction gets the HIGHEST weight, exactly as ACT
  `imitate_episodes.py:256` does (`exp_weights = np.exp(-k * np.arange(n))` over rows
  ordered by prediction time ascending, k=0.01). Cleared by `reset()`.
- `get_optimizer(policy_lr, obs_enc_lr, weight_decay, betas)`: faithful ACT 2-group split
  keyed on `'backbone'` in parameter name — backbone at `obs_enc_lr`, rest at `policy_lr`,
  `weight_decay` on all groups (deviation from oat's 4-group no-decay pattern; faithful to
  upstream, commented).
- `get_observation_modalities() → ['rgb','state']`; `get_observation_ports() → rgb+state`;
  `get_policy_name() → 'actpolicy_rgb'`; `get_observation_encoder() →
  self.model.backbone` (no runtime caller — API completeness); `create_dummy_observation`
  delegating like flow_policy; `set_normalizer` loads state dict only (no obs_encoder to
  forward to); `reset()` clears the ensemble history.

## `oat/config/train_actpolicy.yaml` design

Skeleton copied from `train_flowpolicy.yaml`; blocks `ema / logging / checkpoint /
multi_run / hydra` verbatim. Differences:

```yaml
defaults: [_self_, task/policy: libero/libero10]
name: train_actpolicy
_target_: oat.workspace.train_policy.TrainPolicyWorkspace
seed: 42
horizon: 16          # == ACT chunk size (num_queries)
n_action_steps: 8    # receding horizon, matches diffpolicy/flowpolicy
n_obs_steps: 1       # ACT is single-frame
policy:
  _target_: oat.policy.actpolicy.ACTPolicy
  shape_meta/horizon/n_action_steps/n_obs_steps: ${...}
  hidden_dim: 512, dim_feedforward: 3200, enc_layers: 4,
  dec_layers: 1      # upstream configures 7 but only layer 1 is used/trained (hs[0] bug)
  nheads: 8, dropout: 0.1, latent_dim: 32, kl_weight: 10.0,
  backbone: resnet18, pre_norm: False,
  temporal_ensemble: False, temporal_ensemble_k: 0.01
training:  # same as flowpolicy: num_epochs 5001, use_ema True, num_demo 500,
           # lr_scheduler constant (ACT: no scheduler), max_grad_norm 1.0
optimizer:
  policy_lr: 1.0e-4    # upstream 1e-5 @ batch 8; scaled for batch 64
  obs_enc_lr: 1.0e-5   # → ResNet backbone param group (ACT lr_backbone)
  weight_decay: 1.0e-4 # upstream default, applied to all params
  betas: [0.9, 0.999]  # AdamW default, as upstream (oat baselines use [0.9, 0.95])
dataloader / val_dataloader: batch_size 64, num_workers 4, (rest as flowpolicy)
```

Interpolation verified: task yaml picks up `n_obs_steps=1` (`pad_before=0`, fine in
`SequenceSampler.create_indices`), `dataset.n_action_steps=${horizon}=16`,
`env_runner n_obs_steps=1 / n_action_steps=8` (fine in `MultiStepWrapper`).

## Implementation order

1. `oat/model/act/transformer.py` → 2. `oat/model/act/backbone.py` →
3. `oat/model/act/detr_vae.py` → 4. `oat/policy/actpolicy.py` →
5. `oat/config/train_actpolicy.yaml` → 6. optional slurm file → 7. verification.

## Verification

1. Import check: `cd /workspace/oat && uv run python -c "import oat.policy.actpolicy"`
   (+ model modules).
2. Smoke script (outside the repo): hydra-compose `train_actpolicy` (register
   `oat.common.hydra_util.register_new_resolvers` first), instantiate policy + real
   `ZarrDataset` (`data/libero/libero10_N500.zarr`), `set_normalizer`, one real batch:
   assert finite scalar loss, `loss.backward()` leaves **no param with `.grad is None`**
   (catches the DDP unused-param class on 1 GPU); `predict_action` shapes
   `[B,8,7]`/`[B,16,7]` with sane un-normalized magnitudes; `copy.deepcopy` + one
   `EMAModel.step`; dummy-obs path; bf16 autocast forward finite.
3. Training dry-run (single GPU, offline, no rollout):
   `HYDRA_FULL_ERROR=1 uv run accelerate launch --num_processes 1 scripts/run_workspace.py
   --config-name=train_actpolicy training.num_epochs=2 training.max_train_steps=10
   training.max_val_steps=2 training.max_reconst_steps=2 dataloader.batch_size=8
   val_dataloader.batch_size=8 logging.mode=offline task.policy.lazy_eval=true
   training.resume=false`
4. Full train: `MUJOCO_GL=egl uv run accelerate launch --num_machines 1 --multi_gpu
   --num_processes 4 scripts/run_workspace.py --config-name=train_actpolicy
   training.num_epochs=5001 training.num_demo=500 task.policy.lazy_eval=false`
5. Eval: `MUJOCO_GL=egl uv run python scripts/eval_policy_sim.py -c <ckpt> -o <outdir>`
   (`from_checkpoint` returns the EMA model; runner filters ports via
   `get_observation_ports()`).

## Fidelity audit vs. ACT GitHub source (verified line-by-line)

Every model file in the ACT repo was read in full and compared against this design.
Line references below are into [tonyzhaozh/act](https://github.com/tonyzhaozh/act)
@ `742c753` (2024-01-28).

**Exact matches (kept verbatim or with pure-refactor changes):**
- CVAE encoder (`detr/models/detr_vae.py:88-110`): `[CLS, Linear(qpos), Linear(actions)]`
  seq-first, fixed sinusoid `pos_table` buffer of size `1+1+num_queries`
  (`.clone().detach()` per forward), key-padding mask `[False,False]+is_pad`, CLS output →
  `latent_proj` → `mu/logvar` split at `latent_dim=32`.
- `reparametrize` (`std=exp(logvar/2); mu+std*eps`) — `torch.randn_like` replaces the
  deprecated `Variable(...normal_())`, identical distribution.
- Inference latent = **zeros** (prior mean, not sampled) → `latent_out_proj`
  (`detr_vae.py:111-114`).
- `kl_divergence`: scalar form `== total_kld[0]` (`klds.sum(1).mean(0)`); the 4-D branch
  never triggers for 2-D mu (`policy.py:71-84`).
- Loss (`policy.py:30-34`): `l1 = (l1_all * ~is_pad).mean()` (full-count mean) `+
  kl_weight * kl`; expression kept verbatim (all-False is_pad ⇒ == plain mean).
- Backbone (`detr/models/backbone.py`): resnet18, FrozenBatchNorm2d —
  `torchvision.ops.FrozenBatchNorm2d` has the SAME eps-before-rsqrt formula (eps=1e-5) and
  the same `num_batches_tracked` deletion; ImageNet-pretrained (`is_main_process()`≡True →
  same `resnet18-f37072fd.pth` weights); ALL backbone weights trainable (upstream's freeze
  loop is commented out) at the backbone lr; `IntermediateLayerGetter({'layer4':"0"})`;
  `num_channels=512`; `Joiner` returning `([features],[pos.to(dtype)])`.
- `PositionEmbeddingSine(hidden//2, temperature=10000, normalize=True, scale=2π)`
  verbatim.
- DETRVAE decoder path (`detr_vae.py:116-139`): SHARED backbone + shared 1×1 `input_proj`
  across cameras, per-camera identical sine pos grid, width-concat (`axis=3`),
  latent+proprio tokens prepended with learned `additional_pos_embed(2,hidden)`,
  `tgt=zeros`, learned `query_embed(num_queries)`, `hs[0]`, `action_head=Linear`.
- Transformer (`detr/models/transformer.py`) copied verbatim: post-norm layers, pos added
  to Q/K only, xavier init on dim>1 params, `return_intermediate_dec=True`, decoder final
  LayerNorm.
- Optimizer (`detr/main.py:80-88`): AdamW, 2 groups split on `"backbone"` substring, wd on
  both groups, betas (0.9, 0.999) — mapped to oat's
  `policy_lr/obs_enc_lr/weight_decay/betas`.
- Image pipeline: `/255` then ImageNet `Normalize` (upstream splits this between dataset
  and policy; here both in `_preprocess_images` — same math).
- Hyperparameters: hidden 512, ffn 3200, enc 4, heads 8, dropout 0.1, kl_weight 10,
  latent 32, backbone lr 1e-5, wd 1e-4 — all README/source values.
- `dec_layers=1 ≡` upstream `dec_layers=7`: confirmed at source level — the decoder
  appends `norm(output)` after EVERY layer (`transformer.py:129-136`), so `hs[0]` is
  `norm(out of layer 1)` regardless of how many layers follow; layers 2-7 receive zero
  gradient.
- Temporal ensembling (`imitate_episodes.py:250-259`): weights `exp(-k·i)/Σ`, i ordered by
  prediction time ascending ⇒ OLDEST prediction weighted highest, k=0.01.

**Deliberate deviations (each documented in the policy docstring):**
1. `dec_layers=1` default + `is_pad_head` removed — zero functional change (above), needed
   for DDP `find_unused_parameters=False`; `dec_layers` stays configurable.
2. Action/proprio normalization: oat `LinearNormalizer` limits→[-1,1] instead of ACT's
   dataset mean/std z-score (oat framework convention; stats travel in the checkpoint).
3. `is_pad` all-False: oat windows are edge-padded to full length (no pad mask exists);
   ACT zero-pads and masks. The masked-mean expression is kept so a future mask just
   works.
4. Proprio = eef_pos+quat+gripper+task_uid (10-d) vs ALOHA's 14-d joint qpos — LIBERO port
   necessity; task_uid included for task-conditioning parity with the other oat baselines.
5. chunk 16 / execute 8 vs paper chunk 100 open-loop; 2×128×128 cameras vs 1×480×640 —
   environment/comparability choices, mechanism unchanged.
6. Workspace-level: EMA on, grad-clip 1.0, constant lr + 100-step warmup (upstream: none).
   All config-overridable (`training.use_ema=False` etc.).
7. `policy_lr=1e-4` vs upstream 1e-5 — scaled for batch 64 vs 8; set
   `optimizer.policy_lr=1e-5` for the literal paper value.

## Risks (handled)

- DDP unused params → `is_pad_head` dropped, `dec_layers=1` default (see above).
- bf16: FrozenBN fp32 buffers promote; sine pos emb cast via `Joiner`; KL upcast to fp32;
  z=0 takes input dtype.
- task_uid int64 (train) vs uint8→float32 (eval): normalizer casts; limits-fit maps
  global uids consistently.
- Deviations from upstream (documented in policy docstring): LinearNormalizer [-1,1]
  instead of mean/std stats; is_pad all-False (edge-padded windows); EMA + grad-clip 1.0 +
  constant-lr warmup from the shared workspace (all config-overridable, e.g.
  `training.use_ema=False`).
- Geometry: 128×128 → 4×4 tokens/cam → 32 image + 2 extra memory tokens; CVAE encoder seq
  1+1+16=18. Model ≈ 55M params (11M backbone).
