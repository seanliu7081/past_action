# FlowPolicyWithEnrichedPast — Architecture Reference (LIBERO-10)

## TL;DR

`FlowPolicyWithEnrichedPast` (`oat/policy/flow_policy_with_enriched_past.py`) is a **Transformer-based rectified flow-matching action-chunk policy** for the LIBERO-10 manipulation benchmark. It encodes two RGB cameras plus proprioceptive state into per-timestep observation tokens, augments them with an **"enriched past"** conditioning sequence built from the robot's recent action history (acceleration, jerk, and the raw last-7 actions), and uses a `TransformerForDiffusion` as a velocity field. Unlike a standard diffusion policy, the flow source is **not** pure Gaussian noise but a **warm-start prior** anchored on the most recent past action. At inference it integrates the learned velocity field with 10 forward-Euler ODE steps to produce a 16-step action chunk, executes the first 8 steps (receding horizon), and rolls a past-action buffer forward across the episode. The training `forward(batch)` *is* the loss: a plain MSE on the straight-line velocity target.

---

## Config at a glance

Source: `oat/config/train_flowpolicy_with_enriched_past.yaml` and `oat/config/task/policy/libero/libero10_with_past.yaml`.

| Hyperparameter | Value | What it controls |
|---|---|---|
| `horizon` (H) | 16 | Length of the generated/denoised action chunk = transformer decoder sequence length T. Also bound to the **dataset's** `n_action_steps` (`n_action_steps: ${horizon}`). |
| `n_action_steps` | 8 | Receding-horizon **execution** count; `predict_action` slices `action_pred[:, :8]`. Used only by the policy/env-runner, never by the dataset. |
| `n_obs_steps` (To) | 2 | Number of observation frames encoded into obs tokens. |
| `past_n` | 7 | Depth of the raw past-action history / inference past buffer. |
| action dim (A) | 7 | LIBERO-10 delta-pose (6) + gripper (1). |
| `embed_dim` (n_emb) | 256 | Transformer model width; also the projection-MLP output width and `cond_dim`. |
| `n_layers` | 4 | Transformer **decoder** layers. |
| `n_heads` | 4 | Attention heads. |
| `dropout` | 0.1 | `p_drop_emb` and `p_drop_attn`. |
| `num_inference_steps` (N) | 10 | Forward-Euler ODE steps at inference; `dt = 1/N = 0.1`. |
| `prior_noise_scale` (sigma) | 1.0 | Std of Gaussian added to the warm-start mean to form the flow source `x0`. |
| `_t_scale` | 1000.0 | Multiplier mapping flow time `t in [0,1]` into the `SinusoidalPosEmb` frequency range. |
| `policy_lr` | 5e-5 | LR for transformer + acc/jerk/raw projection MLPs. |
| `obs_enc_lr` | 1e-5 | LR for the observation encoder (5x lower). |
| `weight_decay` | 0.0 | AdamW weight decay (effectively makes decay/no-decay split cosmetic). |
| `betas` | [0.9, 0.95] | AdamW betas. |
| vision `crop_shape` | [76, 76] | Random (train) / center (eval) crop of the 128x128 RGB inputs. |
| EMA `power` | 0.75 | Overrides the `EMAModel` class default of 2/3. |
| `lazy_eval` | true | No env rollouts -> no `mean_success_rate` -> top-k checkpointing is a no-op. |

Derived at runtime (not a configured constant):
- `obs_feature_dim` = `cond_dim` = **138** = vision 128 + state 10.
- `max_cond_len` = `n_obs_steps + N_EXPLICIT_FEATURES + past_n` = 2 + 2 + 7 = **11** conditioning tokens.

---

## Observation & Action Spaces

`shape_meta` (LIBERO-10, from `libero10_with_past.yaml`):

| Key | Type | Per-frame shape | Notes |
|---|---|---|---|
| `agentview_rgb` | rgb | [128, 128, 3] | uint8 in [0, 255]; routed to vision encoder. |
| `robot0_eye_in_hand_rgb` | rgb | [128, 128, 3] | uint8 in [0, 255]; routed to vision encoder. |
| `robot0_eef_pos` | state | [3] | End-effector position. |
| `robot0_eef_quat` | state | [4] | End-effector orientation (quaternion). |
| `robot0_gripper_qpos` | state | [2] | Gripper joint positions. |
| `task_uid` | state | [1] | Task identifier, treated as a normalized state channel. |
| **`action`** | — | [7] | Target/output (1-D, asserted). Delta-pose + gripper. |

Each obs key arrives with a leading To=2 axis (e.g. `agentview_rgb` -> `(B, 2, 128, 128, 3)`, `robot0_eef_pos` -> `(B, 2, 3)`). State dim = 3 + 4 + 2 + 1 = **10**; vision dim = 64/camera x 2 cameras = **128**; fused obs feature dim = **138**.

---

## End-to-end Data Flow

This is the centerpiece. Shapes are for one batch of size B with To=2, H=16, A=7, past_n=7, d=138, n_emb=256.

```
                          RAW OBSERVATION DICT (per port: B x To=2 x ...)
  agentview_rgb (B,2,128,128,3)   robot0_eye_in_hand_rgb (B,2,128,128,3)
  robot0_eef_pos (B,2,3)  robot0_eef_quat (B,2,4)  robot0_gripper_qpos (B,2,2)  task_uid (B,2,1)
        |                                                            |
        |  (RGB path)                                                |  (state path)
        v                                                            v
  +-------------------------------+                      +----------------------------+
  | RobomimicRgbEncoder (per cam) |                      | ProjectionStateEncoder     |
  |  normalize (limits/[-1,1])    |                      |  normalize each port        |
  |  fold To->batch: (B*2,3,128,128)                     |  concat ports along -1      |
  |  CropRandomizer 76x76         |                      |  (B,2,3+4+2+1=10)           |
  |  ResNet18Conv (GroupNorm)     |                      |  nn.Identity (out_dim=null) |
  |  SpatialSoftmax 32 kp->64     |                      +----------------------------+
  |  Linear -> 64 / camera        |                                   |
  |  concat 2 cams: (B*2,128)     |                                   | state_feat (B,2,10)
  |  reshape: (B,2,128)           |                                   |
  +-------------------------------+                                   |
        |  vision_feat (B,2,128)                                      |
        +---------------------------+----------------------------------+
                                    v
                       FusedObservationEncoder: concat on dim=-1
                                    |
                            obs_features (B, To=2, 138)         past_action (B, past_n=7, 7)  [from dataset/buffer]
                                    |                                   |
                                    |        +-----------+--------------+----------------+
                                    |        | normalizer["action"].normalize -> norm_past (B,7,7)
                                    |        |     a_{t-1}=norm_past[:,-1]  a_{t-2}=[:,-2]  a_{t-3}=[:,-3]
                                    |        |     acc  = a_{t-1}-a_{t-2}            (B,7)
                                    |        |     jerk = a_{t-1}-2 a_{t-2}+a_{t-3}  (B,7)
                                    |        v
                                    |   acc_proj(acc) (B,138)  jerk_proj(jerk) (B,138)  raw_proj(norm_past) (B,7,138)
                                    |        |  stack -> explicit (B,2,138)              raw_feat (B,7,138)
                                    v        v                                          v
            _build_condition:  cat([ obs_features(2) | explicit(2) | raw_feat(7) ], dim=1)
                                    |
                          cond (B, 11, 138)   <-- the "enriched past" + obs conditioning
                                    |
   warm-start:  mu = normalize(past_action[:,-1]).broadcast -> (B, H=16, 7)
   source:      x0 = mu + sigma(1.0)*N(0,I)                  -> (B, 16, 7)
                                    |                                   |
                                    |   (TRAINING)                      |   (INFERENCE: x starts = x0)
                                    v                                   v
   xt = (1-t)*x0 + t*x1 (B,16,7), t~U(0,1)            Euler loop i=0..9, t=i*0.1:
                                    |                       x <- x + 0.1 * model(x, t*1000, cond)
                                    v                                   |
   +---------------------------------------------------------------+    |
   | TransformerForDiffusion  v_theta(x_t, scaled_t, cond)         |<---+
   |   input_emb: Linear(7->256)  sample (B,16,7)->(B,16,256)      |
   |   time_emb: SinusoidalPosEmb(t*1000) -> (B,1,256)             |
   |   cond_obs_emb: Linear(138->256) over 11 cond tokens          |
   |   memory = MLP_enc([time | cond]) (B,12,256) + cond_pos_emb   |
   |   tgt = input_emb + pos_emb[:16]  (B,16,256)                  |
   |   4x TransformerDecoderLayer (norm_first):                    |
   |       self-attn over 16 action tokens (no mask)               |
   |       cross-attn into 12-token memory (no memory_mask)        |
   |   ln_f -> head: Linear(256->7)                                |
   +---------------------------------------------------------------+
                                    |
                          v_pred (B, 16, 7)
                                    |
        (TRAINING)  loss = MSE(v_pred, v_target = x1 - x0)   [scalar]
        (INFERENCE) after N steps -> x (B,16,7)
                                    |
                  action_pred = normalizer["action"].unnormalize(x)  (B,16,7)
                                    |
                  action = action_pred[:, :n_action_steps=8]         (B,8,7)  -> executed
                                    |
                  past buffer <- action_pred[:, 1:8].detach()        (B,7,7)  -> next step's past_action
```

Key observation: the **past enters twice** — once as 9 of the 11 conditioning tokens (acc + jerk + 7 raw), attended via cross-attention, and once as the warm-start prior mean `mu`. This dual use is easy to miss.

---

## Perception stack

Source files: `oat/perception/fused_obs_encoder.py`, `oat/perception/robomimic_vision_encoder.py`, `oat/perception/state_encoder.py`.

### FusedObservationEncoder (`fused_obs_encoder.py:10`)
Parses `shape_meta['obs']` and buckets each port by `type` into `rgb_ports`, `state_ports`, `text_ports` (no text here). It instantiates sub-encoders itself (config sets `_recursive_: false`), injecting the shared `shape_meta`; each leaf re-parses and keeps only its own port type. `forward` runs the vision and state encoders and **concatenates their outputs along the last (feature) dim**: `torch.cat([vision_feat, state_feat], dim=-1)` -> `(B, To, 138)`. Fusion is pure last-dim concatenation, **not** per-key tokens — the To axis is the only token axis the downstream policy sees, and all modalities for a timestep are merged into one 138-d vector. `output_feature_dim()` sums sub-encoder dims (128 + 10 = 138).

### RobomimicRgbEncoder (`robomimic_vision_encoder.py:15`)
Per camera, builds a robomimic `VisualCore`:
- Backbone `ResNet18Conv` (truncated, no FC), `input_coord_conv=False`.
- Pooling `SpatialSoftmax` with `num_kp=32` -> 64 keypoint values.
- `feature_dimension=64`, `flatten=True` -> a Linear head fixes the per-camera output at **64-d** (independent of crop size).
- `share_rgb_model=False`: each of the 2 cameras has **independent** weights.

`CropRandomizer` (`crop_height=76, crop_width=76, num_crops=1`) crops 128x128 -> 76x76 before the CNN: a **random** crop in `training` mode, a deterministic **center** crop in eval. `use_group_norm=True` swaps every `BatchNorm2d` for `GroupNorm(num_groups=C//16)`, so there is no running-stats train/eval difference; the only train/eval difference in vision is random-vs-center crop. `forward` normalizes each RGB key via the `LinearNormalizer` (limits, not just /255), folds To into the batch (`B*To, 3, 128, 128`), runs crop+CNN, concatenates per-key features to `(B*To, 128)`, then reshapes to `(B, To, 128)`.

### ProjectionStateEncoder (`state_encoder.py:13`)
`out_dim=null` -> `nn.Identity` (no learnable projection); output dim = summed raw state dim = 3 + 4 + 2 + 1 = **10**. `forward` normalizes each state port, concatenates in `shape_meta` iteration order (`eef_pos`, `eef_quat`, `gripper_qpos`, `task_uid`) to `(B, To, 10)`, then applies the Identity.

---

## Backbone — TransformerForDiffusion

Source: `oat/model/diffusion/transformer_for_diffusion.py:10` (a GPT/BERT-style net adapted from Diffusion Policy). The flow policy instantiates it (`flow_policy_with_enriched_past.py:93`) with `cond_dim=138>0`, `time_as_cond=True`, `obs_as_cond=True`, `causal_attn=False`, and `n_cond_layers` defaulting to 0 -> the **encoder-decoder** branch.

**Structure:**
- `T = horizon = 16` action tokens; `T_cond = 1 (time) + max_cond_len (11) = 12` conditioning tokens. The constructor arg `n_obs_steps` is **overloaded**: the policy passes `max_cond_len=11`, not the policy's observation horizon (2).
- The conditioning "encoder" is **not a transformer** here — with `n_cond_layers=0` it is a token-wise MLP `Linear(256->1024) -> Mish -> Linear(1024->256)` producing the cross-attention `memory`. It has no attention.
- The decoder is `nn.TransformerDecoder` of **4** `nn.TransformerDecoderLayer`s: self-attention over the 16 action tokens + cross-attention into the 12-token memory + FFN. `nhead=4`, `d_model=256`, `dim_feedforward=4*256=1024`, `dropout=0.1`, `activation='gelu'`, `batch_first=True`, `norm_first=True` (pre-LN, for stability).

**Conditioning injection.** `cond (B,11,256)` is projected by `cond_obs_emb = Linear(138->256)`, the time token is prepended (`cat([time_emb, cond_obs_emb(cond)]) -> (B,12,256)`), learned `cond_pos_emb[:,:12]` is added, dropout, then the MLP encoder yields `memory`. The action tokens (`tgt = input_emb(sample) + pos_emb[:,:16]`) cross-attend into this memory. The flow-time embedding is the **first memory token**, reaching actions only via cross-attention — it is **not** summed into the action tokens.

**Timestep embedding.** `time_emb = SinusoidalPosEmb(n_emb=256)` (`oat/model/common/positional_embedding.py:13`) maps the scalar (scaled) flow time to a 256-d sin/cos vector. The policy passes `t*1000` because `SinusoidalPosEmb` was designed for integer diffusion-step ranges; raw `t in [0,1]` would barely vary the frequencies. The same scaling is applied in training and inference.

**Masking — none.** `causal_attn=False` sets `self.mask = None` and `self.memory_mask = None`. The decoder is called with `tgt_mask=None, memory_mask=None`: **fully bidirectional** self-attention over the 16 action tokens and full cross-attention to all 12 cond tokens — the whole chunk is denoised jointly. The causal/memory-mask construction code is dead in this config.

**Projections:** `input_emb = Linear(7->256)` in; `ln_f` + `head = Linear(256->7)` out. Positional embeddings are learned `nn.Parameter`s (action `pos_emb (1,16,256)`, cond `cond_pos_emb (1,12,256)`), sliced to live length and excluded from weight decay. `forward(sample, timestep, cond)` returns `v_pred (B,16,7)`.

---

## Flow-matching policy core

Source: `oat/policy/flow_policy_with_enriched_past.py`.

### The "enriched past" mechanism (`_build_condition`, lines 223-258)
The injected past info is the `past_n=7` raw (unnormalized) actions immediately preceding the chunk, ordered oldest->newest `[a_{t-7}, ..., a_{t-1}]`. They are normalized via the **shared** `normalizer["action"]` (same key as the target), then turned into three kinds of tokens:
- **acc** (acceleration-level): `a_{t-1} - a_{t-2}`, projected by `acc_proj` (`Linear(7->138)->GELU->Linear(138->138)`).
- **jerk** (jerk-level): `a_{t-1} - 2 a_{t-2} + a_{t-3}`, projected by `jerk_proj` (independent weights from acc — "different scales").
- **raw history**: all 7 normalized past actions, each projected by the **shared** `raw_proj`.

These are concatenated along the token axis after the obs tokens:
`torch.cat([obs_features (B,2,d), explicit (B,2,d), raw_feat (B,7,d)], dim=1)` -> `(B, 11, 138)`. They are not added and not used as FiLM — they become extra cross-attention memory tokens.

### Warm-start prior (`_warm_start_prior`, lines 260-266)
`mu = normalize(past_action[:, -1]).unsqueeze(1).expand(-1, H, -1)` -> `(B, 16, 7)`: the normalized **most-recent** past action broadcast over all 16 chunk steps. The flow source is `x0 = mu + prior_noise_scale * N(0, I)` with `prior_noise_scale=1.0`. The novelty is the nonzero mean; at an episode start the past buffer is zero, so `mu = normalize(0)` and `x0` reduces to (near-)standard noise.

### Training velocity target + loss (`forward`, lines 328-352)
```
x1 = normalize(batch["action"])                 # (B,16,7) data endpoint (t=1)
x0 = mu + 1.0 * randn_like(x1)                   # (B,16,7) warm-start prior (t=0)
t  ~ U(0,1)  (B,)                                # continuous flow time
xt = (1 - t) * x0 + t * x1                       # linear interpolant
v_target = x1 - x0                               # constant straight-line velocity
v_pred   = model(xt, _scale_t(t), cond)          # _scale_t(t) = t * 1000
loss     = F.mse_loss(v_pred, v_target)          # plain MSE, no SNR weighting
```
This is the **standard rectified-flow convention** (`t=0` -> prior, `t=1` -> data), the opposite of the diffusion convention. There is no separate `compute_loss`: the train loop calls `loss = self.model(batch)` (`oat/workspace/train_policy.py:203`), so `forward` *is* the loss.

### Inference ODE loop (`predict_action`, lines 274-324)
Forward Euler, `N=10` uniform steps, fixed `dt=0.1`, integrating `[0,1]` with left-endpoint evaluations at `t in {0, 0.1, ..., 0.9}`:
```
x = mu + 1.0 * randn(mu.shape)
for i in range(10):
    t = i * 0.1
    x = x + 0.1 * model(x, _scale_t(t), cond)    # cond fixed across all steps
action_pred = unnormalize(x)                      # (B,16,7)
action      = action_pred[:, :8]                  # (B,8,7) executed
```

### Action slicing & past buffer
H=16 -> the first `n_action_steps=8` steps are executed (receding horizon). The rolling past buffer is then updated: since `n_exec=8 >= past_n=7`, `_past_buffer = action_pred[:, 1:8].detach().clone()` (the last 7 of the 8 executed steps). Predicted (not GT) actions feed the next chunk's conditioning, giving a mild train/inference distribution shift in `past_action` (training uses GT past from the dataset). `reset()` sets `_past_buffer=None` at episode start so the first chunk uses an all-zero past (zero acc/jerk, zero warm-start mean).

---

## Dataset & normalization

### ZarrDatasetWithPastAction (`oat/dataset/zarr_dataset_with_past.py:10`)
Extends `ZarrDataset` to also return `past_action`. The **dataset's** `n_action_steps` is bound to `${horizon}=16` (`libero10_with_past.yaml`), so internally `Ta = 16` — the top-level `n_action_steps=8` never reaches the dataset. It widens the `SequenceSampler` window:
```
pad_before = (n_obs_steps - 1) + past_n = 1 + 7 = 8
pad_after  = (n_action_steps - 1)       = 15
seq_len    = pad_before + 1 + pad_after = 24
```
`_sample_to_data` slices the flat per-episode buffer at three offsets within the 24-frame window:
- **obs**: `sample[k][7:9]` -> To=2 frames (floats cast to float32, ints kept).
- **action**: `action_start = past_n + (To-1) = 8`; `sample[action_key][8:24]` -> `[16,7]`.
- **past_action**: `sample[action_key][1:8]` -> `[7,7]`, ordered oldest->newest `[a_{t-7}...a_{t-1}]` (the policy relies on this ordering via `[:,-1], [:,-2], [:,-3]`).

`past_n` produces **past actions only** — there is no past-observation tensor; obs is always the 2 latest frames. Note: at an episode start, `SequenceSampler` (`oat/common/seq_sampler.py`) **edge-replicates** the first in-bounds frame (not literal zeros, despite the docstring), so the earliest `past_action` is the first action repeated. The inference `reset()` path uses literal zeros instead. Returned per item (before the DataLoader adds B=32): `obs` dict of `[2, ...]` tensors, `action [16,7] f32`, `past_action [7,7] f32`.

### LinearNormalizer (`oat/model/common/normalizer.py:12`)
`ZarrDataset.get_normalizer` (`oat/dataset/zarr_dataset.py:95`) builds `{'action': ..., **numeric_obs_keys}` and calls `LinearNormalizer().fit(data, last_n_dims=1, mode='limits')`. **'limits'** mode maps each channel to `[-1, 1]` (`scale = (out_max-out_min)/(max-min)`, `offset = out_min - scale*min`); constant channels (range < 1e-4) are handled specially. RGB keys are in `numeric_obs_keys`, so images get fitted params too (used by the vision encoder's internal normalize). Apply: `x*scale + offset`; invert: `(x-offset)/scale`. The policy reaches a single field via `self.normalizer["action"]` (a `SingleFieldLinearNormalizer`) for x1, `mu`, past actions, and the final unnormalize; obs keys are normalized inside the obs encoder. `set_normalizer` (`flow_policy_with_enriched_past.py:184`) loads stats into `self.normalizer` and forwards the same normalizer to `obs_encoder.set_normalizer`, which fans it out to the vision/state sub-encoders.

---

## Training

Source: `oat/workspace/train_policy.py` (`TrainPolicyWorkspace`).

**Loop.** A HuggingFace-Accelerate epoch loop (`log_with="wandb"`, optional `bf16`). Per batch (under `accelerator.accumulate`): move to device, `loss = self.model(batch)` under autocast, `accelerator.backward(loss)`. Only on `accelerator.sync_gradients` (every batch here, since `gradient_accumulate_every=1`): optional grad-norm clip (`max_grad_norm=1.0`), `optimizer.step()`, `zero_grad(set_to_none=True)`, `lr_scheduler.step()`, then **EMA update** `ema.step(unwrap_model(self.model))`. Model, EMA copy (`copy.deepcopy`), and optimizer are built after seeding; the normalizer is fit once and pushed into both model and EMA before `accelerator.prepare`.

**Two-LR optimizer groups** (`get_optimizer`, `flow_policy_with_enriched_past.py:188-215`). Four AdamW param groups (decay if `param.dim()>=2`, else no-decay):
```
{policy_decay,   lr=5e-5, wd=0.0}    # over [model, acc_proj, jerk_proj, raw_proj]
{policy_nodecay, lr=5e-5, wd=0.0}
{encoder_decay,  lr=1e-5, wd=0.0}    # over obs_encoder
{encoder_nodecay,lr=1e-5, wd=0.0}
```
`betas=(0.9, 0.95)`. The obs encoder trains at a 5x lower LR. With `weight_decay=0.0` the decay/no-decay split is cosmetic. `TransformerForDiffusion.configure_optimizers` is **not** used. LR scheduler is `constant` -> diffusers `get_constant_schedule`, which ignores `num_warmup_steps`, so `lr_warmup_steps=100` has no effect.

**EMA** (`oat/model/diffusion/ema_model.py`). Config: `update_after_step=0, inv_gamma=1.0, power=0.75, min_value=0.0, max_value=0.9999`. Per optimizer sync, `decay = 1 - (1 + step/inv_gamma)^-power` (clamped to [0, 0.9999]), then `ema_param.mul_(decay).add_(param, alpha=1-decay)`. The **EMA model is what gets evaluated and saved** — all per-epoch eval uses `accelerator.unwrap_model(self.ema_model)`.

**Eval cadence.** Validation (`val_every=10`): mean `policy(batch)` over the val loader -> `val_loss`. Reconstruction (`sample_every=10`): `predict_action` then `MSE(pred, gt)` -> `test_reconst_mse`. Rollout (`rollout_every=100`): **skipped** because `lazy_eval=true`.

**Checkpointing.** Every `checkpoint_every=10` epochs (main process), weights are temporarily DDP-unwrapped, `latest.ckpt` is written (rolling), and `TopKCheckpointManager` (`oat/common/checkpoint_util.py`, `monitor_key=mean_success_rate, mode=max, k=3`) is consulted. **Gotcha:** with `lazy_eval=true`, `mean_success_rate` never enters `step_log`, so `get_ckpt_path` always returns None — **no top-k checkpoints are written, only `latest.ckpt`**. To enable top-k by SR you must set `lazy_eval=false` (and provide a usable LIBERO env runner).

---

## Flow-matching vs diffusion variant

The DDPM analogue lives at `oat/policy/diffpolicy_with_enriched_past.py` and shares the **exact same scaffolding**: identical `_build_condition` (enriched-past tokens), the same `TransformerForDiffusion` (`causal_attn=False`, `max_cond_len=11`), the same warm-start mean `mu`, and the same receding-horizon past-buffer update. They differ only in the generative core:

| Aspect | Flow-matching (this policy) | DDPM/DDIM variant |
|---|---|---|
| Prediction target | velocity `v = x1 - x0` | noise (epsilon); `target = x1 - mu` (residual), loss is MSE to the added noise |
| Forward process | `xt = (1-t)*x0 + t*x1`, `x0 = mu + sigma*noise` | `noisy = scheduler.add_noise(target, noise, k)` |
| Timestep | continuous `t ~ U(0,1)`, scaled x1000 | discrete `randint(0, num_train_timesteps)` |
| Sampler | plain forward Euler, `N=10` steps, no scheduler | diffusers scheduler `set_timesteps` + `step(...).prev_sample` |
| Source control | `prior_noise_scale` (sigma) | `warm_start` toggle; needs `clip_sample=False` (unbounded residual) |
| Time convention | t=0 -> prior, t=1 -> data | t=0 -> clean data (diffusion) |

---

## File map

| Component | Source file |
|---|---|
| Policy (flow-matching core, loss, sampling) | `oat/policy/flow_policy_with_enriched_past.py` |
| Base policy interface | `oat/policy/base_policy.py` |
| Velocity-field backbone | `oat/model/diffusion/transformer_for_diffusion.py` |
| Timestep embedding | `oat/model/common/positional_embedding.py` |
| Fused observation encoder | `oat/perception/fused_obs_encoder.py` |
| RGB encoder (ResNet18 + SpatialSoftmax) | `oat/perception/robomimic_vision_encoder.py` |
| State encoder (Identity projection) | `oat/perception/state_encoder.py` |
| Base obs encoder interface | `oat/perception/base_obs_encoder.py` |
| Crop randomizer (OAT-local; used only if `eval_fixed_crop`) | `oat/perception/crop_randomizer.py` |
| Normalizer (per-key affine) | `oat/model/common/normalizer.py` |
| Dataset with past actions | `oat/dataset/zarr_dataset_with_past.py` |
| Base Zarr dataset + `get_normalizer` | `oat/dataset/zarr_dataset.py` |
| Sequence sampler (windowing, edge-replicate pad) | `oat/common/seq_sampler.py` |
| Training workspace | `oat/workspace/train_policy.py` |
| Base workspace (checkpoint save/load) | `oat/workspace/base_workspace.py` |
| EMA model | `oat/model/diffusion/ema_model.py` |
| Top-k checkpoint manager | `oat/common/checkpoint_util.py` |
| LR scheduler factory | `oat/model/common/lr_scheduler.py` |
| DDPM/DDIM contrast variant | `oat/policy/diffpolicy_with_enriched_past.py` |
| Train config | `oat/config/train_flowpolicy_with_enriched_past.yaml` |
| Task/policy/dataset config | `oat/config/task/policy/libero/libero10_with_past.yaml` |
