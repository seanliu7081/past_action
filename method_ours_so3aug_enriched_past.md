# Our Method — SO(3)-Augmented Tokenizer + Enriched-Past Autoregressive Policy

This document describes the **"ours"** method in the `oat/` codebase: a two-stage action-chunking pipeline where (1) an SO(3)-augmented action autoencoder learns a discrete codebook over 16-step action chunks, and (2) an autoregressive transformer policy generates those discrete codes from observations *and* a rich representation of the recent action past (raw history + explicit acceleration and jerk features).

Stage 1 yields a frozen tokenizer; Stage 2 trains the policy on top of it.

---

## 1. Overview

```
                                ╔══════════════════════════════════════════════╗
                                ║  STAGE 1 — TOKENIZER (action autoencoder)    ║
                                ╚══════════════════════════════════════════════╝
                                            (action chunks only)

  raw actions (B, 16, 7)
        │
        │   1. SO(3) augmentation (training only, p=0.6)
        │        - sample Q ∈ SO(3),  max_angle = 60°
        │        - apply  R_aug = Q @ R  to rotation slice [3:6]
        │
        ▼
  augmented actions ──► normalize ──► RegisterEncoder ──► FSQ ──► tokens (B, 8)
                                                                         │
                                                                         ▼
                                                            SinglePassDecoder
                                                                         │
                                                                         ▼
                                                                MSE loss vs. normalized augmented actions


                                ╔══════════════════════════════════════════════╗
                                ║  STAGE 2 — POLICY (autoregressive over codes) ║
                                ╚══════════════════════════════════════════════╝

  obs_dict {rgb×2, state×4}              past_action  a_{t-7..t-1}
      │                                       │
      ▼                                       ▼
  FusedObservationEncoder              acc = a_{t-1} − a_{t-2}
  (ResNet18 + SpatialSoftmax           jerk = a_{t-1} − 2a_{t-2} + a_{t-3}
   for RGB; Identity proj
   for state)                          acc_proj(acc), jerk_proj(jerk), raw_proj(past)
      │                                       │
      └────────────► concat ◄─────────────────┘
                       │
                       │   condition sequence of length 2 + 2 + 7 = 11 tokens
                       ▼
              AutoregressiveModel (causal self-attn + cross-attn to cond)
                       │
                       ▼
                  next-token logits over codebook (size 1001 incl. <BOS>)

  At inference: AR-generate 8 tokens with KV cache → detokenize via frozen
                tokenizer → execute first 8 of 16 reconstructed actions
                (receding horizon) → update past_buffer with predicted actions.
```

**Key numbers:** action dim `D=7`, action horizon `T=16`, observation horizon `To=2`, executed horizon `n_action_steps=8`, past horizon `past_n=7`, FSQ levels `[8, 5, 5, 5]` → codebook size **1000**, latent horizon (= register count) **8**, condition length **11**, AR backbone width 256 / depth 4 / 4 heads.

---

## 2. Code structure

Files that define this method (relative to `/workspace/oat/`):

```
oat/
├── config/
│   ├── train_oattok_so3aug.yaml                    [Stage 1 hydra entry point]
│   ├── train_oatpolicy_with_enriched_past.yaml     [Stage 2 hydra entry point]
│   └── task/
│       ├── tokenizer/libero/libero10.yaml          [tokenizer task — ZarrDataset, no past]
│       └── policy/libero/libero10_with_past.yaml   [policy task — ZarrDatasetWithPastAction]
│
├── workspace/
│   ├── train_oattok.py                             [Stage 1 training loop]
│   └── train_policy.py                             [Stage 2 training loop + rollout]
│
├── tokenizer/
│   ├── base_tokenizer.py                           [abstract base, from_checkpoint loader]
│   └── oat/
│       ├── tokenizer.py                            [OATTok base class — used by from_checkpoint]
│       ├── tokenizer_so3_aug.py                    [OATTokSO3Aug — adds SO(3) aug in forward]
│       ├── encoder/register_encoder.py             [RegisterEncoder, causal-last mask]
│       ├── decoder/single_pass_decoder.py          [SinglePassDecoder, Matryoshka dropout]
│       ├── quantizer/fsq.py                        [FSQ — factorized scalar quantization]
│       └── augment/so3_action_chunk_aug.py         [SO3ActionChunkAug — left/right/conjugate]
│
├── policy/
│   ├── base_policy.py                              [abstract policy interface]
│   └── oat_policy_with_enriched_past.py            [OATPolicyWithEnrichedPast]
│
├── model/
│   ├── autoregressive/transformer_cache.py         [AutoregressiveModel — KV-cached AR]
│   └── common/normalizer.py                        [LinearNormalizer — fit on dataset]
│
├── perception/
│   ├── fused_obs_encoder.py                        [FusedObservationEncoder — multimodal fusion]
│   ├── robomimic_vision_encoder.py                 [ResNet18 + SpatialSoftmax]
│   └── state_encoder.py                            [ProjectionStateEncoder]
│
├── dataset/
│   └── zarr_dataset_with_past.py                   [ZarrDatasetWithPastAction — adds past_n slice]
│
└── env_runner/
    └── libero_runner.py                            [LiberoRunner — parallel rollout for SR metric]
```

Entry points (`python -m hydra.main` style; the workspace `__init__` reads the yaml and runs `.run()`):

| Stage | Config (relative) | Workspace class | Output |
|---|---|---|---|
| 1 | `config/train_oattok_so3aug.yaml` | `oat.workspace.train_oattok.TrainOATTokWorkspace` | `*.ckpt` containing the frozen tokenizer |
| 2 | `config/train_oatpolicy_with_enriched_past.yaml` (override `policy.action_tokenizer.checkpoint=<stage1.ckpt>`) | `oat.workspace.train_policy.TrainPolicyWorkspace` | `*.ckpt` containing the full policy (tokenizer + AR backbone + projections) |

---

## 3. Neural network framework

### 3.1 Tokenizer side — `OATTokSO3Aug`

`OATTokSO3Aug` (in [oat/tokenizer/oat/tokenizer_so3_aug.py](oat/oat/tokenizer/oat/tokenizer_so3_aug.py)) is a 25-line subclass of `OATTok` that only differs in `forward()`:

```python
def forward(self, batch):
    samples = batch["action"]                        # (B, 16, 7) raw actions
    if self.action_aug is not None:
        samples = self.action_aug(samples)           # SO(3) aug on rotation slice
    nsamples = self.normalizer["action"].normalize(samples)
    latents = self.encoder(nsamples)                 # (B, 8, latent_dim)
    latents, _ = self.quantizer(latents)             # quantize, latents ← codebook(STE)
    recons = self.decoder(latents)                   # (B, 16, 7) normalized recon
    return F.mse_loss(recons, nsamples)
```

Everything else (`encode`, `decode`, `tokenize`, `detokenize`, optimizer factory, normalizer plumbing) is inherited unchanged from [oat/tokenizer/oat/tokenizer.py](oat/oat/tokenizer/oat/tokenizer.py).

#### 3.1.1 `SO3ActionChunkAug` — augmentation module

File: [oat/tokenizer/oat/augment/so3_action_chunk_aug.py](oat/oat/tokenizer/oat/augment/so3_action_chunk_aug.py). Configured in YAML as:

```yaml
action_aug:
  _target_: oat.tokenizer.oat.augment.so3_action_chunk_aug.SO3ActionChunkAug
  p: 0.6                   # per-sample probability of augmenting
  max_angle_deg: 60.0      # magnitude cap (uniform in [0, max_angle_rad])
  mode: left_noise         # Q @ R
  augment_position: false  # do not also rotate the position slice
  pos_start: 0; pos_end: 3
  rot_start: 3; rot_end: 6
```

**Math.** Each robot action is `a = [pos (3) | rotvec (3) | gripper (1)] ∈ ℝ⁷`. The rotation slice is interpreted as a Rodrigues vector (axis × angle).

1. Sample mask `m ~ Bernoulli(p)` per batch element (one mask per chunk — *all 16 timesteps share the same Q*).
2. Sample `Q ∈ SO(3)`:
   - axis `n ~ Uniform(S²)` (unit-vector by normalizing `randn(3)`),
   - angle `θ ~ Uniform(0, max_angle_rad)`,
   - rotation vector `ω = θ · n`, then `Q = expmap(ω)` via Rodrigues' formula.
3. Decode current rotation: `R(t) = expmap(rotvec(t))` for each `t ∈ [0..15]`.
4. Apply mode (with `mode=left_noise`): `R_aug(t) = Q · R(t)`.
5. Re-encode: `rotvec_aug(t) = logmap(R_aug(t))`.
6. Replace rotvec slice only where mask is 1; positions and gripper untouched.

The `so3_exp_map` / `so3_log_map` helpers use the standard Rodrigues formula and its Taylor expansions near `θ ≈ 0` (`θ² < 1e-8`) for numerical stability, with intermediate computation upcast via `_compute_dtype` to float32.

Augmentation is disabled outside `self.training`.

#### 3.1.2 `RegisterEncoder`

File: [oat/tokenizer/oat/encoder/register_encoder.py](oat/oat/tokenizer/oat/encoder/register_encoder.py). Configured as:

```yaml
encoder:
  sample_dim: 7        # action_dim
  sample_horizon: 16   # horizon T
  emb_dim: 256
  head_dim: 64         # → n_heads = 256/64 = 4
  depth: 2
  pdropout: 0.1
  latent_dim: 4        # = len(quantizer.levels)
  num_registers: 8     # bottleneck size = latent_horizon
```

Architecture (forward pass):

```
samples (B, 16, 7)
  │
  ▼
SampleEmbedder: Linear(7→256)            → (B, 16, 256)
  │
  ▼
PositionalEmbeddingAdder (1D sincos)     → (B, 16, 256)
  │
  ▼
Concatenate with N=8 learnable registers → (B, 16+8, 256)
  │
  ▼
Transformer (depth=2, 4-head MSA + 4× MLP, pre-norm, SiLU)
   with "causal-last" attention mask:
   ┌──────────────────────────────────┐
   │ Action tokens  : full attention  │
   │ Register tokens: full over actions
   │                  causal over registers (lower-triangular among themselves)
   └──────────────────────────────────┘
  │
  ▼
Slice out the 8 register positions       → (B, 8, 256)
  │
  ▼
LinearHead: 256 → 4                      → (B, 8, 4)  latents
```

The causal-last mask forces the registers to act as a sequential information bottleneck: register `r_i` can only attend to actions plus `r_0, ..., r_i`. This implicitly orders the latents and supports the Matryoshka token-dropout at decode time.

#### 3.1.3 `FSQ` — Finite Scalar Quantization

File: [oat/tokenizer/oat/quantizer/fsq.py](oat/oat/tokenizer/oat/quantizer/fsq.py). Configured as:

```yaml
quantizer:
  levels: [8, 5, 5, 5]    # → codebook_size = 8·5·5·5 = 1000
```

Each latent vector `z ∈ ℝ⁴` is quantized **dimension-independently**:

```
For dim i with L_i levels:
  bound(z_i) = (L_i - 1)/2 · tanh(z_i + shift_i)        # squashes into valid range
  quantize:   ẑ_i = round_ste(bound(z_i)) / (L_i // 2)  # straight-through round
```

The codebook is implicit (no learnable embedding table), with mixed-radix flat indexing:
```
basis = [1, 8, 8·5, 8·5·5]
flat_index(ẑ) = Σ_i ẑ_i_int · basis[i]      # one int ∈ [0, 1000)
```

`indices_to_embedding(flat_idx)` is the exact inverse. The straight-through estimator is `ẑ = z + (round(z) - z).detach()`. Optional `drop_quant_p` randomly skips quantization on a fraction of latents and `corrupt_tokens_p` replaces some quantized codes with uniform-random codebook entries (both default to 0 in `train_oattok_so3aug.yaml`).

**No codebook collapse** since each codebook entry is fixed — every code is always reachable.

#### 3.1.4 `SinglePassDecoder`

File: [oat/tokenizer/oat/decoder/single_pass_decoder.py](oat/oat/tokenizer/oat/decoder/single_pass_decoder.py). Configured as:

```yaml
decoder:
  sample_dim: 7
  sample_horizon: 16
  emb_dim: 256
  head_dim: 64        # n_heads = 4
  depth: 4
  pdropout: 0.1
  token_dropout_mode: pow2
  latent_dim: 4
  latent_horizon: 8   # = num_registers
  use_causal_decoder: true
```

Forward pass:

```
latents (B, 8, 4)
  │
  ▼
LinearLayer 4 → 256                                   → (B, 8, 256)
  │
  ▼
PositionalEmbeddingAdder (1D sincos)                  → (B, 8, 256)
  │
  ▼
MaskedNestedDropout (mode='pow2'):
   k ~ Uniform({1, 2, 4, 8}); replace tokens at positions ≥ k with a
   learnable <MASK> embedding. Acts as Matryoshka training — at inference,
   eval_keep_k can be set to any of {1,2,4,8} for a latency/quality knob.
  │
  ▼
Generate sample-side positional queries (1, 256, 16) → (B, 16, 256)
  │
  ▼
nn.TransformerDecoder(depth=4, 4-head, GELU, norm_first=True):
   - tgt = positional queries (size 16)
   - memory = latents (size 8)
   - tgt_mask = causal lower-triangular (use_causal_decoder=true)
   - cross-attn (tgt ← memory) has no mask
  │
  ▼
LinearHead: 256 → 7                                   → (B, 16, 7)  reconstructed (normalized) actions
```

The causal self-attention on the output side enforces temporal coherence; cross-attention is unrestricted so each output step can pull from any active register.

### 3.2 Policy side — `OATPolicyWithEnrichedPast`

File: [oat/policy/oat_policy_with_enriched_past.py](oat/oat/policy/oat_policy_with_enriched_past.py).

#### 3.2.1 Observation encoder

`FusedObservationEncoder` ([oat/perception/fused_obs_encoder.py](oat/oat/perception/fused_obs_encoder.py)) is a thin wrapper that, given a `shape_meta`, instantiates and concatenates:

- **`RobomimicRgbEncoder`** ([oat/perception/robomimic_vision_encoder.py](oat/oat/perception/robomimic_vision_encoder.py)) — a robomimic `VisualCore` with a `ResNet18Conv` backbone (no BatchNorm, replaced with GroupNorm), `SpatialSoftmax` pooling (32 keypoints), flattened. `crop_shape=[76, 76]` does a random crop during training and a center crop at eval. Applied independently to each RGB port (`agentview_rgb`, `robot0_eye_in_hand_rgb`).
- **`ProjectionStateEncoder`** ([oat/perception/state_encoder.py](oat/oat/perception/state_encoder.py)) — concatenates state ports (`robot0_eef_pos[3]`, `robot0_eef_quat[4]`, `robot0_gripper_qpos[2]`, `task_uid[1]` → 10 dims) and applies an `nn.Identity` projection (`out_dim: null` in YAML).

Output shape: `(B, To=2, d)` where `d = (Σ vision dims) + (Σ state dims)` is the fused per-step feature dimension.

#### 3.2.2 Condition assembly — the *enriched past*

This is the core architectural change. `_build_condition(obs_features, past_actions)` constructs an extended condition sequence of length **`To + 2 + past_n = 2 + 2 + 7 = 11`**.

```python
norm_past = action_normalizer["action"].normalize(past_actions)      # (B, 7, 7)

# Explicit higher-order features from the last 3 past actions
a_t1, a_t2, a_t3 = norm_past[:, -1], norm_past[:, -2], norm_past[:, -3]
acc  = a_t1 -        a_t2                                            # (B, 7)
jerk = a_t1 - 2.0 *  a_t2 + a_t3                                    # (B, 7)

acc_feat  = acc_proj(acc)        # (B, d)         Linear(7, d) → GELU → Linear(d, d)
jerk_feat = jerk_proj(jerk)      # (B, d)         Linear(7, d) → GELU → Linear(d, d)
raw_feat  = raw_proj(norm_past)  # (B, 7, d)      shared Linear(7, d) → GELU → Linear(d, d) over time

explicit = torch.stack([acc_feat, jerk_feat], dim=1)  # (B, 2, d)

cond = torch.cat([obs_features, explicit, raw_feat], dim=1)  # (B, 11, d)
```

Notes:

- `d = obs_feature_dim` is fixed by the observation encoder's output, so the three new projection MLPs all share that dimension.
- Three *separate* MLPs (not one shared) — each input has a different scale (raw values O(1), acc O(Δ), jerk O(Δ²)) and a different semantic role.
- Computation is done on **normalized** past actions (matched to how the tokenizer ingests actions), so differencing is in normalized space.

#### 3.2.3 AR backbone — `AutoregressiveModel`

File: [oat/model/autoregressive/transformer_cache.py](oat/oat/model/autoregressive/transformer_cache.py). Configured as:

```yaml
embed_dim: 256
n_layers: 4
n_heads:  4
dropout:  0.1
# vocab_size  = codebook_size + 1   = 1001  (+1 for <BOS>)
# max_seq_len = latent_horizon + 1  = 9     (BOS + 8 code tokens)
# max_cond_len = 11
# cond_dim   = obs_feature_dim
```

Per-layer structure (pre-norm, RMSNorm):

```
Block:
  x = x + CausalSelfAttention( RMSNorm(x),   layer_past=KV cache )
  x = x + CrossAttention     ( RMSNorm(x),   memory=cond_encoded, memory_kv=memory cache )
  x = x + MLP                ( RMSNorm(x) )  # n_emb → 4·n_emb → n_emb, GELU
```

Inputs:

- `tok_emb(tokens) + tok_pos_emb` for the AR sequence (learned positional embeddings of size 9).
- `cond_emb(cond) + cond_pos_emb` then a 2-layer MLP encoder (`Linear → Mish → Linear`) to produce the **memory** used by every cross-attention layer.

Output head is a `Linear → vocab_size` tied to `tok_emb` weights.

**Generation** is KV-cached:
1. Process the BOS prefix once, caching K/V per layer for both self- and cross-attention.
2. For each of the 8 generation steps, embed the just-sampled token, run the blocks with the cached past, apply top-k temperature sampling on the resulting logits.

Cross-attention to `cond` is unmasked (the model can look at any condition position from any AR step). Self-attention is causal (an AR step at position `i` can only see positions `0..i`).

---

## 4. Methodology

### 4.1 Stage 1 — Tokenizer training

**Data.** `ZarrDataset` on `data/libero/libero10_N500.zarr`, **action-only** (`obs_keys: []`, `n_obs_steps: 0`, `n_action_steps: 16`). One sample = one 16-step action chunk. Sliding window over each demo with `pad_after = 15`; `val_ratio = 0.1`.

**Normalizer.** `LinearNormalizer.fit(...)` on the *unaugmented* training actions, `mode='limits'`, `last_n_dims=1` → per-channel min-max scaling to `[-1, +1]`. Frozen for the rest of training. (Augmentation is applied to *raw* actions **before** normalization; the normalizer's statistics are estimated once from the raw distribution.)

**Loss.** `F.mse_loss(decoder(quantize(encoder(normalize(aug(raw))))), normalize(aug(raw)))` — reconstruct the *augmented*, *normalized* action chunk. The straight-through estimator in FSQ lets gradients flow through quantization.

**Optimization.** AdamW (`lr=5e-5`, `betas=(0.9, 0.95)`, `weight_decay=0`); weight decay split (2D params decay, 1D do not — though `weight_decay=0` here makes this moot). LR scheduler `constant` with `lr_warmup_steps=100`. EMA with `power=0.75`, `inv_gamma=1.0`, `max_value=0.9999`. Trained for up to `5001` epochs with `gradient_accumulate_every=1`, `batch_size=256`.

**Evaluation cadence.** Every 10 epochs: validation MSE (on normalized actions, no augmentation), reconstruction MSE on a held-out sample (`tokenizer.autoencode(samples)` then MSE in *un-normalized* space). Top-3 checkpoints by `test_reconst_mse` (mode=min). Last checkpoint always saved.

### 4.2 Stage 2 — Policy training

**Data.** `ZarrDatasetWithPastAction` ([oat/dataset/zarr_dataset_with_past.py](oat/oat/dataset/zarr_dataset_with_past.py)) on the same zarr. Sliding window with `pad_before = max(n_obs_steps − 1, 0) + past_n = 1 + 7 = 8` and `pad_after = max(n_action_steps − 1, 0) = 15`. Each sample yields:

```
obs         : dict of {(B,) To=2 frames of each rgb/state port}
action      : (B, 16, 7)   the chunk to predict
past_action : (B, 7, 7)    ordered [a_{t-7}, a_{t-6}, ..., a_{t-1}]
```

Padding at episode starts is zero-filled by the underlying `SequenceSampler`.

**Loss.** Frozen-tokenizer cross-entropy on next-action-token prediction (teacher forcing):

```python
with torch.no_grad():
    action_tokens = self.action_tokenizer.tokenize(batch["action"])   # (B, 8) ∈ [0, 1000)

cond = self._build_condition(self.obs_encoder(batch["obs"]), batch["past_action"])  # (B, 11, d)

action_tokens = torch.cat([BOS, action_tokens], dim=1)  # (B, 9)
logits = self.model(action_tokens[:, :-1], cond=cond)   # (B, 8, 1001)
loss   = F.cross_entropy(logits.flatten(0,1), action_tokens[:, 1:].flatten())
```

The **tokenizer is fully frozen** (`requires_grad_(False)` + `.eval()`); only the observation encoder, the three projection MLPs (`acc_proj`, `jerk_proj`, `raw_proj`), and the AR backbone are trained.

**Optimization.** AdamW with two LR groups — `policy_lr=5e-5` for AR backbone + the three projection MLPs, `obs_enc_lr=1e-5` for the visual/state encoders (vision pretrained at higher LR would be wasteful). `weight_decay=0`, `betas=(0.9, 0.95)`. Same constant scheduler + warmup, same EMA settings. `batch_size=64`, `num_workers=2`.

**Evaluation cadence.** Every 10 epochs: validation cross-entropy, sampled-action MSE (decode predicted tokens and compare). Every `rollout_every=200` epochs: full `LiberoRunner` rollout over 500 init configs across 10 tasks, in 20 parallel envs. Top-3 checkpoints by `mean_success_rate` (mode=max).

### 4.3 Inference

`predict_action(obs_dict)` (called by `LiberoRunner` once per `n_action_steps=8` environment steps via `MultiStepWrapper`):

```python
features = self.obs_encoder(obs_dict)                # (B, 2, d)

# Lazily init/reset past buffer (e.g. at the start of a new episode after reset())
if self._past_buffer is None or shape/device mismatch:
    self._past_buffer = zeros(B, past_n=7, action_dim=7)

cond = self._build_condition(features, self._past_buffer)              # (B, 11, d)

action_tokens = AutoregressiveModel.generate(
    prefix=BOS,
    cond=cond,
    max_new_tokens=8,    # = use_k_tokens, clamped to max_seq_len
    temperature=1.0,
    top_k=10,
)[:, 1:]                                             # drop BOS  → (B, 8)

action_pred = self.action_tokenizer.detokenize(action_tokens)   # (B, 16, 7)
action      = action_pred[:, : n_action_steps]                  # (B, 8, 7)  executed slice

# Update past buffer (predicted, not executed) — used as a_{t-7..t-1} on the next call.
if n_action_steps >= past_n:                                # 8 ≥ 7 → yes
    self._past_buffer = action_pred[:, n_action_steps - past_n : n_action_steps]
    # = action_pred[:, 1:8]  i.e. the most recent past_n=7 of the 8 executed steps
else:
    self._past_buffer = torch.cat([self._past_buffer[:, n_action_steps:],
                                   action_pred[:, : n_action_steps]], dim=1)
```

`reset()` is called by the env runner at episode start and sets `_past_buffer = None` so the next `predict_action` re-initializes it with zeros.

**Important:** the past buffer stores *predicted actions* — exactly what the AR head emitted last time — not what was returned by the environment. This is by design (see § 5).

### 4.4 Hyperparameters

| Group | Param | Stage 1 (tokenizer) | Stage 2 (policy) |
|---|---|---|---|
| Action | `horizon T` | 16 | 16 |
| Action | `action_dim D` | 7 | 7 |
| Action | `n_action_steps` | — | 8 (executed slice) |
| Obs | `n_obs_steps To` | 0 (action-only) | 2 |
| Past | `past_n` | — | 7 |
| Tokenizer | `latent_dim` | 4 (= len(levels)) | inherited |
| Tokenizer | `num_registers / latent_horizon` | 8 | inherited |
| Tokenizer | `emb_dim / head_dim / depth (enc)` | 256 / 64 / 2 | — |
| Tokenizer | `emb_dim / head_dim / depth (dec)` | 256 / 64 / 4 | — |
| Tokenizer | `token_dropout_mode` | pow2 | — |
| Quantizer | `FSQ levels` | [8, 5, 5, 5] | inherited |
| Quantizer | codebook size | 1000 | 1000 |
| Aug | `p / max_angle_deg / mode` | 0.6 / 60° / left_noise | — |
| Aug | `augment_position` | false | — |
| Policy | `embed_dim / n_layers / n_heads` | — | 256 / 4 / 4 |
| Policy | `dropout` | — | 0.1 |
| Policy | `temperature / topk` | — | 1.0 / 10 |
| Policy | `max_cond_len` | — | 11 |
| Optim | `learning_rate` (tokenizer) | 5e-5 | — |
| Optim | `policy_lr / obs_enc_lr` | — | 5e-5 / 1e-5 |
| Optim | `weight_decay / betas` | 0 / (0.9, 0.95) | 0 / (0.9, 0.95) |
| Optim | `max_grad_norm` | 1.0 | 1.0 |
| Sched | `lr_scheduler / lr_warmup_steps` | constant / 100 | constant / 100 |
| EMA | `power / inv_gamma / max_value` | 0.75 / 1.0 / 0.9999 | 0.75 / 1.0 / 0.9999 |
| Data | `batch_size / num_workers` | 256 / 4 | 64 / 2 |
| Data | `num_demo / val_ratio` | 500 / 0.1 | 500 / 0.1 |
| Train | `num_epochs` | 5001 | 5001 |
| Train | `val_every / sample_every` | 10 / 10 | 10 / 10 |
| Train | `checkpoint_every / rollout_every` | 10 / — | 10 / 200 |
| Train | `bf16` | yes | yes |
| Ckpt | monitor / k | test_reconst_mse (min) / 3 | mean_success_rate (max) / 3 |

---

## 5. Theoretical justification — why this design

### 5.1 Information content of observations vs. raw past actions

The LIBERO observation includes `robot0_eef_pos` (3), `robot0_eef_quat` (4), `robot0_gripper_qpos` (2), and `task_uid` (1) per frame, plus two RGB views. With `n_obs_steps=2`, the policy has access to *position* and a one-step *backward-Euler velocity estimate*. But:

- Proprioceptive states are post-controller — they reflect what the robot achieved, with some lag relative to commanded actions.
- A two-frame stack at control frequency gives one velocity estimate per dimension; second-order derivatives (acceleration, jerk) would require fitting a cubic with too few samples.

Past *actions* (`past_action`) are the *commanded* values. They are noise-free, instantaneous, and reveal command inertia (e.g., the policy was already heading in a direction even if the robot hasn't quite reached it yet). For manipulation under contact, commanded force/velocity is often more informative for the *next* command than the post-contact observed state.

### 5.2 Why explicit `acc` and `jerk` in addition to raw past

A transformer with cross-attention to a 7-token raw-past sequence can in principle learn `a_{t-1} - 2a_{t-2} + a_{t-3}` by allocating attention heads to position-shifted subtraction. But this means *learning the subtraction pattern* from gradient signal alone. By pre-computing the differences and giving the model two extra condition tokens whose embedding is exactly the higher-order derivative, the model gets that inductive bias for free.

This is the classical move of "features the model could derive, but shouldn't have to": injecting it as a special token costs negligible parameters and saves capacity for the genuinely hard parts of the mapping (cross-modal grounding of language ↔ vision ↔ action).

### 5.3 Why three separate projection MLPs (`acc_proj`, `jerk_proj`, `raw_proj`)

The three inputs to the projection layer live on different scales:

- `norm_past` ∈ [−1, +1]ⁿ (after limits-normalization).
- `acc = a_{t-1} − a_{t-2}` is the change between consecutive *normalized* actions — typically O(0.01–0.1) for smooth trajectories.
- `jerk = a_{t-1} − 2a_{t-2} + a_{t-3}` is O(0.001–0.01).

A *shared* projection would either (a) be dominated by the raw-past path because of the magnitude difference, or (b) need to learn input-dependent scaling on top. Three small MLPs (`Linear(7, d) → GELU → Linear(d, d)`, where `d ≈ 80–100` typically) — total parameter cost is a few hundred K — keep the scales decoupled and let each path settle at its natural gain.

### 5.4 Why SO(3) data augmentation on actions

Robot orientation lives on a 3-D manifold (`SO(3)`), not in Euclidean space. Adding Gaussian noise directly to the Rodrigues vector is **not** equivalent to perturbing the rotation: e.g., a rotation vector `(0, 0, π)` and `(0, 0, π + ε)` represent very different rotations near the discontinuity. To stay on the manifold while perturbing, the standard move is to sample a rotation `Q ∈ SO(3)` and apply it via group multiplication:

```
R_aug = Q · R         (left-multiplication, "world-frame" perturbation)
```

This always produces a valid rotation. The 16-step chunk gets a *single, consistent* `Q` so the relative motion across the chunk is preserved (it's the whole chunk that's "rotated"), forcing the tokenizer to encode the relative geometry rather than the absolute reference frame.

For a 1000-token codebook over 16-step chunks, the encoder has only so many discrete buckets to spread the rotation manifold over. Without rotation augmentation, the codes can specialize to the narrow rotation cone present in the LIBERO demos; with augmentation they have to generalize across the manifold.

### 5.5 Why `mode = left_noise` (not `right_noise` or `conjugate`)

The three modes correspond to different physical interpretations:

| Mode | Formula | Interpretation |
|---|---|---|
| `left_noise` | `Q · R` | Perturb the *world / reference frame*: same intended grip, viewed from a tilted world. |
| `right_noise` | `R · Q` | Perturb the *end-effector frame*: same world target, but a different roll/pitch/yaw of the gripper. |
| `conjugate` | `Q · R · Qᵀ` | Frame-invariant rotation of the rotation itself — symmetric but more aggressive. |

For an action representation that's *world-frame deltas to the eef*, `left_noise` is the "task-preserving" choice: it simulates "the camera/table is rotated a bit relative to where the demo was recorded" without changing what the robot is supposed to *do* in its own frame. `right_noise` *changes the semantics* of the action (a different grip orientation) which is what you'd want for grasp diversity, not codebook robustness. `conjugate` is a symmetric perturbation that touches both interpretations.

Consistent with this, `augment_position: false` in the config: under a world-frame perturbation, the positions *should* in principle rotate too, but for the action representation used here (relative deltas in a fixed task frame), letting the rotation slice float while keeping positions fixed is a deliberate decoupling — it forces the encoder to learn rotation invariance for the orientation slice specifically.

### 5.6 Why `p = 0.6` and `max_angle_deg = 60°`

- `p = 0.6` — aggressive enough that the augmented distribution dominates training, but `p < 1.0` keeps 40 % of batches as anchors so reconstruction on un-augmented data is still well-targeted.
- `max_angle_deg = 60°` — large by RL-augmentation standards. The yaml shows two commented-out alternatives (`10°`, `30°`), suggesting `60°` is the result of a sweep. A `60°` per-chunk rotation is bigger than what any single demo trajectory exhibits, but matches the *across-demo* rotational variation of LIBERO tasks where the same skill is performed from different angles.

### 5.7 Why the past buffer is *predicted* (not *executed*) actions

At training time, `past_action` comes from the dataset — these are the recorded human/expert commands, identical to the commanded action stream. At inference time, the closest thing to "the previous command stream" is the policy's *own* most recent predictions: if the policy is generating an 8-step chunk every 8 env steps, then `action_pred[:, 1:8]` are exactly what was sent to the controller (modulo the MultiStepWrapper). Storing these in `_past_buffer` keeps the train/test distributions aligned.

Using the env's `info["last_action"]` (post-controller) or proprioceptive state derivatives would introduce a shift: during training the network sees clean commands; during rollout it would see noisy realized motion. The receding-horizon update rule `_past_buffer = action_pred[:, 1:8]` is the natural matched-distribution choice.

### 5.8 Why the rest of the architecture is unchanged

The encoder, decoder, quantizer, observation encoder, and AR backbone are inherited verbatim from the baseline. The contribution of this method is in the *training signal* (SO(3) augmentation on actions) and the *condition signal* (enriched past), not in the architectural primitives. Keeping the rest fixed:

- makes the comparison clean (any improvement attributable to those two specific changes),
- lets the frozen tokenizer trained under SO(3) augmentation be plug-replaced into either policy (baseline or ours) for ablation,
- avoids hyperparameter co-tuning that would muddy the attribution.
