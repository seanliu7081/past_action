# Architecture Summary — `oattok_so3aug` Tokenizer + Enriched-Past Policy

A current-state walkthrough of the two-stage method in [oat/](oat/):

1. **Stage 1 — `oattok_so3aug`** — an SO(3)-augmented action autoencoder that compresses 16-step action chunks into 8 discrete codebook tokens.
2. **Stage 2 — enriched-past policy** — an autoregressive transformer that predicts those 8 tokens conditioned on observations plus a richly-featurized recent action history (raw past + acceleration + jerk).

Stage 1 produces a frozen tokenizer; Stage 2 trains the policy on top of it.

---

## 1. End-to-end data flow

```
                       ╔════════════════════════════════════════════════╗
                       ║  STAGE 1 — TOKENIZER  (OATTokSO3Aug)           ║
                       ║  action chunks only — observations not used    ║
                       ╚════════════════════════════════════════════════╝

  raw actions  a ∈ ℝ^(B, 16, 7)
       │
       │  SO3ActionChunkAug   (training only, p=1.0, ±30°)
       │  - sample one Q ∈ SO(3) per chunk, axis ~ Uniform(S²), angle ~ U[0, 30°]
       │  - R(t) = expmap(rotvec(t));  R_aug(t) = Q · R(t);  rotvec_aug(t) = logmap(R_aug(t))
       │  - rotation slice [3:6] replaced with rotvec_aug; positions [0:3] and gripper [6] untouched
       ▼
  augmented actions  ──►  LinearNormalizer (limits → [-1, +1])
                                              │
                                              ▼
                                   RegisterEncoder
                                   (16 sample tokens + 8 registers,
                                    "causal-last" attention mask)
                                              │
                                              ▼
                                   latents  (B, 8, 4)
                                              │
                                              ▼
                                   FSQ quantizer (levels [8,5,5,5,5])
                                              │
                                              ▼
                                   discrete tokens  (B, 8)  ∈ [0, 5000)
                                              │
                                              ▼
                                   SinglePassDecoder
                                   (causal self-attn over 16 positional queries
                                    + cross-attn to 8 latents,
                                    Matryoshka pow2 token dropout)
                                              │
                                              ▼
                                   reconstructed (normalized) actions  (B, 16, 7)
                                              │
                                              ▼
                                   MSE loss vs. normalized augmented actions


                       ╔════════════════════════════════════════════════╗
                       ║  STAGE 2 — POLICY  (OATPolicyWithEnrichedPast) ║
                       ╚════════════════════════════════════════════════╝

  obs_dict {agentview_rgb, eye_in_hand_rgb, eef_pos, eef_quat, gripper_qpos, task_uid}
       │
       ▼
  FusedObservationEncoder   →   obs_features  (B, To=2, d)
        ResNet18 + SpatialSoftmax (per RGB port; 76×76 crop)
        ProjectionStateEncoder = Identity over concatenated state ports (10 dims)

                                                   past_action  (B, 7, 7)   = [a_{t-7}, …, a_{t-1}]
                                                          │
                                                  LinearNormalizer (shared with action tokenizer)
                                                          │
                              ┌───────────────────────────┴─────────────────────────┐
                              │                           │                         │
                              ▼                           ▼                         ▼
                       acc = a_{t-1} - a_{t-2}    jerk = a_{t-1} - 2a_{t-2} + a_{t-3}     norm_past (B, 7, 7)
                              │                           │                         │
                       acc_proj                    jerk_proj                  raw_proj
                       (Linear→GELU→Linear)        (Linear→GELU→Linear)       (Linear→GELU→Linear,
                                                                              applied per timestep)
                              │                           │                         │
                              ▼                           ▼                         ▼
                          (B, 1, d)                   (B, 1, d)                 (B, 7, d)
                              └─────────────── concat ─────────────────┐
                                                                       ▼
                                            condition sequence  (B, 11, d) = obs(2) | acc | jerk | raw(7)
                                                                       │
                                                                       ▼
                                  AutoregressiveModel (4 layers · 4 heads · 256 dim)
                                  · causal self-attn over BOS + 8 code tokens
                                  · cross-attn to the 11-token condition memory
                                                                       │
                                                                       ▼
                            next-token logits over vocab = codebook_size + 1 = 5001

  Inference: AR-generate 8 tokens with KV cache  →  detokenize via frozen tokenizer
             →  execute first n_action_steps = 8 of the 16 reconstructed actions
             →  update _past_buffer with the most recent 7 *predicted* actions
                (matched-distribution to training).
```

---

## 2. Key shapes & hyperparameters (current configs)

| Group                          | Param                                | Value                                         |
|--------------------------------|--------------------------------------|-----------------------------------------------|
| Action                         | `action_dim D`                       | 7  (`[pos(3) | rotvec(3) | gripper(1)]`)      |
| Action                         | `horizon T`                          | 16                                            |
| Action                         | `n_action_steps` (executed slice)    | 8                                             |
| Obs                            | `n_obs_steps To`                     | 2                                             |
| Past                           | `past_n`                             | 7                                             |
| Tokenizer encoder              | `emb_dim / head_dim / depth`         | 256 / 64 / 2  (→ 4 heads)                     |
| Tokenizer encoder              | `num_registers` = latent horizon     | 8                                             |
| Tokenizer encoder              | `latent_dim`                         | 5  (= `len(quantizer.levels)`)                |
| Tokenizer decoder              | `emb_dim / head_dim / depth`         | 256 / 64 / 4                                  |
| Tokenizer decoder              | `token_dropout_mode`                 | `pow2`                                        |
| Tokenizer decoder              | `use_causal_decoder`                 | true                                          |
| Quantizer (FSQ)                | `levels`                             | `[8, 5, 5, 5, 5]`                             |
| Quantizer                      | codebook size                        | 8·5·5·5·5 = **5000**                          |
| SO(3) aug                      | `p`                                  | 1.0                                           |
| SO(3) aug                      | `max_angle_deg`                      | 30.0                                          |
| SO(3) aug                      | `mode`                               | `left_noise`  (`Q · R`)                       |
| SO(3) aug                      | `augment_position`                   | false                                         |
| Policy backbone                | `embed_dim / n_layers / n_heads`     | 256 / 4 / 4                                   |
| Policy backbone                | `dropout`                            | 0.1                                           |
| Policy backbone                | `vocab_size`                         | 5001  (codebook + `<BOS>`)                    |
| Policy backbone                | `max_seq_len`                        | 9   (BOS + 8 code tokens)                     |
| Policy backbone                | `max_cond_len`                       | 11  = `To + 2 + past_n`                       |
| Policy inference               | `temperature / topk`                 | 1.0 / 10                                      |
| Optim — tokenizer              | `lr`                                 | 5e-5                                          |
| Optim — policy backbone        | `policy_lr`                          | 5e-5                                          |
| Optim — vision/state encoder   | `obs_enc_lr`                         | 1e-5                                          |
| Optim                          | `weight_decay / betas / grad-clip`   | 0.0 / (0.9, 0.95) / 1.0                       |
| LR schedule                    | scheduler / warmup steps             | constant / 100                                |
| EMA                            | power / inv_gamma / max_value        | 0.75 / 1.0 / 0.9999                           |
| Data                           | batch size (tok / policy)            | 256 / 64                                      |
| Data                           | num demos / val ratio                | 500 / 0.1                                     |
| Train                          | num epochs / checkpoint every        | 5001 / 10                                     |
| Train                          | rollout every (policy)               | 200                                           |
| Train                          | precision                            | bf16                                          |
| Checkpoint monitor             | tokenizer / policy                   | `test_reconst_mse` (min) / `mean_success_rate` (max) |

> **Note on drift from earlier docs.** Earlier method docs report `p=0.6`, `max_angle_deg=60`, and FSQ levels `[8,5,5,5]` (codebook 1000). The live YAML at [oat/config/train_oattok_so3aug.yaml](oat/config/train_oattok_so3aug.yaml) now has `p=1.0`, `max_angle_deg=30`, and levels `[8,5,5,5,5]` (codebook 5000). The five-level setting also makes the encoder's `latent_dim = 5`. The vocab in `AutoregressiveModel` is computed from `quantizer.codebook_size + 1` so it tracks this automatically.

---

## 3. Files that define the method

```
oat/
├── config/
│   ├── train_oattok_so3aug.yaml                       [Stage 1 hydra entry]
│   ├── train_oatpolicy_with_enriched_past.yaml        [Stage 2 hydra entry]
│   └── task/
│       ├── tokenizer/libero/libero10.yaml             [tokenizer task — ZarrDataset, action-only]
│       └── policy/libero/libero10_with_past.yaml      [policy task — ZarrDatasetWithPastAction]
│
├── workspace/
│   ├── train_oattok.py                                [Stage 1 training loop]
│   └── train_policy.py                                [Stage 2 training loop + rollout]
│
├── tokenizer/
│   ├── base_tokenizer.py                              [abstract base, from_checkpoint loader]
│   └── oat/
│       ├── tokenizer.py                               [OATTok base — encode/decode/tokenize/detokenize]
│       ├── tokenizer_so3_aug.py                       [OATTokSO3Aug — overrides forward()]
│       ├── encoder/register_encoder.py                [RegisterEncoder w/ causal-last mask]
│       ├── decoder/single_pass_decoder.py             [SinglePassDecoder w/ Matryoshka dropout]
│       ├── quantizer/fsq.py                           [FSQ — factorized scalar quantization]
│       └── augment/so3_action_chunk_aug.py            [SO3ActionChunkAug — left/right/conjugate]
│
├── policy/
│   ├── base_policy.py                                 [abstract policy interface]
│   └── oat_policy_with_enriched_past.py               [OATPolicyWithEnrichedPast]
│
├── model/
│   ├── autoregressive/transformer_cache.py            [AutoregressiveModel — KV-cached AR]
│   └── common/normalizer.py                           [LinearNormalizer — fit on dataset]
│
├── perception/
│   ├── fused_obs_encoder.py                           [FusedObservationEncoder — multimodal fusion]
│   ├── robomimic_vision_encoder.py                    [ResNet18 + SpatialSoftmax]
│   └── state_encoder.py                               [ProjectionStateEncoder]
│
├── dataset/
│   └── zarr_dataset_with_past.py                      [ZarrDatasetWithPastAction — adds past_n slice]
│
└── env_runner/
    └── libero_runner.py                               [LiberoRunner — parallel rollout]
```

| Stage | Entry config                                                                      | Workspace class                                | Output                                            |
|-------|-----------------------------------------------------------------------------------|------------------------------------------------|---------------------------------------------------|
| 1     | [oat/config/train_oattok_so3aug.yaml](oat/config/train_oattok_so3aug.yaml)        | `oat.workspace.train_oattok.TrainOATTokWorkspace` | `*.ckpt` of frozen tokenizer                      |
| 2     | [oat/config/train_oatpolicy_with_enriched_past.yaml](oat/config/train_oatpolicy_with_enriched_past.yaml) (override `policy.action_tokenizer.checkpoint=<stage1.ckpt>`) | `oat.workspace.train_policy.TrainPolicyWorkspace` | `*.ckpt` of full policy (incl. tokenizer weights) |

---

## 4. Stage 1 — `oattok_so3aug` tokenizer architecture

### 4.1 `OATTokSO3Aug` class

Defined in [oat/tokenizer/oat/tokenizer_so3_aug.py](oat/tokenizer/oat/tokenizer_so3_aug.py). It is a minimal subclass of `OATTok` (in [oat/tokenizer/oat/tokenizer.py](oat/tokenizer/oat/tokenizer.py)) that only overrides `forward()` to apply `action_aug` before normalization:

```python
def forward(self, batch):
    samples = batch["action"]                        # (B, 16, 7) raw actions
    if self.action_aug is not None:
        samples = self.action_aug(samples)           # SO(3) aug on rotation slice
    nsamples = self.normalizer["action"].normalize(samples)
    latents = self.encoder(nsamples)                 # (B, 8, latent_dim)
    latents, _ = self.quantizer(latents)             # quantize (STE)
    recons = self.decoder(latents)                   # (B, 16, 7) normalized recon
    return F.mse_loss(recons, nsamples)
```

`encode`, `decode`, `tokenize`, `detokenize`, optimizer factory, and normalizer plumbing are inherited unchanged. At inference time the augmentation is bypassed (`self.training is False`).

### 4.2 `SO3ActionChunkAug` — manifold-aware action augmentation

File: [oat/tokenizer/oat/augment/so3_action_chunk_aug.py](oat/tokenizer/oat/augment/so3_action_chunk_aug.py). Treats the action vector as `[pos(3) | rotvec(3) | gripper(1)]` where the rotation slice is a Rodrigues vector (axis × angle).

Per training batch:

1. Sample a Bernoulli mask `m ~ Bernoulli(p=1.0)` per element. (One mask per **chunk** — all 16 timesteps share the same perturbation `Q`.)
2. Sample `Q ∈ SO(3)`:
   - axis `n ~ Uniform(S²)` (unit-vector via normalized `randn(3)`),
   - angle `θ ~ Uniform(0, max_angle_rad)`  with `max_angle_deg = 30°`,
   - `ω = θ · n`,  `Q = expmap(ω)` via Rodrigues.
3. For each `t`: decode `R(t) = expmap(rotvec(t))`.
4. Apply mode (default `left_noise`):
   - `left_noise`:   `R_aug(t) = Q · R(t)`               — perturb the *world frame*
   - `right_noise`:  `R_aug(t) = R(t) · Q`                — perturb the *end-effector frame*
   - `conjugate`:    `R_aug(t) = Q · R(t) · Qᵀ`           — frame-invariant
5. Re-encode `rotvec_aug(t) = logmap(R_aug(t))`.
6. Replace the rotvec slice (positions and gripper untouched; `augment_position=false`).

The `so3_exp_map` / `so3_log_map` helpers use Rodrigues' formula with Taylor expansions for `θ² < 1e-8`, with intermediate computation upcast to float32 via `_compute_dtype` for numerical stability under bf16. The whole block runs inside `torch.autocast(enabled=False)`.

### 4.3 `LinearNormalizer`

Fit once before training on the unaugmented training actions: `mode='limits'`, `last_n_dims=1` → per-channel min-max scaling to `[-1, +1]`. Augmentation is applied to **raw** actions *before* normalization, so the normalizer's statistics are estimated from the raw distribution. After fitting it is frozen and reused in Stage 2 (`policy.set_normalizer(...)`).

### 4.4 `RegisterEncoder`

File: [oat/tokenizer/oat/encoder/register_encoder.py](oat/tokenizer/oat/encoder/register_encoder.py).

```
samples (B, 16, 7)
   │
   ▼
SampleEmbedder: Linear(7 → 256)                              → (B, 16, 256)
   │
PositionalEmbeddingAdder (1D sincos over T=16)               → (B, 16, 256)
   │
Concatenate with N=8 learnable registers                     → (B, 24, 256)
   │
Transformer (depth=2, 4-head MSA + MLP, pre-norm, SiLU)
  with "causal-last" attention mask:
    · action tokens    : full attention
    · register tokens  : full attention over actions
                         lower-triangular over registers
   │
Slice the 8 register positions                               → (B, 8, 256)
   │
LinearHead 256 → latent_dim=5                                → (B, 8, 5)
```

The causal-last mask forces the registers to act as a sequential information bottleneck — register `r_i` may attend to all action tokens plus `r_0, …, r_i`. This implicitly orders the latents and supports the Matryoshka pow2 token dropout used by the decoder.

### 4.5 `FSQ` — Finite Scalar Quantization

File: [oat/tokenizer/oat/quantizer/fsq.py](oat/tokenizer/oat/quantizer/fsq.py). Levels `[8, 5, 5, 5, 5]` → codebook size **5000**.

Each latent `z ∈ ℝ^5` is quantized **dimension-independently**:

```
For dim i with L_i levels:
  bound(z_i) = (L_i - 1)/2 · tanh(z_i + shift_i)        # squash into valid range
  quantize:   ẑ_i = round_ste(bound(z_i)) / (L_i // 2)  # straight-through round
```

The codebook is implicit (no learnable embedding table). Flat indices are mixed-radix:
```
basis = [1, 8, 8·5, 8·5·5, 8·5·5·5]
flat_index(ẑ) = Σ_i ẑ_i_int · basis[i]      ∈ [0, 5000)
```
`indices_to_embedding(flat_idx)` is the exact inverse. STE: `ẑ = z + (round(z) - z).detach()`. Because every grid point is always reachable, **codebook collapse is structurally impossible**.

### 4.6 `SinglePassDecoder`

File: [oat/tokenizer/oat/decoder/single_pass_decoder.py](oat/tokenizer/oat/decoder/single_pass_decoder.py).

```
latents (B, 8, 5)
   │
LinearLayer 5 → 256                                      → (B, 8, 256)
   │
PositionalEmbeddingAdder (1D sincos over latent horizon) → (B, 8, 256)
   │
MaskedNestedDropout (mode='pow2'):
   k ~ Uniform({1, 2, 4, 8}); tokens at positions ≥ k replaced
   with a learnable <MASK> embedding. Acts as Matryoshka training —
   at inference, `eval_keep_k` chooses any of {1,2,4,8} for a
   latency/quality knob.
   │
Generate sample-side positional queries (1, 256, 16) and broadcast
   │
nn.TransformerDecoder (depth=4, 4-head, GELU, norm_first=True):
   · tgt        = positional queries  (size 16)
   · memory     = post-dropout latents (size 8)
   · tgt_mask   = causal lower-triangular (use_causal_decoder=true)
   · cross-attn (tgt ← memory) has no mask
   │
LinearHead 256 → 7                                       → (B, 16, 7)
```

Causal self-attention on the output side enforces temporal coherence; cross-attention is unrestricted so each output step can pull from any active register.

---

## 5. Stage 2 — `OATPolicyWithEnrichedPast` architecture

File: [oat/policy/oat_policy_with_enriched_past.py](oat/policy/oat_policy_with_enriched_past.py).

### 5.1 Observation encoder — `FusedObservationEncoder`

File: [oat/perception/fused_obs_encoder.py](oat/perception/fused_obs_encoder.py). Iterates over `shape_meta["obs"]`, instantiates a per-modality encoder, and concatenates outputs along the feature axis:

- **`RobomimicRgbEncoder`** ([oat/perception/robomimic_vision_encoder.py](oat/perception/robomimic_vision_encoder.py)): robomimic `VisualCore` with `ResNet18Conv` backbone (BatchNorm → GroupNorm), `SpatialSoftmax` pooling (32 keypoints), then flattened. `crop_shape=[76, 76]` is a random crop during training and a center crop at eval. Applied independently to `agentview_rgb` and `robot0_eye_in_hand_rgb`.
- **`ProjectionStateEncoder`** ([oat/perception/state_encoder.py](oat/perception/state_encoder.py)): concatenates `robot0_eef_pos[3]`, `robot0_eef_quat[4]`, `robot0_gripper_qpos[2]`, `task_uid[1]` (total 10 dims) and applies `nn.Identity` (since `out_dim: null`).

Output: `(B, To=2, d)` with `d = (Σ vision dims) + (Σ state dims) = obs_feature_dim`.

### 5.2 Enriched-past condition assembly — the core innovation

`_build_condition(obs_features, past_actions)` constructs a condition sequence of length **`To + 2 + past_n = 2 + 2 + 7 = 11`**:

```python
norm_past = self.action_normalizer["action"].normalize(past_actions)   # (B, 7, 7)

# Higher-order features from the last 3 past steps
a_t1, a_t2, a_t3 = norm_past[:, -1], norm_past[:, -2], norm_past[:, -3]
acc  = a_t1 -        a_t2                                              # (B, 7)
jerk = a_t1 - 2.0 *  a_t2 + a_t3                                       # (B, 7)

acc_feat  = self.acc_proj(acc)        # (B, d)  Linear(7,d) → GELU → Linear(d,d)
jerk_feat = self.jerk_proj(jerk)      # (B, d)  Linear(7,d) → GELU → Linear(d,d)
raw_feat  = self.raw_proj(norm_past)  # (B, 7, d)  shared Linear(7,d) → GELU → Linear(d,d) per step

explicit = torch.stack([acc_feat, jerk_feat], dim=1)                   # (B, 2, d)
cond     = torch.cat([obs_features, explicit, raw_feat], dim=1)        # (B, 11, d)
```

Why three separate MLPs (not one shared):
- `norm_past ∈ [-1, +1]^7` (raw values, O(1)).
- `acc = a_{t-1} - a_{t-2}` is the difference of two normalized vectors — typically O(0.01–0.1).
- `jerk = a_{t-1} - 2 a_{t-2} + a_{t-3}` is O(0.001–0.01).

A shared projection would be dominated by the raw-past path or would need to learn input-conditioned gain. Three small projection MLPs decouple the scales for negligible parameter cost.

All differencing is done in **normalized** action space, matched to how the tokenizer ingests actions (so the past representation lives on the same scale the tokenizer was trained against).

### 5.3 AR backbone — `AutoregressiveModel`

File: [oat/model/autoregressive/transformer_cache.py](oat/model/autoregressive/transformer_cache.py). Configured from the policy YAML:

```
vocab_size   = codebook_size + 1 = 5001        (+1 for <BOS>)
max_seq_len  = latent_horizon + 1 = 9          (BOS + 8 code tokens)
max_cond_len = 11
cond_dim     = obs_feature_dim
n_emb        = 256;  n_layer = 4;  n_head = 4;  p_drop_emb = p_drop_attn = 0.1
```

Per-layer structure (pre-norm with RMSNorm):

```
Block:
  x = x + CausalSelfAttention( RMSNorm(x), layer_past = KV cache )
  x = x + CrossAttention     ( RMSNorm(x), memory = cond_encoded, memory_kv = memory cache )
  x = x + MLP                ( RMSNorm(x) )   # n_emb → 4·n_emb → n_emb, GELU
```

Inputs:
- AR side: `tok_emb(tokens) + tok_pos_emb` over the sequence of length up to `max_seq_len`.
- Condition side: `cond_emb(cond) + cond_pos_emb` then a 2-layer MLP encoder (`Linear → Mish → Linear`) produces the **memory** consumed by every cross-attention layer.

Output head: `Linear → vocab_size`, weight-tied to `tok_emb`.

**Generation** is KV-cached:
1. Process the `<BOS>` prefix once, caching per-layer K/V for both self- and cross-attention.
2. For each of the 8 generation steps: embed the just-sampled token, run blocks against the cached past, apply temperature + top-k sampling.

Cross-attention to `cond` is unmasked; self-attention over the AR sequence is causal.

### 5.4 Trainable parameters

`__init__` freezes the action tokenizer:

```python
for p in action_tokenizer.parameters():
    p.requires_grad_(False)
action_tokenizer.eval()
```

Trained parameters: `obs_encoder` (vision + state) + `acc_proj` + `jerk_proj` + `raw_proj` + `model` (AR backbone).

### 5.5 Optimizer — two LR groups

`get_optimizer(...)` splits parameters into four groups so weight decay applies only to ≥2D tensors and the visual encoder gets a smaller LR:

| Group              | Members                              | LR           | Weight decay     |
|--------------------|--------------------------------------|--------------|------------------|
| policy decay       | 2D params in `model`, `acc_proj`, `jerk_proj`, `raw_proj` | `policy_lr = 5e-5` | `wd`             |
| policy nodecay     | 1D params (biases, norms)            | `policy_lr`  | 0                |
| encoder decay      | 2D params in `obs_encoder`           | `obs_enc_lr = 1e-5` | `wd`             |
| encoder nodecay    | 1D params in `obs_encoder`           | `obs_enc_lr` | 0                |

Currently `wd = 0` and `betas = (0.9, 0.95)`.

### 5.6 Training loss

```python
with torch.no_grad():
    action_tokens = self.action_tokenizer.tokenize(batch["action"])    # (B, 8) ∈ [0, 5000)

features = self.obs_encoder(batch["obs"])                              # (B, 2, d)
cond     = self._build_condition(features, batch["past_action"])       # (B, 11, d)

action_tokens = torch.cat([BOS, action_tokens], dim=1)                 # (B, 9)
logits        = self.model(action_tokens[:, :-1], cond=cond)           # (B, 8, 5001)
loss          = F.cross_entropy(logits.flatten(0,1), action_tokens[:, 1:].flatten())
```

Teacher-forcing cross-entropy over the next code token. Tokenizer is fully frozen.

### 5.7 Inference — `predict_action(obs_dict)`

Called by `LiberoRunner` once per `n_action_steps = 8` env steps (via `MultiStepWrapper`):

```python
features = self.obs_encoder(obs_dict)                                  # (B, 2, d)

# Lazily init/reset past buffer (e.g. at the start of a new episode after reset())
if self._past_buffer is None or shape/device mismatch:
    self._past_buffer = zeros(B, past_n=7, action_dim=7)

cond = self._build_condition(features, self._past_buffer)              # (B, 11, d)

action_tokens = self.model.generate(
    prefix=BOS,
    cond=cond,
    max_new_tokens=8,                                                  # use_k_tokens, ≤ max_seq_len
    temperature=1.0,
    top_k=10,
)[:, 1:]                                                               # drop BOS  → (B, 8)
action_tokens = action_tokens.clamp(0, codebook_size - 1)

with torch.inference_mode():
    action_pred = self.action_tokenizer.detokenize(action_tokens)      # (B, 16, 7)

action = action_pred[:, : self.n_action_steps]                         # (B, 8, 7) executed slice

# Past buffer update — predicted actions (NOT executed/env-returned)
if n_action_steps >= past_n:                                           # 8 ≥ 7 → yes
    self._past_buffer = action_pred[:, n_action_steps - past_n : n_action_steps]
    # = action_pred[:, 1:8]   the most recent past_n=7 of the executed slice
else:
    self._past_buffer = torch.cat([self._past_buffer[:, n_action_steps:],
                                   action_pred[:, : n_action_steps]], dim=1)
```

`reset()` is called at episode start and sets `_past_buffer = None` so the next `predict_action` re-initializes it with zeros.

The past buffer storing *predicted* (commanded) actions — not env-returned states — preserves the train/test distribution match: during training `past_action` is the demo's commanded action stream, and at rollout the closest match is the policy's own most recent emissions.

---

## 6. Dataset & training loops

### 6.1 Stage 1 — `ZarrDataset` (action-only)

`obs_keys: []`, `n_obs_steps: 0`, `n_action_steps: 16`. One sample = one 16-step chunk. Sliding window with `pad_after = 15`. `val_ratio = 0.1`, `num_demo = 500`. Evaluation cadence:

- Every 10 epochs: validation MSE on **unaugmented**, normalized actions.
- Every 10 epochs: a held-out sample `tokenizer.autoencode(...)` computes `test_reconst_mse` in *un-normalized* space.
- Top-3 checkpoints by `test_reconst_mse` (min); last checkpoint always saved.

### 6.2 Stage 2 — `ZarrDatasetWithPastAction`

File: [oat/dataset/zarr_dataset_with_past.py](oat/dataset/zarr_dataset_with_past.py). Sliding window with `pad_before = max(n_obs_steps - 1, 0) + past_n = 1 + 7 = 8` and `pad_after = max(n_action_steps - 1, 0) = 15`. Each sample yields:

```
obs         : dict of (To=2, ...) frames per rgb/state port
action      : (16, 7)            chunk to predict
past_action : (past_n=7, 7)      ordered [a_{t-7}, ..., a_{t-1}]
```

Padding at episode starts is zero-filled by the underlying `SequenceSampler`. Evaluation cadence:

- Every 10 epochs: validation cross-entropy and sampled-action MSE (decode predicted tokens, compare to GT).
- Every `rollout_every = 200` epochs: full `LiberoRunner` rollout over 500 init configs across 10 tasks, in 20 parallel envs (`mean_success_rate` is the monitor).
- Top-3 checkpoints by `mean_success_rate` (max); last checkpoint always saved.

---

## 7. Design rationale — why these choices

### 7.1 Why SO(3) augmentation (vs. Gaussian noise on the rotvec)

Robot orientation lives on `SO(3)`, not `ℝ³`. Adding noise directly to the Rodrigues vector is **not** equivalent to perturbing the rotation (near `‖ω‖ ≈ π` the encoding is multivalued). The standard manifold-aware perturbation is

```
R_aug = Q · R                  (left-multiplication; world-frame rotation)
```

which is closed in `SO(3)`. Re-encoding via `logmap` produces a valid rotvec.

A single `Q` per **chunk** preserves the relative motion across the 16 steps — the trajectory is rigidly rotated, not jittered — so the encoder is forced to encode the *relative geometry* rather than the absolute reference frame. For a finite codebook this is a large effective increase in coverage of the orientation manifold without changing the dataset size.

### 7.2 Why `mode = left_noise`

| Mode           | Formula           | Interpretation                                                            |
|----------------|-------------------|----------------------------------------------------------------------------|
| `left_noise`   | `Q · R`           | Perturb the *world frame*: same intended grip, world tilted. Task-preserving. |
| `right_noise`  | `R · Q`           | Perturb the *end-effector frame*: a different gripper roll/pitch/yaw.       |
| `conjugate`    | `Q · R · Qᵀ`      | Frame-invariant perturbation; symmetric and more aggressive.               |

For end-effector action representations in a task frame, `left_noise` simulates "the camera/table is tilted relative to where the demo was recorded" without changing what the robot should *do*. Consistent with `augment_position: false` — under a pure world-frame perturbation, positions would also rotate, but keeping positions fixed deliberately decouples the orientation from the position channel, forcing the encoder to learn **rotation invariance for the orientation slice specifically**.

### 7.3 Why `p = 1.0` and `max_angle_deg = 30°`

`p = 1.0` means every training sample is augmented — with a 5000-entry codebook the redundancy is large enough that anchoring on un-augmented data is no longer required. `30°` is the result of a sweep (commented alternatives `10°`/`60°` are left in the YAML); large enough to span typical across-demo orientation variation while small enough to preserve manipulability of the chunks.

### 7.4 Why the *enriched past* on the policy side

LIBERO obs gives `eef_pos`, `eef_quat`, `gripper_qpos`, `task_uid` and two RGB views at `To=2`. From those alone you get:

- A position snapshot (post-controller, slightly lagged behind commands).
- One backward-Euler velocity estimate per dimension.

Past *actions* are the *commanded* values — noise-free, instantaneous — and encode command inertia (the policy was already driving in a direction even if the state hasn't caught up yet). For contact-rich manipulation this is often more informative for the *next* command than the post-contact observed state.

### 7.5 Why explicit `acc` and `jerk` in addition to raw past

A cross-attention layer over the 7-token raw-past sequence *could* in principle learn the linear combinations `a_{t-1} - a_{t-2}` and `a_{t-1} - 2 a_{t-2} + a_{t-3}` by allocating heads to position-shifted subtraction. But this means *learning the subtraction pattern* from gradient signal alone, and burning attention heads on it.

By pre-computing the differences and presenting them as two extra condition tokens whose embedding is exactly the higher-order derivative, the model gets that inductive bias for free. The cost is negligible parameters; the saved capacity goes toward the genuinely hard cross-modal grounding of language ↔ vision ↔ action.

### 7.6 Why the past buffer stores *predicted* (not env-returned) actions

At training time `past_action` is the demo's commanded stream. At rollout the closest matching distribution is the policy's own most recent predictions — `action_pred[:, 1:8]` is exactly what was sent to the env's controller (modulo `MultiStepWrapper`). Storing those keeps train and test conditions on the same data distribution. Substituting `info["last_action"]` (post-controller realized motion) or proprioceptive derivatives would introduce a clean-vs-noisy gap between train and rollout.

### 7.7 Why nothing else is changed from the baseline

The encoder, decoder, quantizer, observation encoder, and AR backbone are inherited verbatim from the baseline `oattok` + `oatpolicy` pipeline. The contributions are confined to:

1. **Training signal of the tokenizer** — SO(3) augmentation of action chunks.
2. **Condition signal of the policy** — enriched past (raw + acc + jerk).

Keeping everything else fixed (a) makes ablation attribution clean, (b) lets the SO(3)-trained tokenizer plug into either baseline or enriched-past policy, and (c) avoids hyperparameter co-tuning that would muddy what we can claim.
