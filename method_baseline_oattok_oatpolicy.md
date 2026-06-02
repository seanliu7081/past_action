# Baseline Method — OATTok Tokenizer + OATPolicy

This document describes the **baseline** method in the `oat/` codebase: a two-stage action-chunking pipeline where (1) `OATTok` learns a discrete codebook over 16-step action chunks with no augmentation, and (2) `OATPolicy`, an autoregressive transformer, generates those codes conditioned **only** on the recent observation history (`To = 2` frames). It is the reference against which the SO(3)-augmented + enriched-past variant is measured.

Stage 1 produces a frozen tokenizer; Stage 2 trains the policy on top of it.

---

## 1. Overview

```
                                ╔══════════════════════════════════════════════╗
                                ║  STAGE 1 — TOKENIZER (action autoencoder)    ║
                                ╚══════════════════════════════════════════════╝
                                            (action chunks only)

  raw actions (B, 16, 7)
        │
        ▼
  normalize ──► RegisterEncoder ──► FSQ ──► tokens (B, 8)
                                              │
                                              ▼
                                    SinglePassDecoder
                                              │
                                              ▼
                                    MSE loss vs. normalized actions


                                ╔══════════════════════════════════════════════╗
                                ║  STAGE 2 — POLICY (autoregressive over codes) ║
                                ╚══════════════════════════════════════════════╝

  obs_dict {rgb×2, state×4}
      │
      ▼
  FusedObservationEncoder        (B, To=2, d)   ← condition sequence, length 2
      │
      ▼
  AutoregressiveModel (causal self-attn + cross-attn to cond)
      │
      ▼
  next-token logits over codebook (size 1001 incl. <BOS>)

  At inference: AR-generate 8 tokens with KV cache → detokenize via frozen
                tokenizer → execute first 8 of 16 reconstructed actions
                (receding horizon).
```

**Key numbers:** action dim `D=7`, action horizon `T=16`, observation horizon `To=2`, executed horizon `n_action_steps=8`, FSQ levels `[8, 5, 5, 5]` → codebook size **1000**, latent horizon (= register count) **8**, condition length **2** (only `To` frames; no past actions), AR backbone width 256 / depth 4 / 4 heads.

---

## 2. Code structure

Files that define this method (relative to `/workspace/oat/`):

```
oat/
├── config/
│   ├── train_oattok.yaml                     [Stage 1 hydra entry point]
│   ├── train_oatpolicy.yaml                  [Stage 2 hydra entry point]
│   └── task/
│       ├── tokenizer/libero/libero10.yaml    [tokenizer task — ZarrDataset, no past]
│       └── policy/libero/libero10.yaml       [policy task — ZarrDataset, no past_action]
│
├── workspace/
│   ├── train_oattok.py                       [Stage 1 training loop]
│   └── train_policy.py                       [Stage 2 training loop + rollout]
│
├── tokenizer/
│   ├── base_tokenizer.py                     [abstract base, from_checkpoint loader]
│   └── oat/
│       ├── tokenizer.py                      [OATTok — base tokenizer class]
│       ├── encoder/register_encoder.py       [RegisterEncoder, causal-last mask]
│       ├── decoder/single_pass_decoder.py    [SinglePassDecoder, Matryoshka dropout]
│       └── quantizer/fsq.py                  [FSQ — factorized scalar quantization]
│
├── policy/
│   ├── base_policy.py                        [abstract policy interface]
│   └── oatpolicy.py                          [OATPolicy]
│
├── model/
│   ├── autoregressive/transformer_cache.py   [AutoregressiveModel — KV-cached AR]
│   └── common/normalizer.py                  [LinearNormalizer — fit on dataset]
│
├── perception/
│   ├── fused_obs_encoder.py                  [FusedObservationEncoder — multimodal fusion]
│   ├── robomimic_vision_encoder.py           [ResNet18 + SpatialSoftmax]
│   └── state_encoder.py                      [ProjectionStateEncoder]
│
├── dataset/
│   └── zarr_dataset.py                       [ZarrDataset — no past_action]
│
└── env_runner/
    └── libero_runner.py                      [LiberoRunner — parallel rollout for SR metric]
```

Entry points:

| Stage | Config (relative) | Workspace class | Output |
|---|---|---|---|
| 1 | `config/train_oattok.yaml` | `oat.workspace.train_oattok.TrainOATTokWorkspace` | `*.ckpt` containing the frozen tokenizer |
| 2 | `config/train_oatpolicy.yaml` (override `policy.action_tokenizer.checkpoint=<stage1.ckpt>`) | `oat.workspace.train_policy.TrainPolicyWorkspace` | `*.ckpt` containing the full policy (tokenizer + AR backbone) |

---

## 3. Neural network framework

### 3.1 Tokenizer side — `OATTok`

File: [oat/tokenizer/oat/tokenizer.py](oat/oat/tokenizer/oat/tokenizer.py). The full forward pass is:

```python
def forward(self, batch):
    samples = batch["action"]                        # (B, 16, 7) raw actions
    nsamples = self.normalizer["action"].normalize(samples)
    latents = self.encoder(nsamples)                 # (B, 8, latent_dim)
    latents, _ = self.quantizer(latents)             # quantize via FSQ (STE)
    recons = self.decoder(latents)                   # (B, 16, 7) normalized recon
    return F.mse_loss(recons, nsamples)
```

Standard interface methods inherited from `BaseTokenizer`:

| Method | Purpose |
|---|---|
| `encode(samples)` | Normalize → encode → quantize. Returns `(latents, tokens)`. |
| `decode(latents, eval_keep_k=None)` | Decode → unnormalize. `eval_keep_k` allows Matryoshka-style truncated decoding. |
| `autoencode(samples)` | `encode` then `decode`. |
| `tokenize(samples)` | Return only the discrete token indices `(B, 8)`. |
| `detokenize(tokens)` | `indices_to_embedding` then `decode`. |
| `from_checkpoint(ckpt)` | Class method that reads the saved hydra config and re-instantiates. Used by the policy to load a frozen tokenizer. |

The constructor wires three submodules + a `LinearNormalizer`, and exposes `latent_horizon = decoder.latent_horizon` (used by the policy to size its AR sequence).

#### 3.1.1 `RegisterEncoder`

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

Forward pass:

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
   │ Register tokens: full over actions,
   │                  causal over registers (lower-triangular among themselves)
   └──────────────────────────────────┘
  │
  ▼
Slice out the 8 register positions       → (B, 8, 256)
  │
  ▼
LinearHead: 256 → 4                      → (B, 8, 4)  latents
```

The causal-last mask makes the registers an ordered bottleneck: `r_i` can only see actions plus `r_0, ..., r_i`. This supports inference-time Matryoshka truncation (a model trained with this mask still produces meaningful first-few latents in isolation).

#### 3.1.2 `FSQ` — Finite Scalar Quantization

File: [oat/tokenizer/oat/quantizer/fsq.py](oat/oat/tokenizer/oat/quantizer/fsq.py). Configured with `levels: [8, 5, 5, 5]` → codebook size **1000**.

Each latent `z ∈ ℝ⁴` is quantized **dimension-independently**:

```
For dim i with L_i levels:
  bound(z_i) = (L_i - 1)/2 · tanh(z_i + shift_i)        # squashes into valid range
  ẑ_i      = round_ste(bound(z_i)) / (L_i // 2)         # straight-through round, normalized
```

The codebook is implicit (no learnable embedding table). A flat index is computed via mixed-radix encoding:

```
basis = [1, 8, 40, 200]
flat_index(ẑ) = Σ_i ẑ_i_int · basis[i]      ∈ [0, 1000)
```

`indices_to_embedding(flat_idx)` is the exact inverse, used at decode time. The straight-through estimator `ẑ = z + (round(z) - z).detach()` lets gradients flow through quantization. The yaml leaves `drop_quant_p`, `corrupt_tokens_p` at their defaults (0), so no quantization-side regularization is active.

**Property:** since every code is a fixed dimension-product, no codebook collapse is possible — every code is always reachable.

#### 3.1.3 `SinglePassDecoder`

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
LinearLayer 4 → 256                                  → (B, 8, 256)
  │
  ▼
PositionalEmbeddingAdder (1D sincos)                 → (B, 8, 256)
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
LinearHead: 256 → 7                                  → (B, 16, 7)  reconstructed (normalized) actions
```

Causal self-attention on the output side enforces temporal coherence; cross-attention is unrestricted so each output step can pull from any active register.

### 3.2 Policy side — `OATPolicy`

File: [oat/policy/oatpolicy.py](oat/oat/policy/oatpolicy.py).

#### 3.2.1 Observation encoder

`FusedObservationEncoder` ([oat/perception/fused_obs_encoder.py](oat/oat/perception/fused_obs_encoder.py)) instantiates and concatenates:

- **`RobomimicRgbEncoder`** ([oat/perception/robomimic_vision_encoder.py](oat/oat/perception/robomimic_vision_encoder.py)) — robomimic `VisualCore` with `ResNet18Conv` backbone (BatchNorm replaced with GroupNorm), `SpatialSoftmax` pooling (32 keypoints), flattened. `crop_shape=[76, 76]` does random crop in training, center crop at eval. Applied independently to `agentview_rgb` and `robot0_eye_in_hand_rgb`.
- **`ProjectionStateEncoder`** ([oat/perception/state_encoder.py](oat/oat/perception/state_encoder.py)) — concatenates `robot0_eef_pos[3]`, `robot0_eef_quat[4]`, `robot0_gripper_qpos[2]`, `task_uid[1]` → 10 dims, then `nn.Identity` projection (`out_dim: null`).

Output shape: `(B, To=2, d)` where `d = (Σ vision dims) + (Σ state dims)` is the fused per-step feature dimension.

#### 3.2.2 Conditioning

The condition fed into the AR backbone is simply the obs feature stack — no past-action augmentation, no derived features:

```python
features = self.obs_encoder(batch["obs"])    # (B, To=2, d)
cond     = features                          # (B, 2, d)
```

So `max_cond_len = n_obs_steps = 2`.

#### 3.2.3 AR backbone — `AutoregressiveModel`

File: [oat/model/autoregressive/transformer_cache.py](oat/oat/model/autoregressive/transformer_cache.py). Configured as:

```yaml
embed_dim: 256
n_layers: 4
n_heads:  4
dropout:  0.1
# vocab_size    = codebook_size + 1   = 1001  (+1 for <BOS>)
# max_seq_len   = latent_horizon + 1  = 9     (BOS + 8 code tokens)
# max_cond_len  = n_obs_steps         = 2
# cond_dim      = obs_feature_dim
```

Per-layer structure (pre-norm, RMSNorm):

```
Block:
  x = x + CausalSelfAttention( RMSNorm(x),   layer_past=KV cache )
  x = x + CrossAttention     ( RMSNorm(x),   memory=cond_encoded, memory_kv=memory cache )
  x = x + MLP                ( RMSNorm(x) )   # n_emb → 4·n_emb → n_emb, GELU
```

Inputs:

- `tok_emb(tokens) + tok_pos_emb` for the AR sequence (learned positional embeddings, size 9).
- `cond_emb(cond) + cond_pos_emb` then a 2-layer MLP (`Linear → Mish → Linear`) to produce the **memory** used by every cross-attention layer.

Output head is `Linear → vocab_size`, weight-tied to `tok_emb`.

**Generation** is KV-cached:
1. Process the BOS prefix once, caching K/V per layer for both self- and cross-attention.
2. For each of the 8 generation steps, embed the just-sampled token, run blocks with cached past, apply top-k temperature sampling on the resulting logits.

Cross-attention to `cond` is unmasked (any AR position can look at any condition position). Self-attention is causal.

---

## 4. Methodology

### 4.1 Stage 1 — Tokenizer training

**Data.** `ZarrDataset` ([oat/dataset/zarr_dataset.py](oat/oat/dataset/zarr_dataset.py)) on `data/libero/libero10_N500.zarr`, **action-only** (`obs_keys: []`, `n_obs_steps: 0`, `n_action_steps: 16`). One sample = one 16-step action chunk. Sliding window over each demo with `pad_after = 15`. `val_ratio = 0.1`.

Sample assembly (from `_sample_to_data`):

```python
start = max(To - 1, 0)        # = 0 since To = 0
end   = start + Ta             # = 16
return {
    "obs":    {},              # empty — action-only
    "action": sample[action_key][start:end],   # (16, 7)
}
```

**Normalizer.** `LinearNormalizer.fit(...)` on the training actions, `mode='limits'`, `last_n_dims=1` → per-channel min-max scaling to `[-1, +1]`. Frozen for the rest of training.

**Loss.** `F.mse_loss(decoder(quantize(encoder(normalize(action)))), normalize(action))` — standard reconstruction loss in normalized space. The FSQ straight-through estimator passes gradients through quantization.

**Optimization.** AdamW (`lr=5e-5`, `betas=(0.9, 0.95)`, `weight_decay=0`); 2D-only weight decay split (no-op with `weight_decay=0`). LR scheduler `constant` with `lr_warmup_steps=100`. EMA with `power=0.75`, `inv_gamma=1.0`, `max_value=0.9999`. Trained up to `5001` epochs, `gradient_accumulate_every=1`, `batch_size=256`, `num_workers=4`, `bf16`.

**Evaluation cadence.** Every 10 epochs: validation MSE on the held-out 10%, and a reconstruction MSE on a sampled batch (`tokenizer.autoencode(samples)` then MSE in un-normalized space, stored as `test_reconst_mse`). Top-3 checkpoints by `test_reconst_mse` (mode=min). Last checkpoint always saved.

### 4.2 Stage 2 — Policy training

**Data.** `ZarrDataset` on the same zarr (note: **`zarr_dataset.py`**, not the with-past variant). Sliding window with `pad_before = max(n_obs_steps − 1, 0) = 1` and `pad_after = max(n_action_steps − 1, 0) = 15`. Each sample yields:

```
obs    : dict of {(B,) To=2 frames of each rgb/state port}
action : (B, 16, 7)   the chunk to predict
```

No `past_action` field.

**Loss.** Frozen-tokenizer cross-entropy on next-action-token prediction (teacher forcing):

```python
with torch.inference_mode():                       # frozen tokenizer
    action_tokens = self.action_tokenizer.tokenize(batch["action"])   # (B, 8)

features = self.obs_encoder(batch["obs"])          # (B, 2, d)

action_tokens = torch.cat([BOS, action_tokens], dim=1)  # (B, 9)
logits = self.model(action_tokens[:, :-1], cond=features)  # (B, 8, 1001)
loss   = F.cross_entropy(logits.flatten(0,1), action_tokens[:, 1:].flatten())
```

The **tokenizer is fully frozen** (`requires_grad_(False)` + `.eval()`); only the observation encoder and the AR backbone are trained.

**Optimization.** AdamW with two LR groups — `policy_lr=5e-5` for AR backbone, `obs_enc_lr=1e-5` for the visual/state encoders. `weight_decay=0`, `betas=(0.9, 0.95)`. Same constant scheduler + warmup, same EMA settings. `batch_size=64`, `num_workers=2`.

**Evaluation cadence.** Every 10 epochs: validation cross-entropy, sampled-action MSE. Every `rollout_every=200` epochs: full `LiberoRunner` rollout over 500 init configs across 10 tasks, 20 parallel envs. Top-3 checkpoints by `mean_success_rate` (mode=max).

### 4.3 Inference

`predict_action(obs_dict)` (called by `LiberoRunner` once per `n_action_steps=8` env steps via `MultiStepWrapper`):

```python
features = self.obs_encoder(obs_dict)               # (B, 2, d)

action_tokens = AutoregressiveModel.generate(
    prefix=BOS,
    cond=features,
    max_new_tokens=8,    # = use_k_tokens, clamped to max_seq_len
    temperature=1.0,
    top_k=10,
)[:, 1:]                                            # drop BOS → (B, 8)
action_tokens = action_tokens.clamp(0, self.bos_id - 1)

action_pred = self.action_tokenizer.detokenize(action_tokens)   # (B, 16, 7)
action      = action_pred[:, : n_action_steps]                  # (B, 8, 7)  executed slice
```

`reset()` is the default no-op on `BasePolicy` — there is no internal state to clear between episodes for the baseline.

### 4.4 Hyperparameters

| Group | Param | Stage 1 (tokenizer) | Stage 2 (policy) |
|---|---|---|---|
| Action | `horizon T` | 16 | 16 |
| Action | `action_dim D` | 7 | 7 |
| Action | `n_action_steps` | — | 8 (executed slice) |
| Obs | `n_obs_steps To` | 0 (action-only) | 2 |
| Tokenizer | `latent_dim` | 4 (= len(levels)) | inherited |
| Tokenizer | `num_registers / latent_horizon` | 8 | inherited |
| Tokenizer | `emb_dim / head_dim / depth (enc)` | 256 / 64 / 2 | — |
| Tokenizer | `emb_dim / head_dim / depth (dec)` | 256 / 64 / 4 | — |
| Tokenizer | `token_dropout_mode` | pow2 | — |
| Quantizer | `FSQ levels` | [8, 5, 5, 5] | inherited |
| Quantizer | codebook size | 1000 | 1000 |
| Policy | `embed_dim / n_layers / n_heads` | — | 256 / 4 / 4 |
| Policy | `dropout` | — | 0.1 |
| Policy | `temperature / topk` | — | 1.0 / 10 |
| Policy | `max_cond_len` | — | 2 |
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

## 5. Design rationale (as encoded in the architecture)

The baseline is the *bare* OAT recipe: discrete action codes via a register-bottleneck autoencoder, followed by an AR transformer conditioned on a short observation stack. Three design choices are worth flagging because they are inherited unchanged by the "ours" variant:

1. **Action-only tokenizer.** The codebook learns the geometry of *action chunks* in isolation, agnostic to observations. This decouples representation learning of motion primitives from the higher-level control problem, and lets the same tokenizer serve multiple downstream policies (or tasks) without retraining.

2. **FSQ over learned VQ.** Avoids codebook-collapse and dead-codes by construction: every index in `[0, 1000)` corresponds to a fixed implicit codebook entry, so the codebook usage histogram is bounded by the distribution of latents, not by gradient dynamics on a learnable embedding matrix.

3. **Causal-last attention in the encoder + causal self-attention + Matryoshka token dropout in the decoder.** Together these make the latents *ordered* — earlier registers contain coarser/more-essential information, later ones add refinement. This is what makes `eval_keep_k < latent_horizon` meaningful: at deploy time you can spend fewer tokens for faster decoding with graceful quality degradation.

4. **AR over the codes with cross-attention to obs.** Each generated code token sees all prior code tokens (causal self-attn) and all obs frames (unmasked cross-attn) — i.e., the cross-attention is *prefix-LM-like* over the condition. The KV cache amortizes both during generation: condition K/V is computed once per call, code K/V is built up incrementally over the 8 generation steps.

What the baseline does **not** include — and what the "ours" variant adds — is information about the *recent action history*. The baseline assumes the policy can recover any needed temporal context from the two-frame observation stack alone.
