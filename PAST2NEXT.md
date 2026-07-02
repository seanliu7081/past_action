# Past2Next — Architecture Summary

**Past2Next** is a two-stage action model for robot manipulation (LIBERO-10 here):

1. **Stage 1 — Action tokenizer** (`train_oattok_so3aug.yaml`): a Matryoshka-style
   autoencoder that compresses a 16-step chunk of continuous 7-DoF actions into
   **8 discrete tokens** (FSQ), and reconstructs it. Trained with reconstruction
   MSE + SO(3) rotation augmentation.
2. **Stage 2 — Autoregressive policy** (`train_oatpolicy_with_enriched_past.yaml`):
   a causal transformer that, conditioned on an **enriched past** (current
   observations + raw past actions + explicit acceleration/jerk), predicts the
   **8 tokens of the next action chunk** left-to-right, then decodes them with the
   frozen tokenizer and executes them (receding horizon).

The name captures the core idea: **turn the (enriched) *Past* into the *Next*
action chunk**, in a discrete token space defined by a learned action tokenizer.

All paths below are relative to the repo root (`/workspace/oat`).

---

## 1. End-to-end data flow

```
                        STAGE 1 (train once)                          STAGE 2 (train with frozen tokenizer)
   ┌──────────────────────────────────────────────┐    ┌───────────────────────────────────────────────────────┐
   │ action chunk  [B,16,7]                         │    │  obs (2 frames) ─► FusedObservationEncoder ─► [B,2,138] │
   │   └─ SO(3) aug (rot only) ─► normalize         │    │  past 7 actions ─► raw_proj      ──────────► [B,7,d]    │
   │      └─ RegisterEncoder ─► 8 latents [B,8,5]   │    │  acc, jerk (finite diffs) ─► acc/jerk_proj ─► [B,2,d]   │
   │         └─ FSQ quantize ─► 8 tokens [B,8]      │    │        └──── concat ─► cond memory [B, 2+2+7=11, d]     │
   │            └─ SinglePassDecoder ─► recon [B,16,7]│  │                                                         │
   │               loss = MSE(recon, chunk)         │    │  AR transformer: [<BOS>, z1..z8]  (causal self-attn)   │
   └──────────────────────────────────────────────┘    │     cross-attends to cond memory                        │
                    │ produces                           │     predicts next token ─► 8 tokens over vocab 5001     │
                    ▼                                    │        └─ frozen tokenizer.detokenize ─► [B,16,7]        │
        frozen tokenizer checkpoint  ───────────────────►│           execute first n_action_steps=8               │
        (defines vocab + encode/decode)                  │           feed executed actions back into past buffer   │
                                                         └───────────────────────────────────────────────────────┘
```

**Two-stage dependency:** the policy config has `action_tokenizer.checkpoint: ???`
(`train_oatpolicy_with_enriched_past.yaml:43`) — you must train Stage 1 first and
pass its checkpoint path as an override to Stage 2.

---

## 2. Stage 1 — The action tokenizer (`train_oattok_so3aug.yaml`)

An **action-only autoencoder** with a discrete bottleneck. It never sees
observations — it just learns a compact, quantized, reconstructable code for
16-step action chunks. Class: `OATTokSO3Aug` (subclass of `OATTok`,
`oat/tokenizer/oat/tokenizer_so3_aug.py`).

### 2.1 Components

| Piece | Class / file | What it does |
|---|---|---|
| Encoder | `RegisterEncoder` (`oat/tokenizer/oat/encoder/register_encoder.py`) | Embeds the 16 action steps, appends **8 learnable "register" tokens**, runs a transformer (depth 2, dim 256, 4 heads) with a custom mask, and reads out the 8 register outputs projected to `latent_dim=5` → latents `[B, 8, 5]`. |
| Quantizer | `FSQ` (`oat/tokenizer/oat/quantizer/fsq.py`) | Finite Scalar Quantization with `levels=[8,5,5,5,5]`. Bounds each of the 5 channels with `tanh`, rounds to the per-channel grid (straight-through gradient), and folds the digits into **one integer token** per register. |
| Decoder | `SinglePassDecoder` (`oat/tokenizer/oat/decoder/single_pass_decoder.py`) | Reconstructs all 16 steps **in one non-autoregressive pass**: per-timestep sin-cos positional query tokens cross-attend (transformer decoder, depth 4) to the 8 (quantized) latents. |

Forward pass (`OATTok.forward`, `oat/tokenizer/oat/tokenizer.py:63-77`):
`normalize → encoder → FSQ → decoder → MSE(recon, normalized_input)`. The loss is
computed in **normalized** action space; `decode()`/`detokenize()` un-normalize on
the way out.

### 2.2 The FSQ codebook — where "8 tokens" and "5000 vocab" come from

- `latent_dim = len(levels) = 5` — each register is a 5-D vector.
- `codebook_size = prod(levels) = 8·5·5·5·5 = 5000` (`fsq.py:107`).
- So **each of the 8 registers emits one token drawn from a 5000-way vocabulary.**
- FSQ has **no learned codebook** — the code grid is implicit; the token index is a
  fixed mixed-radix encoding (`_basis=[1,8,40,200,1000]`), losslessly invertible
  (`codes_to_indices` / `indices_to_embedding`).
- One action chunk ⇔ **8 tokens**. This is the vocabulary the Stage-2 policy predicts.

### 2.3 Matryoshka / variable-rate tokens

The decoder applies `MaskedNestedDropout` with `token_dropout_mode='pow2'`: during
training it keeps a random power-of-two **prefix** of the 8 latents ({1,2,4,8}) and
masks the rest, so the decoder learns to reconstruct from *any prefix*. Combined
with the encoder's causal-over-registers mask, this imposes a **coarse-to-fine
ordering** on the 8 tokens. Consequences:
- A shorter token prefix already yields a usable chunk → variable-rate tokenization.
- The ordering makes the tokens well-suited to **left-to-right autoregressive**
  generation by the policy, which may stop early (`use_k_tokens < 8`); `detokenize`
  passes the true per-example length as `eval_keep_k` to the decoder.

### 2.4 SO(3) action-chunk augmentation

`SO3ActionChunkAug` (`oat/tokenizer/oat/augment/so3_action_chunk_aug.py`) perturbs
the **rotation** part of each chunk before encoding, to learn rotation-robust codes.

- Action layout assumed: `pos [0:3]`, **`rotvec [3:6]` (axis-angle)**, `gripper [6]`.
- Treats `[3:6]` as axis-angle → `so3_exp_map` to R(t) → left-multiply by **one**
  random rotation `Q` shared across all 16 steps (mode `left_noise`: `R_aug = Q·R`) →
  `so3_log_map` back to rotvecs. The maps are genuine Rodrigues exp/log with
  small-angle Taylor branches, run in fp32.
- Config: `p=0.6` (per-chunk probability), `max_angle_deg=30` (angle ~ Uniform(0,30°)),
  `augment_position=false` (position & gripper untouched).
- It is a **training-only** hook (no-op in eval), and only overrides `forward()` —
  `encode`/`decode`/`tokenize`/`detokenize` are unchanged, so at policy time behavior
  is identical to plain `OATTok`.

> Note: the config file specifies `max_angle_deg=30` (with `10`/`60` commented).
> The specific tokenizer checkpoint used by the shipped policy run was trained with
> `60`. Either way it only affects Stage-1 training, not the inference contract.

### 2.5 Training (`TrainOATTokWorkspace`, `oat/workspace/train_oattok.py`)

- **Loss:** reconstruction MSE only. FSQ needs no commitment/codebook loss (straight-through rounding, no learned codebook).
- **Two MSE quantities logged:** `train_loss`/`val_loss` are MSE in *normalized* space; `test_reconst_mse` is the full `autoencode` round-trip MSE in *raw* action units.
- **Checkpoint selection:** TopK on `test_reconst_mse` (`mode: min`), file pattern `ep-{epoch:04d}_mse-{...}.ckpt`.
- EMA weights are what get saved/evaluated and later loaded by the policy.
- AdamW `lr=5e-5`, batch 256, `num_epochs=5001`, bf16, weight-decay only on ≥2-D params (but `weight_decay=0.0` here).

### 2.6 Config cheat-sheet (`train_oattok_so3aug.yaml`)

| Field | Value | Meaning |
|---|---|---|
| `horizon` | 16 | action chunk length T |
| `action_dim` | 7 | from `task.tokenizer.shape_meta` (pos3 + rot3 + gripper1) |
| `latent_dim` | 5 | `len(quantizer.levels)` |
| `encoder.num_registers` | 8 | tokens per chunk |
| `decoder.latent_horizon` | `${...num_registers}` = 8 | must equal num_registers |
| `decoder.token_dropout_mode` | `pow2` | Matryoshka nested dropout |
| `decoder.use_causal_decoder` | true | causal over the time axis |
| `quantizer.levels` | `[8,5,5,5,5]` | FSQ grid → codebook 5000 |
| dataset | `ZarrDataset`, `libero10_N500.zarr`, `obs_keys: []` | action-only |

---

## 3. Stage 2 — The policy (`train_oatpolicy_with_enriched_past.yaml`)

Class: `OATPolicyWithEnrichedPast` (`oat/policy/oat_policy_with_enriched_past.py`).
An autoregressive transformer (`AutoregressiveModel`,
`oat/model/autoregressive/transformer_cache.py`) that predicts the next chunk's 8
tokens. "Enriched past" is entirely about **what conditions the prediction**.

### 3.1 The "enriched past" conditioning (`_build_condition`, lines 236-272)

The cross-attention **memory** is built from three sources and concatenated:

| Source | Shape | Notes |
|---|---|---|
| Observations (2 frames) | `[B, 2, d]` | `FusedObservationEncoder` output |
| Explicit **acc, jerk** | `[B, 2, d]` | finite diffs of normalized past actions: `acc = a_{t-1}−a_{t-2}`, `jerk = a_{t-1}−2a_{t-2}+a_{t-3}`, each via its own MLP |
| **Raw past 7 actions** | `[B, 7, d]` | the 7 individual past action steps `a_{t-7..t-1}`, shared `raw_proj` MLP |

→ `cond = [obs, acc, jerk, raw]` of length **2 + 2 + 7 = 11** tokens
(`max_cond_len`). Rationale (docstring): observations give position + coarse
velocity; raw past actions give exact velocity/temporal patterns; explicit
acc/jerk give higher-order derivatives the model would otherwise have to learn to
difference out.

> Important: the past is **7 raw continuous action steps** (never tokenized), fed
> as cross-attention memory — *not* 7 past token-chunks. The three enriched-past
> variants differ only in this memory: `OATPolicy` (obs only) ⊂ `OATPolicyWithPast`
> (obs + projected raw past) ⊂ `OATPolicyWithEnrichedPast` (obs + raw past + acc/jerk).

### 3.2 The autoregressive "Past→Next" mechanism

- The **next chunk** = its 8 FSQ register tokens. Decoder input is `[<BOS>, z1..z8]`
  with `<BOS> = codebook_size = 5000`; vocab size = `codebook_size + 1 = 5001`.
- **Causal self-attention** over the 8 tokens (token *i* predicts *i+1* from tokens ≤ *i*);
  **non-causal cross-attention** to the 11-token enriched-past memory (all context
  visible at every step). So each register token is generated left-to-right,
  conditioned on the same chunk's earlier tokens plus the full past.
- Sizes are read off the frozen tokenizer at init: `codebook_size` and
  `latent_horizon (=8)` come from `action_tokenizer` — the "8" the policy predicts is
  provably the same 8 registers the tokenizer produces (not hard-coded).
- The policy predicts **opaque integer ids** in `[0, 4999]`; the FSQ mixed-radix
  structure is invisible to it — the tokenizer handles decoding.

### 3.3 Observation encoders

`FusedObservationEncoder` (`_recursive_=false`) dispatches per modality and
concatenates (`oat/perception/`):
- **Vision** `RobomimicRgbEncoder`: per camera, random-crop 128→**76×76**, ResNet18 →
  SpatialSoftmax (32 keypoints) → Linear to **64-D**. 2 cameras (agentview +
  eye-in-hand), independent weights → `[B, 2, 128]`.
- **State** `ProjectionStateEncoder` (identity): `eef_pos(3) + eef_quat(4) +
  gripper_qpos(2) + task_uid(1) = 10` → `[B, 2, 10]`.
- **Fused** → `obs_feature_dim = 128 + 10 = 138` per frame. The projection 138→256
  (`embed_dim`) happens *inside* the AR transformer (`cond_emb`), not in the encoder.

### 3.4 Inference (`predict_action`, lines 276-348)

1. Encode obs → features `[B,2,138]`.
2. Build `cond` from features + `_past_buffer` (a `[B,7,7]` buffer, **zero-initialized**
   at episode start via `reset()`).
3. Autoregressively `generate` from `<BOS>`, sampling with **temperature=1.0** and
   **top-k=10** (KV-cached); drop `<BOS>`, clamp to `[0,4999]`.
4. `action_tokenizer.detokenize(tokens)` → full **16-step** chunk `action_pred`.
5. **Receding horizon:** execute only the first `n_action_steps=8` steps.
6. Update `_past_buffer` from the *predicted* chunk (the 7 steps ending at step 8),
   so the policy conditions on its own recent predictions.

### 3.5 Training (`TrainPolicyWorkspace`, `oat/workspace/train_policy.py`)

- **Loss:** next-token **cross-entropy** over the 8 chunk tokens (`forward`, lines 352-385).
  Ground-truth actions are tokenized by the **frozen** tokenizer under `no_grad` to
  make targets; teacher-forced logits vs shifted targets.
- **Two learning rates** (`get_optimizer`): `policy_lr=5e-5` for the transformer +
  acc/jerk/raw projections, `obs_enc_lr=1e-5` for the observation encoder.
- **EMA** weights used for all eval and checkpoints.
- **Rollout eval** every `rollout_every=200` epochs runs `LiberoRunner` in sim to get
  `mean_success_rate`; TopK checkpoints on it (`mode: max`, `ep-{...}_sr-{...}.ckpt`).
- `val_loss` (CE) every 10 epochs; `test_reconst_mse` (MSE of predicted vs GT chunk)
  every 10 epochs as a diagnostic.
- Batch 64, bf16, `num_epochs=5001`.

> `lazy_eval: true` in the task config turns the in-training sim rollout **off** by
> default. The referenced `ep-1200_sr-0.824.ckpt` (epoch 1200 = a multiple of
> `rollout_every=200`) means that run overrode `lazy_eval=false` so success-rate
> checkpoint selection could happen.

### 3.6 Config cheat-sheet (`train_oatpolicy_with_enriched_past.yaml`)

| Field | Value | Meaning |
|---|---|---|
| `horizon` | 16 | predicted chunk length |
| `n_action_steps` | 8 | steps actually executed per call (receding horizon) |
| `n_obs_steps` | 2 | observation frames |
| `past_n` | 7 | raw past action steps in the enriched context |
| `policy.embed_dim / n_layers / n_heads` | 256 / 4 / 4 | AR transformer |
| `policy.temperature / topk` | 1.0 / 10 | sampling at inference |
| `action_tokenizer` | `OATTok.from_checkpoint(???)` | frozen Stage-1 tokenizer (path required) |
| dataset | `ZarrDatasetWithPastAction`, `past_n` | adds the `past_action` stream |

---

## 4. The tokenizer ↔ policy linkage (the critical handoff)

This is the load-bearing contract; all four points verified against source:

1. **8 tokens, vocab 5001.** Encoder produces 8 registers; FSQ → 8 token ids over a
   5000 codebook; policy builds its head with `vocab = codebook_size+1` and emits
   exactly `latent_horizon = 8` tokens. `num_registers` (encoder) and
   `latent_horizon` (decoder) **must stay equal** — enforced only by the config
   interpolation `latent_horizon: ${tokenizer.encoder.num_registers}`.
2. **Frozen & shared.** The `action_tokenizer` params are `requires_grad_(False)` +
   `.eval()` and excluded from the optimizer. The **same instance** makes training
   targets (`tokenize`) and decodes at inference (`detokenize`) — same weights, same
   embedded `LinearNormalizer`, so train-time targets and rollout decoding use one
   consistent vocabulary. Loaded once via `OATTok.from_checkpoint` (returns the EMA
   weights since `use_ema=True`).
3. **Predict 16, act 8.** The tokenizer decodes 8 tokens into a full 16-step chunk;
   only the first `n_action_steps=8` are executed (receding horizon). The dataset
   supplies 16-step target chunks.
4. **Normalizer travels in the checkpoint**, so past-action normalization at policy
   level matches the tokenizer's.

---

## 5. Key numbers at a glance

| Quantity | Value |
|---|---|
| Action dim | 7 = pos(3) + axis-angle rot(3) + gripper(1) |
| Chunk length (horizon) | 16 steps |
| Tokens per chunk | 8 (one per register) |
| FSQ levels / latent dim | `[8,5,5,5,5]` / 5 |
| Codebook / policy vocab | 5000 / 5001 (+1 `<BOS>`=5000) |
| Obs frames / past actions | 2 / 7 |
| Condition (memory) length | 11 = 2 obs + 2 (acc,jerk) + 7 past |
| Obs feature dim | 138 = 2 cameras×64 + 10 state |
| Executed steps per call | 8 of 16 (receding horizon) |
| Tokenizer transformer | enc depth 2 / dec depth 4, dim 256, 4 heads |
| Policy transformer | 4 layers, dim 256, 4 heads |
| Sampling | temperature 1.0, top-k 10 |

---

## 6. How to train & run

```bash
# Stage 1 — train the action tokenizer (produces ep-XXXX_mse-*.ckpt)
python oat/workspace/train_oattok.py --config-name train_oattok_so3aug

# Stage 2 — train the policy, pointing at the Stage-1 checkpoint
python oat/workspace/train_policy.py --config-name train_oatpolicy_with_enriched_past \
    policy.action_tokenizer.checkpoint=/path/to/tokenizer/ep-XXXX_mse-0.001.ckpt

# Evaluate a trained policy in LIBERO-10 sim (success rate; smoothness variant available)
python scripts/eval_policy_sim.py -c /path/to/policy.ckpt -o out_dir
```

---

## 7. Subtleties & gotchas (verified)

- **Train/inference past mismatch.** The dataset **edge-pads** the initial past
  (repeats the first frame) at episode starts, but at rollout the policy's
  `_past_buffer` is **zero-initialized** — so early-episode acc/jerk features have a
  different distribution between training and deployment.
- **Past = own predictions at rollout.** The past buffer is filled from the model's
  predicted actions, not environment-observed ones, so errors can compound.
- **acc/jerk are in normalized action space** (finite diffs of normalized past
  actions), not physical units.
- **BOS is weight-tied** into the output head; a spuriously sampled BOS is clamped
  out at inference and is never a training target.
- **SO(3) aug is training-only and rotation-only**; position and gripper are never
  perturbed, and the same `Q` is shared across all 16 steps of a chunk.
- **Reconstruction MSE is measured two ways** in Stage 1 (normalized-space loss vs
  raw-space `test_reconst_mse`); checkpoint selection uses the raw-space one.
- **Config invariants not asserted in code:** `num_registers == latent_horizon`, and
  `latent_dim == len(FSQ levels)` — both held together by config interpolation, not
  runtime checks.
</content>
