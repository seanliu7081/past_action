# Plan: OAT-VLA — PaliGemma2 decodes OAT action tokens

## Context

`/workspace/oat` has two trained components to fuse into a VLA:

1. **OAT action tokenizer** ([train_oattok_so3aug.yaml](../oat/config/train_oattok_so3aug.yaml)): FSQ tokenizer — `(B,16,7)` action chunk → **8 discrete tokens**, vocab **5000**, prefix-decodable (k∈{1,2,4,8}). Frozen ckpt: `output/20260514/085735_train_oattok_so3aug_libero10_N500/checkpoints/ep-1750_mse-0.002.ckpt` (loads via `OATTok.from_checkpoint`).
2. **Enriched-past AR policy** ([train_oatpolicy_with_enriched_past.yaml](../oat/config/train_oatpolicy_with_enriched_past.yaml)): conditions on 2 obs frames + 7 raw past actions (acc/jerk/raw MLP features); small AR head predicts the 8 tokens; executes 8 of 16 steps; rolling `_past_buffer`. Baseline SR 0.824.

**Decisions:** (a) **VLM decodes action tokens** (OpenVLA/π0-FAST style) — PaliGemma2 replaces the small AR head and autoregressively emits the 8 OAT ids; supersedes the conditioner-style [integrate_vla.md](../integrate_vla.md). (b) **Frozen backbone + LoRA flag** (peft optional). (c) User provides **HF_TOKEN** for the gated `google/paligemma2-3b-pt-224`.

**Constraint:** prefer new files over editing shared ones → **zero edits to existing files**.

## Architecture

```
prefix (bidirectional attn), P = 1068:
  [4×256 img tokens: agentview t-1,t + eye_in_hand t-1,t (128→224px, SigLIP+projector, ÷√D)]
  [BOS + instruction + "\n", padded to 32 (pad cols attn-masked); instruction = task_uid→language lookup]
  [2 state tokens (eef pos/quat/gripper, 9-d → MLP)] [acc] [jerk] [7 raw past-action tokens]  (all soft, ÷√D)
  [BOA]                        ← own nn.Embedding(5001, D), id 5000 = BOA
suffix (causal), 8 positions:
  [a0 … a7]                    ← teacher-forced at train; sampled at inference
head: nn.Linear(D, 5000, bias=False) on last_hidden_state → CE over 8 positions
  → clamp(0,4999) → OATTok.detokenize → (B,16,7) → execute first 8; _past_buffer = pred[:,1:8]
```

- D = `config.text_config.hidden_size` (Gemma2-2B: 2304); read at runtime, never hardcoded.
- **No HF vocab resize**: separate action embedding + untied head keep the backbone pristine.
- Enriched-past scheme ported verbatim from [oat_policy_with_enriched_past.py](../oat/policy/oat_policy_with_enriched_past.py) (L253-272): acc = a₋₁−a₋₂, jerk = a₋₁−2a₋₂+a₋₃ on normalized past actions; MLPs `Linear(in→D)→GELU→Linear(D→D)`.

### Verified API mechanics (transformers 4.57.6, installed source — spot-checked)
- Assemble `inputs_embeds` manually; skip processor/placeholder path (`PaliGemmaModel.forward` supports `inputs_embeds` with `pixel_values=None`).
- Image features: replicate `get_image_features` — vision tower (under `no_grad`) → `multi_modal_projector` → **÷√D** (modeling_paligemma.py:244). Preprocess in-policy: `x/255`, normalize mean=std=0.5, bicubic resize to 224.
- **Scale gotcha**: Gemma2 multiplies all `inputs_embeds` by √D → all soft-token MLP outputs ÷√D; action embedding init `normal_(0, 0.02)`; verify RMS vs text embedding table at first real load.
- **Masking**: build the 4D mask via `self.vlm.model._update_causal_mask(..., is_training=True)` (prefix bidirectional via `token_type_ids==0` unmask, suffix causal, padding applied after — verified lines 191-227); 4D masks pass through untouched. Sliding window (4096) never binds at L≈1076. Private API — pinned to 4.57.6, guarded by a parity smoke test.
- **Generation**: prefill prefix with `DynamicCache`, then 8 cached single-token steps; sampling mirrors `transformer_cache.py:301-312` (T=0→argmax; else logits/T, top-k, multinomial).
- **Gradients**: trunk must run WITH grad (soft tokens/embeddings need backprop through it); only SigLIP under `no_grad`. Backbone loaded bf16 (frozen ⇒ no master copies); new modules fp32; head applied to `.float()` hidden states. **Gradient checkpointing required** (non-reentrant; needs LM in train mode — safe, Gemma2 dropout=0). `attn_implementation: eager` (sdpa drops Gemma2's attn softcap); sdpa is a speed lever.
- `policy.dtype` resolves to the fp32 `_dummy_variable` → runner casts rollout obs to fp32, not bf16 (verified `ModuleAttrMixin`).

## Prompt resolution (zarr `<U44` truncation fix)

Never read the zarr `prompt` array (dtype `<U44` truncates long LIBERO-10 instructions). Build `task_uid → task.language` from the same enumeration the env uses ([env.py](../oat/env/libero/env.py) global task list + `libero.libero.benchmark`) — consistent train/rollout by construction; the policy keys on `task_uid` (already in obs at train and rollout) and never consumes the string port.

## New files (no edits to existing files)

1. **`oat/model/vla/paligemma_action_decoder.py`** — `PaliGemmaActionDecoder(nn.Module)`: loads `PaliGemmaForConditionalGeneration` (bf16, eager, `requires_grad_(False)`, grad-ckpt); optional `train_mm_projector` and `use_lora` (peft `inject_adapter_in_model` on q/k/v/o, LoRA params cast fp32; ImportError with install hint if peft absent); `tiny_debug: true` builds a small random `PaliGemmaConfig` (tests without HF token). Methods: `set_instruction_table(uid→text)` (tokenize `BOS+text+"\n"`, `add_special_tokens=False`, right-pad to 32), `encode_prefix(images, state, past_norm, uid) → (prefix_embeds, attn2d)`, `forward_train(...) → CE loss`, `generate(..., k, temperature, topk) → (B,k)`. Plus `oat/model/vla/__init__.py`.
2. **`oat/env/libero/prompt_table.py`** — `get_task_uid_prompt_table()` per above (function-local libero imports).
3. **`oat/policy/oat_vla_policy.py`** — `OATVLAPolicy(BasePolicy)`: same external contract as `OATPolicyWithEnrichedPast` (`forward(batch)→loss`; `predict_action` with identical `_past_buffer`/`reset()`/return-dict/clamp semantics — keep the zero-init-vs-repeat-padding quirk for parity; `set_normalizer`; `get_observation_ports()` from shape_meta — `task_uid` used only for prompt lookup, not as a state feature; **`get_optimizer(policy_lr, obs_enc_lr, weight_decay, betas)` exact signature** so `TrainPolicyWorkspace` runs unchanged). Tokenizer frozen + **`train()` override forcing `action_tokenizer.eval()`** (dropout would make `tokenize()` non-deterministic) while leaving the LM in train mode (grad-ckpt gate needs it). `state_dict()`/`load_state_dict()` drop frozen `vlm.*` keys unless `save_backbone_in_checkpoint: true` (~200 MB vs ~6.2 GB ckpts; filtered ckpts need the HF cache at load — verified compatible with `BaseWorkspace.save_checkpoint`/`load_payload`).
4. **`oat/config/train_oatvla_paligemma.yaml`** — reuses task fragment `task/policy: libero/libero10_with_past` as-is. Key settings: tokenizer ckpt path baked in; `use_ema: False` (workspace would deep-copy the 3B backbone; verified `use_ema: False` → `ema_model = None`); `batch_size: 8`, `gradient_accumulate_every: 4` (effective 64 on 2 GPUs); `num_epochs: 15`, cosine, warmup 1000; `policy_lr: 1e-4` (fresh embeds/head/MLPs), `obs_enc_lr: 5e-5` (LoRA/projector group); `val_every/sample_every/checkpoint_every: 1`, `max_val_steps: 50`, `max_reconst_steps: 10`; **topk monitor `test_reconst_mse` (min)** — the inherited `mean_success_rate` monitor never fires under `lazy_eval: true`.
5. **`scripts/smoke_test_oatvla.py`** — token-free via `tiny_debug`: shapes; **train/decode logits parity** (teacher-forced `forward_train` vs step-by-step `generate` path — validates mask+cache+positions); tokenizer-eval invariant; state-dict filter round-trip; every trainable param in exactly one optimizer group; prompt-table vs truncated-zarr consistency.

Optimizer groups: `policy_lr` → action_embed, action_head, state/acc/jerk/raw MLPs (~44M, decay split by dim≥2); `obs_enc_lr` → trainable `vlm.*` (LoRA/projector; omitted when empty).

## Memory / throughput (2× RTX 4090 24 GB)

Per GPU at micro-batch 8, L=1076, eager, grad-ckpt: ~6.1 GB weights + ~0.7 GB trainables/Adam + ~1.7 GB activations/recompute + ~1.5 GB SigLIP/misc ≈ **~10 GB** — comfortable; try b=12-16 later. ~2-2.5 s/micro-step → ~5-6 h/epoch → **15 epochs ≈ 3.5 days**. Cost levers in order: `n_image_obs_steps: 1` (~2×), `attn_implementation: sdpa` (~20-30%), larger micro-batch. LoRA run only after the frozen baseline trains.

## Prerequisites

1. `uv pip install sentencepiece` (+ `peft` only when `use_lora`).
2. Accept Gemma license, set `HF_TOKEN`; download `google/paligemma2-3b-pt-224` (~6 GB) into `HF_HOME=/workspace/.hf_home`.

## Verification (staged)

1. **No token needed**: `scripts/smoke_test_oatvla.py` (tiny_debug); 200-step single-batch overfit with tiny model (CE→~0 proves gradient flow through frozen trunk); one real `ZarrDatasetWithPastAction` batch through `forward` + `predict_action`.
2. **With token**: real-backbone load + embedding-RMS check + one fwd/bwd at b=8 (record peak mem); single-batch overfit CE→~0; short train (`max_train_steps: 100`, 2-3 epochs; start loss ≈ ln 5000 ≈ 8.5, watch val CE + `test_reconst_mse`).
3. **Full train**: `HF_HOME=/workspace/.hf_home accelerate launch --num_processes 2 scripts/run_workspace.py --config-name=train_oatvla_paligemma`.
4. **Offline eval**: `scripts/eval_policy_sim.py -c <ckpt> -o <dir>` → LIBERO-10 `mean_success_rate` vs 0.824 baseline (~1 h for n_test=500 at B=20 prefill ≈1.4 s + 7 cached decodes per chunk); plus `--temperature 0` variant and an empty-instruction ablation to confirm language is used.

## Risks

- Hub config unverified until token arrives (dims read from loaded config, so degrades to nothing).
- `_update_causal_mask` is private API — pinned 4.57.6; parity smoke test catches drift.
- Throughput estimates ±40%; `n_image_obs_steps: 1` is the sanctioned fallback.
- Inherited train/rollout past-buffer mismatch kept deliberately for baseline parity.