# OAT-VLA: Integrating PaliGemma with OAT as a Vision-Language-Action Model

## Context

The codebase at [oat/](/workspace/oat) is a clean, modular research stack for autoregressive (AR) robot policy learning. Its novel contribution is an **action tokenizer** (`OATTok`, [oat/tokenizer/oat/tokenizer.py:28](/workspace/oat/oat/tokenizer/oat/tokenizer.py#L28)) that maps continuous actions to discrete tokens with good *ordinal* structure (see [ordered_lattice_quantization.md](/workspace/oat/ordered_lattice_quantization.md)) — so that a small AR transformer ([oat/model/autoregressive/transformer_cache.py:156](/workspace/oat/oat/model/autoregressive/transformer_cache.py#L156)) can predict actions cleanly. Today, the policy ([oat/policy/oatpolicy.py:12](/workspace/oat/oat/policy/oatpolicy.py#L12)) conditions on RGB + proprioceptive state encoded by [FusedObservationEncoder](/workspace/oat/oat/perception/fused_obs_encoder.py); language *exists in the dataset* (LIBERO `task.language` is stored as `'prompt'`) but the policy never reads it.

The goal of this change is to turn OAT into a **VLA**: replace the current vision-only stack with a frozen VLM (PaliGemma-3B) that grounds (image, instruction) into rich features, and feed those features into the existing OAT AR transformer via cross-attention. This is structurally a **π0-style** design — VLM provides perception/language grounding; OAT's AR transformer with discrete action tokens is the action expert (analogous to π0's flow-matching expert, but tokenized). It preserves OAT's dynamic-feature pathways (state encoder, past-action variants, kinematics) and keeps the OAT tokenizer as the headline contribution.

**Decisions locked in (from clarifying Qs):**
- VLM: **PaliGemma-3B** (`google/paligemma-3b-pt-224`).
- Architecture: **VLM as conditioner** + existing OAT AR head (π0-style, not OpenVLA-style backbone).
- Training scope (v1): **Frozen VLM**, train only OAT AR transformer + a thin projection adapter.
- Frame history: **encode each of `To` frames independently** with PaliGemma, stack along sequence axis.
- Benchmark: **LIBERO** (already wired up).

## Architecture

```
                         ┌─ instruction text (LIBERO `prompt`)
                         │
RGB frames [B,To,H,W,3] ─┤
                         │
                         ▼
              ┌─────────────────────┐
              │ Frozen PaliGemma-3B │   one forward per (frame, instruction)
              │  SigLIP + Gemma     │
              └────────┬────────────┘
                       │  hidden states at all positions
                       ▼
              [B, To*N_vlm, 2048]  (N_vlm = #image tokens + #text tokens per frame)
                       │
                       │  Linear adapter (2048 → cond_dim, e.g. 512)
                       ▼
              [B, To*N_vlm, cond_dim]
                       │                    ┌───────────────────────────┐
                       ├────────────────────►                           │
                                            │                           │
state [B,To,Ds] ─► state MLP ─► [B,To,cd]──►│  cross-attn memory        │
                                            │                           │
(optional: past-action tokens)             │                           │
                                            │  OAT AR transformer       │  predicts
                                            │  (existing, unchanged)    │──► action tokens
                                            │                           │      [B, latent_horizon]
                                            └───────────────────────────┘            │
                                                                                     ▼
                                                                          OAT detokenize → continuous actions
```

The OAT AR transformer's cross-attention treats the entire sequence above (vision-language tokens + state tokens) as memory. Self-attention runs over action tokens. **Nothing inside the AR transformer or the OAT tokenizer changes.**

## File-Level Changes

### New files

1. **`oat/perception/paligemma_obs_encoder.py`** — new `PaliGemmaObsEncoder(BaseObservationEncoder)`.
   - `__init__`: load `PaliGemmaForConditionalGeneration` and `PaliGemmaProcessor` from HF, set `requires_grad=False`, `eval()`. Add `nn.Linear(vlm_hidden_dim, out_dim)` adapter (trainable). Read `image_ports`, `prompt_port` (default `'prompt'`), `state_ports` from `shape_meta`.
   - `forward(obs_dict)`: for each (frame, camera) run PaliGemma with the per-batch instruction string; collect last-layer hidden states at *image positions only* (drop BOS / instruction tokens, or keep them — make it a flag). Reshape to `[B, To*n_cameras*n_img_tokens, vlm_hidden_dim]`, project through adapter to `[B, T_cond, out_dim]`. Run state encoder separately and concatenate **along the sequence axis** to produce final memory `[B, T_cond_total, out_dim]`.
   - `output_feature_dim()` returns `out_dim` (e.g. 512). `modalities()` returns `['rgb','text','state']`.
   - Key efficiency: tokenize the instruction *once per batch* (cache in dataloader); flatten `(B, To, n_cameras)` into a single PaliGemma forward of size `B*To*n_cameras` images sharing the prompt; wrap in `torch.no_grad()` + `bf16`.

2. **`oat/config/task/policy/libero/libero10_vla.yaml`** — copy of `libero10.yaml` with:
   - `obs.prompt: {shape: [1], type: text}` added.
   - Removes `task_uid` from state ports (the language prompt now carries task identity).

3. **`oat/config/train_oatpolicy_vla.yaml`** — copy of `train_oatpolicy.yaml` with:
   - `obs_encoder._target_` → `oat.perception.paligemma_obs_encoder.PaliGemmaObsEncoder`.
   - `obs_encoder.vlm_name: google/paligemma-3b-pt-224`, `out_dim: 512`, `precision: bf16`, `use_image_tokens_only: true`.
   - `policy.embed_dim: 512` (matches adapter `out_dim`, so AR transformer's `cond_dim=512`).
   - `policy.obs_enc_lr: 0` (VLM frozen) — but adapter is registered under `obs_encoder` and trains at this lr; expose a separate `adapter_lr` if necessary (see optimizer change below).
   - Smaller batch size (e.g. 16) and grad-accum to fit PaliGemma in memory.

### Edits to existing files

4. **`oat/policy/oatpolicy.py`** — minor:
   - In [`__init__`](/workspace/oat/oat/policy/oatpolicy.py#L13), the AR model is built with `max_cond_len=n_obs_steps` ([line 56](/workspace/oat/oat/policy/oatpolicy.py#L56)). With VLM features, the cross-attention sequence is much longer (`To * N_vlm + state_tokens`). Change to `max_cond_len = obs_encoder.max_cond_seq_len()` (new method on the encoder; falls back to `n_obs_steps` for the existing FusedObservationEncoder).
   - In [`get_optimizer`](/workspace/oat/oat/policy/oatpolicy.py#L129), already separates `policy_lr` vs `obs_enc_lr`. The frozen VLM has `requires_grad=False` so it's auto-skipped; only the adapter and state MLP train at `obs_enc_lr`. No code change required, just config.

5. **`oat/perception/base_obs_encoder.py`** — add an optional `max_cond_seq_len(self, n_obs_steps: int) -> int` method (default returns `n_obs_steps`). Implement it in both `FusedObservationEncoder` (returns `n_obs_steps`) and the new `PaliGemmaObsEncoder` (returns `n_obs_steps * n_cameras * n_vlm_tokens_per_frame + n_state_tokens`). This is the only change touching the BaseObservationEncoder contract.

6. **`oat/dataset/zarr_dataset.py`** — already returns text via `text_obs_keys` ([dataset_conversion.py:32](/workspace/oat/oat/dataset/dataset_conversion.py#L32)); no schema change. Add a small collate-time hook to pre-tokenize the instruction with the PaliGemma processor so the GPU forward doesn't pay tokenization cost (optional optimization, not required for correctness).

### Files **not** changed

- `oat/tokenizer/oat/**` — frozen, reused as-is.
- `oat/model/autoregressive/transformer_cache.py` — reused as-is. Cross-attention already supports arbitrary `cond_dim` and `max_cond_len`.
- `oat/policy/oat_policy_with_past.py`, `_with_kinematics.py`, `_with_round_past.py` — work unchanged because they consume the same `obs_encoder` interface; once the new encoder is dropped in, all variants inherit VLA capability.
- `oat/workspace/train_policy.py` — runs unchanged via Hydra config swap.

## Reused Components

- [`OATTok` tokenize/detokenize](/workspace/oat/oat/tokenizer/oat/tokenizer.py#L116-L143) — used as-is for action ↔ token mapping.
- [`AutoregressiveModel` with KV cache](/workspace/oat/oat/model/autoregressive/transformer_cache.py#L156) — reused; its cross-attention path consumes the new `[B, T_cond, cond_dim]` memory unchanged. The KV-cache for cross-attention is precomputed once per episode (already supported).
- [`ProjectionStateEncoder`](/workspace/oat/oat/perception/state_encoder.py) — reused inside `PaliGemmaObsEncoder` for proprioception. Output projected to `out_dim` and appended to memory as additional sequence tokens.
- [`LinearNormalizer`](/workspace/oat/oat/model/common/normalizer.py) — reused for action and state normalization.
- HuggingFace `transformers.PaliGemmaForConditionalGeneration` and `PaliGemmaProcessor` — pulled in as dependencies (`pyproject.toml` add `transformers>=4.45`, `pillow`).

## Training Plan

1. Reuse the frozen OAT tokenizer checkpoint already in the repo (`ep-0470_mse-0.002.ckpt` / `tok_ckpt/`). No retraining of the tokenizer.
2. Train OAT AR transformer + adapter + state MLP on LIBERO-10 with the new config. Keep PaliGemma in `bf16` `eval()` and wrap its forward in `torch.no_grad()` to make memory tractable on a single A100/H100.
3. Sanity-check: with the prompt port present but the VLM disabled (e.g. an ablation that replaces PaliGemma with a small projector over zeros), the run should match the original `train_oatpolicy.yaml` numbers. This isolates the VLM contribution.
4. Headline run: PaliGemma frozen → measure LIBERO-10 success rate vs the existing baseline (e.g. `ep-0800_sr-0.656.ckpt`). The expected gain comes from language-conditioned task disambiguation (no more reliance on `task_uid` one-hots).

## Verification

End-to-end checks, in order:

1. **Unit shape test**: instantiate `PaliGemmaObsEncoder` from the new config and run a synthetic batch (`B=2, To=2, two 224×224 RGB cameras, prompt strings`). Assert output shape is `[2, T_cond_expected, 512]` and that VLM params have `requires_grad=False` while the adapter is trainable.
2. **Policy forward**: build `OATPolicy` with the new encoder; run `policy.forward(batch)` on one LIBERO sample → expect a finite scalar loss. Run `policy.predict_action(obs_dict)` → expect a `[B, n_action_steps, 7]` tensor.
3. **Overfit test**: train on a single LIBERO trajectory for ~500 steps; loss should drop close to zero, confirming the adapter + AR transformer can use VLM features.
4. **Short training run**: 5k steps on LIBERO-10; eval with the LIBERO env runner ([oat/env_runner/](/workspace/oat/oat/env_runner/)) on at least one task; success rate should be non-trivial (>~10%) — confirms the language path is wired.
5. **Full training run**: same hyperparams as the existing OAT-LIBERO baseline; compare LIBERO-10 mean success rate against the no-language baseline. Also run an ablation that swaps the prompt for an empty string to verify language is actually being used.

## Open Questions / Future Work (out of scope for v1)

- Replace the frozen VLM with **LoRA fine-tuning** once v1 is stable — drop-in change to `PaliGemmaObsEncoder.__init__` (apply `peft.get_peft_model`).
- A v2 OpenVLA-style refactor (PaliGemma decodes OAT tokens directly) — would require extending PaliGemma's vocab with `codebook_size+1` action tokens and a custom training loop; deferred.
- Larger datasets (Bridge / DROID / OXE) — would need new dataset converters under `oat/dataset/`; deferred.
