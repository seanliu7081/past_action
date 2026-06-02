# Comparison — Ours (SO(3)-aug Tokenizer + Enriched-Past Policy) vs. Baseline (OATTok + OATPolicy)

This document compares the two methods head-to-head. The standalone deep-dives live at:

- [method_ours_so3aug_enriched_past.md](method_ours_so3aug_enriched_past.md)
- [method_baseline_oattok_oatpolicy.md](method_baseline_oattok_oatpolicy.md)

The "ours" method is **strictly additive**: it inherits the entire baseline architecture and adds (a) an SO(3) data augmentation on raw actions before normalization in Stage 1, and (b) an enriched conditioning sequence (raw past actions + acceleration + jerk) in Stage 2.

---

## 1. Side-by-side summary

| Axis | Baseline | Ours |
|---|---|---|
| **Stage-1 entry config** | [oat/config/train_oattok.yaml](oat/oat/config/train_oattok.yaml) | [oat/config/train_oattok_so3aug.yaml](oat/oat/config/train_oattok_so3aug.yaml) |
| **Tokenizer class** | `OATTok` ([oat/tokenizer/oat/tokenizer.py](oat/oat/tokenizer/oat/tokenizer.py)) | `OATTokSO3Aug` ([oat/tokenizer/oat/tokenizer_so3_aug.py](oat/oat/tokenizer/oat/tokenizer_so3_aug.py)) — subclass of `OATTok` |
| **Action augmentation** | none | `SO3ActionChunkAug` ([oat/tokenizer/oat/augment/so3_action_chunk_aug.py](oat/oat/tokenizer/oat/augment/so3_action_chunk_aug.py)) with `p=0.6, max_angle_deg=60°, mode=left_noise, augment_position=false, rot_slice=[3:6]` |
| **Encoder / Decoder / Quantizer** | RegisterEncoder + SinglePassDecoder + FSQ — *unchanged* | *Identical* (encoder/decoder/quantizer yaml blocks are byte-for-byte equivalent) |
| **FSQ levels / codebook size** | [8, 5, 5, 5] / 1000 | [8, 5, 5, 5] / 1000 |
| **Stage-2 entry config** | [oat/config/train_oatpolicy.yaml](oat/oat/config/train_oatpolicy.yaml) | [oat/config/train_oatpolicy_with_enriched_past.yaml](oat/oat/config/train_oatpolicy_with_enriched_past.yaml) |
| **Policy class** | `OATPolicy` ([oat/policy/oatpolicy.py](oat/oat/policy/oatpolicy.py)) | `OATPolicyWithEnrichedPast` ([oat/policy/oat_policy_with_enriched_past.py](oat/oat/policy/oat_policy_with_enriched_past.py)) |
| **Condition length (`max_cond_len`)** | `n_obs_steps = 2` | `n_obs_steps + 2 + past_n = 2 + 2 + 7 = 11` |
| **Condition layout** | `[obs₁, obs₂]` | `[obs₁, obs₂, acc, jerk, a_{t-7}, ..., a_{t-1}]` |
| **Added trainable modules** | — | `acc_proj`, `jerk_proj`, `raw_proj` (each `Linear(7, d) → GELU → Linear(d, d)`) — added to the **policy LR** optimizer group |
| **Task config (policy)** | `task/policy/libero/libero10.yaml` (ZarrDataset) | `task/policy/libero/libero10_with_past.yaml` (ZarrDatasetWithPastAction) |
| **Dataset class (policy)** | `ZarrDataset` ([oat/dataset/zarr_dataset.py](oat/oat/dataset/zarr_dataset.py)) | `ZarrDatasetWithPastAction` ([oat/dataset/zarr_dataset_with_past.py](oat/oat/dataset/zarr_dataset_with_past.py)) |
| **Extra sample field** | — | `past_action: (B, 7, 7)` |
| **Sliding-window `pad_before`** | `max(To−1, 0) = 1` | `max(To−1, 0) + past_n = 8` |
| **Inference state** | stateless (`reset()` is no-op) | `_past_buffer: (B, 7, 7)`, lazily inited and updated each call |
| **`reset()` semantics** | inherited no-op | `self._past_buffer = None` (re-zero-init on next predict) |
| **AR backbone / Obs encoder** | unchanged | unchanged |
| **Stage-2 optimizer groups** | (policy decay/no-decay, obs-enc decay/no-decay) | identical structure, but the policy groups now include `acc_proj`, `jerk_proj`, `raw_proj` parameters in addition to `self.model` |
| **Checkpoint monitor (Stage 1)** | `test_reconst_mse` (min, top-3) | same |
| **Checkpoint monitor (Stage 2)** | `mean_success_rate` (max, top-3) | same |
| **Training schedule (epochs / batch / lr)** | 5001 / 256 (tok), 64 (pol) / 5e-5 + 1e-5 | identical |

What is *intentionally* unchanged: encoder/decoder/quantizer architecture and hyperparameters, FSQ levels, observation encoder (vision + state), AR transformer (embed_dim / depth / heads), action/horizon counts, optimizer/EMA/checkpoint cadence, normalizer fitting strategy, top-k temperature sampling, rollout cadence.

---

## 2. Tokenizer differences — exactly what changed

The diff between `OATTok` and `OATTokSO3Aug` is the smallest possible:

```diff
  class OATTokSO3Aug(OATTok):
+     def __init__(self, encoder, decoder, quantizer, action_aug=None):
+         super().__init__(encoder=encoder, decoder=decoder, quantizer=quantizer)
+         self.action_aug = action_aug
+
      def forward(self, batch):
          samples = batch["action"]
+         if self.action_aug is not None:
+             samples = self.action_aug(samples)
          nsamples = self.normalizer["action"].normalize(samples)
          latents = self.encoder(nsamples)
          latents, _ = self.quantizer(latents)
          recons = self.decoder(latents)
          return F.mse_loss(recons, nsamples)
```

Everything else — `encode`, `decode`, `tokenize`, `detokenize`, `from_checkpoint`, the optimizer factory, the normalizer plumbing, and the whole inference path — is inherited verbatim.

The augmentation itself (`SO3ActionChunkAug.forward`):

```
1. Per-batch-element mask m ~ Bernoulli(p=0.6).
2. Sample one Q ∈ SO(3) per batch element (axis ~ Uniform(S²), angle ~ Uniform(0, 60°)).
3. For each timestep t ∈ [0..15]:
       R(t) = expmap(actions[t, 3:6])
       R_aug(t) = Q · R(t)                          # 'left_noise' mode
       actions[t, 3:6] = logmap(R_aug(t))           # only where m[b] = 1
4. Position slice [0:3] is *not* rotated (augment_position=false).
5. Gripper slice [6:7] is untouched.
```

The same `Q` is applied across all 16 timesteps of one chunk, preserving the chunk's *relative* rotational motion while shifting its *absolute* orientation by `Q`.

**Why a subclass and not a flag.** Hydra-instantiates each tokenizer class fresh from the yaml, and `BaseTokenizer.from_checkpoint` reads `cfg._target_` to decide what class to re-instantiate. By making `OATTokSO3Aug` a separate `_target_`, checkpoints from the two methods are unambiguously distinguishable and `from_checkpoint` "just works" without needing an extra `enable_aug` flag persisted somewhere.

---

## 3. Policy differences — what's added in ours

### 3.1 New trainable modules

```python
# In OATPolicyWithEnrichedPast.__init__:
acc_proj  = nn.Sequential(nn.Linear(action_dim, obs_feature_dim),
                          nn.GELU(),
                          nn.Linear(obs_feature_dim, obs_feature_dim))
jerk_proj = nn.Sequential(nn.Linear(action_dim, obs_feature_dim),
                          nn.GELU(),
                          nn.Linear(obs_feature_dim, obs_feature_dim))
raw_proj  = nn.Sequential(nn.Linear(action_dim, obs_feature_dim),
                          nn.GELU(),
                          nn.Linear(obs_feature_dim, obs_feature_dim))
action_normalizer = LinearNormalizer()    # for normalizing past_action at runtime
```

Three independent `Linear → GELU → Linear` blocks (same output dim as the obs encoder) plus a dedicated `LinearNormalizer` that loads the same state as the obs encoder's action normalizer (so `_build_condition` can normalize raw past actions consistently with what the tokenizer expects).

### 3.2 Condition length grows from 2 to 11

The AR backbone's `max_cond_len` (`AutoregressiveModel(... max_cond_len=...)`) is sized differently:

| | Baseline | Ours |
|---|---|---|
| `max_cond_len` | `n_obs_steps = 2` | `n_obs_steps + 2 + past_n = 11` |
| `cond_pos_emb` table size | 2 × `n_emb` | 11 × `n_emb` |

This is the only architectural difference inside the AR backbone — all per-layer modules are identical (same depth, width, heads, dropout). The cost is a slightly larger condition-positional-embedding table and longer attention-key sequences in every cross-attention layer.

### 3.3 `_build_condition` — how the new tokens are assembled

```python
def _build_condition(self, obs_features, past_actions):
    # obs_features: (B, To=2, d);  past_actions: (B, past_n=7, action_dim=7)
    norm_past = self.action_normalizer["action"].normalize(past_actions)

    a_t1, a_t2, a_t3 = norm_past[:, -1], norm_past[:, -2], norm_past[:, -3]
    acc  = a_t1 -        a_t2                          # (B, 7)
    jerk = a_t1 - 2.0 *  a_t2 + a_t3                  # (B, 7)

    acc_feat  = self.acc_proj(acc)                     # (B, d)
    jerk_feat = self.jerk_proj(jerk)                   # (B, d)
    raw_feat  = self.raw_proj(norm_past)               # (B, 7, d)  (shared MLP over time)

    explicit  = torch.stack([acc_feat, jerk_feat], dim=1)         # (B, 2, d)
    return torch.cat([obs_features, explicit, raw_feat], dim=1)   # (B, 11, d)
```

Points worth noting:

- Differences are taken in **normalized space**. This matches the tokenizer (which also operates on normalized actions) and keeps magnitudes consistent across the three projection heads.
- `raw_proj` is **shared across all 7 past timesteps** (applied via batched matmul on the last dim), saving parameters and providing translation invariance over the past axis.
- `acc` and `jerk` use the **last 3** past steps only (`a_{t-3}, a_{t-2}, a_{t-1}`). The other 4 past steps (`a_{t-7}, ..., a_{t-4}`) participate only via the `raw_proj` path.
- The concat order is `[obs, explicit, raw]` and is fixed — the AR backbone's `cond_pos_emb` table assigns a unique learned positional embedding to each of the 11 slots, so the model learns the role of each slot.

### 3.4 Training-time signal

```diff
  def forward(self, batch):
      with torch.no_grad():
          action_tokens = self.action_tokenizer.tokenize(batch["action"])
-     features = self.obs_encoder(batch["obs"])
-     cond = features
+     features = self.obs_encoder(batch["obs"])
+     past_actions = batch["past_action"]                # (B, 7, 7), from ZarrDatasetWithPastAction
+     cond = self._build_condition(features, past_actions)
      action_tokens = torch.cat([BOS, action_tokens], dim=1)
      logits = self.model(action_tokens[:, :-1], cond=cond)
      return F.cross_entropy(logits.flatten(0,1), action_tokens[:, 1:].flatten())
```

The cross-entropy target is unchanged. The only training-time difference is the additional condition tokens.

Frozen-tokenizer semantics:

- Baseline uses `with torch.inference_mode()` for tokenization.
- Ours uses `with torch.no_grad()` (functionally equivalent in this context, slight implementation difference — `inference_mode` is more aggressive about disabling autograd machinery).

### 3.5 Inference — past buffer lifecycle

The baseline's `predict_action` is stateless and `reset()` is a no-op. Ours adds an explicit buffer:

```python
def reset(self):
    self._past_buffer = None      # cleared at the start of every rollout episode

def predict_action(self, obs_dict, ...):
    features = self.obs_encoder(obs_dict)              # (B, 2, d)

    # lazy zero-init on first call (and on shape/device mismatch)
    if (self._past_buffer is None
        or self._past_buffer.shape[0] != B
        or self._past_buffer.device != self.device):
        self._past_buffer = torch.zeros(B, 7, 7, device=self.device, dtype=features.dtype)

    cond = self._build_condition(features, self._past_buffer)        # (B, 11, d)

    action_tokens = self.model.generate(BOS, cond=cond, max_new_tokens=8, ...)[:, 1:]
    action_pred   = self.action_tokenizer.detokenize(action_tokens)  # (B, 16, 7)
    action        = action_pred[:, : self.n_action_steps]            # (B, 8, 7)

    # Update past buffer with predicted (not executed) actions.
    n_exec = self.n_action_steps    # 8
    past_n = self.past_n             # 7
    if n_exec >= past_n:                                       # 8 >= 7 → yes
        self._past_buffer = action_pred[:, n_exec - past_n : n_exec]   # = action_pred[:, 1:8]
    else:
        self._past_buffer = torch.cat([self._past_buffer[:, n_exec:],
                                       action_pred[:, : n_exec]], dim=1)
    return {"action": action, "action_pred": action_pred}
```

The buffer holds the *last 7 predicted actions* — exactly the same form as `batch["past_action"]` in training.

### 3.6 Optimizer group composition

Baseline ([oat/policy/oatpolicy.py](oat/oat/policy/oatpolicy.py)):

```python
policy_modules = [self.model]
```

Ours ([oat/policy/oat_policy_with_enriched_past.py](oat/oat/policy/oat_policy_with_enriched_past.py)):

```python
policy_modules = [self.model, self.acc_proj, self.jerk_proj, self.raw_proj]
```

In both cases the modules contribute to the `policy_lr` group (5e-5); decay/no-decay splitting is by `param.dim() >= 2`.

---

## 4. Training-pipeline differences

| | Baseline | Ours |
|---|---|---|
| **Task config (policy)** | `task: libero/libero10` | `task: libero/libero10_with_past` |
| **Dataset class** | `ZarrDataset` | `ZarrDatasetWithPastAction` |
| **Extra dataset arg** | — | `past_n: ${past_n}` (7) |
| **`pad_before` in `SequenceSampler`** | `max(To−1, 0) = 1` | `max(To−1, 0) + past_n = 8` |
| **Sample keys** | `{obs, action}` | `{obs, action, past_action}` |
| **`past_action` shape** | — | `(7, 7)` (i.e. `past_n × action_dim`) |
| **Episode-boundary handling** | trailing zero-padding on `action` | trailing zero-padding on `action`; leading zero-padding on `past_action` for early-in-episode samples |

`zarr_dataset_with_past.py` does *not* introduce a new sampling primitive — it reuses `SequenceSampler` with a wider window and slices an extra `past_n` chunk out of the front:

```python
# In _sample_to_data (with To=2, Ta=16, past_n=7):
obs_start    = past_n                                       # = 7
obs_end      = past_n + To                                  # = 9
action_start = past_n + max(To - 1, 0)                      # = 8
action_end   = action_start + Ta                            # = 24
past_action  = sample[action_key][action_start - past_n : action_start]   # = [1:8]
obs          = {k: sample[k][obs_start : obs_end]}          # the To=2 frames preceding action
action       = sample[action_key][action_start : action_end]
```

So one with-past sample spans demo indices `[1, 24)` (instead of `[0, 17)` for the baseline), shifted by `past_n = 7`. The window grows from `1 + 1 + 16 - 1 = 17` to `1 + 8 + 16 - 1 = 24` samples per draw.

---

## 5. Design rationale (as stated in code)

The docstring on `OATPolicyWithEnrichedPast` makes the design intent explicit:

> Observations provide: position (`eef_pos`) + coarse velocity (2-frame diff). Raw past actions provide: exact velocity, temporal patterns, task progress, command inertia. Explicit acc/jerk provide: higher-order derivative info that obs cannot access and that the model would otherwise need to learn to extract via cross-timestep differencing.

For the tokenizer-side change, no docstring; the rationale is implicit in the configuration but consistent with standard practice: action chunks live on a Cartesian × SO(3) × ℝ¹ manifold, so on-manifold augmentation (sampling rotations from the group, not noise on a coordinate representation) is the principled way to expand the rotational support seen during training.

---

## 6. Theoretical-justification deep-dive

### 6.1 What information is *missing* from the baseline conditioning

The baseline's condition is `obs_features ∈ ℝ^(B, 2, d)`, where each frame contains:

- `agentview_rgb (128, 128, 3)` and `robot0_eye_in_hand_rgb (128, 128, 3)` → encoded by ResNet18 + SpatialSoftmax to a fixed feature vector.
- `robot0_eef_pos (3)`, `robot0_eef_quat (4)`, `robot0_gripper_qpos (2)`, `task_uid (1)` → concatenated and identity-projected.

From this, the model can read off **position** and a **one-step backward-Euler velocity estimate** (differencing the two frames). Critically, the *commanded* actions that produced the current state are **not** present in the conditioning; only the *observed* post-controller state is. So the policy must:

1. *Infer* the recent commanded action stream from a state derivative, then
2. *Extrapolate* it forward into the next action chunk.

Step (1) is informationally lossy whenever there's controller dynamics, friction, contact, or actuator lag — i.e. always for real manipulation. The state derivative reflects what the robot *actually did*, not what was *commanded*.

### 6.2 Why raw past actions (the `raw_proj` path)

`past_action` is the *commanded* values directly from the demo. It carries:

- **Exact velocity in command space**: a single difference `a_{t-1} − a_{t-2}` already gives a clean velocity, unaffected by physics noise.
- **Task progress signal**: in a multi-stage task, the commanded action distribution shifts as the policy enters a new phase — see this in raw actions directly.
- **Command inertia**: if the previous chunks were all heading in one direction, the new chunk's first step is unlikely to reverse — providing this as a soft prior costs the model nothing to ignore (since it's a separate token in the condition).

Giving the model 7 past commands (rather than just 1) lets it pick up on multi-step rhythms (e.g. an oscillation around a target) that are not visible in a single-step delta.

### 6.3 Why also explicit acc/jerk (the `acc_proj`, `jerk_proj` paths)

A transformer with cross-attention to the 7-token past sequence can *in principle* learn second- and third-order derivatives by allocating attention heads to position-shifted subtraction. But:

- Doing so consumes head capacity that could go to cross-modal binding (vision ↔ action).
- The features are simple fixed linear combinations of three positions in the input — exactly the kind of pre-computable feature that pre-feeding into the model saves training compute.

Mathematically `acc = a_{t-1} − a_{t-2}` and `jerk = a_{t-1} − 2a_{t-2} + a_{t-3}` are the standard central-difference approximations of the first and second time-derivative. For smooth manipulation, *jerk* matters because human demos are typically jerk-minimal; deviations from "small jerk" are informative about phase transitions and contact events.

This is the classical engineering pattern of "features the model could learn, but shouldn't have to": handing them in as their own condition tokens is essentially free in parameters and saves capacity for the genuinely hard parts.

### 6.4 Why three separate projection MLPs (and not one shared)

The three inputs to projection live on very different scales after `LinearNormalizer.normalize` (limits-mode, output range `[-1, +1]`):

- `norm_past[t] ∈ [-1, +1]^7`
- `acc = a_{t-1} − a_{t-2}` is the consecutive-step difference in normalized space — typically O(0.01) to O(0.1).
- `jerk = a_{t-1} − 2a_{t-2} + a_{t-3}` is the second difference — typically O(0.001) to O(0.01).

A *shared* MLP would either be dominated by the `norm_past` magnitudes (acc/jerk effectively become "noise" at initialization), or need to learn input-dependent scaling on top of the projection. Three small MLPs decouple this cleanly:

- Total cost: `3 × (7·d + d² + d)` parameters ≈ a few × 100K for typical `d ≈ 80`. Negligible vs. the 4-layer AR backbone (~ 2-3 M).
- Each branch settles at its natural gain without a fight.

The choice to make `raw_proj` *shared across the 7 time positions* (rather than 7 separate MLPs) is also deliberate: it imposes a translation-invariance prior on the past axis, which is consistent with the AR backbone's learned positional embeddings handling the position-specific information.

### 6.5 Why SO(3) augmentation rather than e.g. Gaussian noise on the rotation slice

Robot orientation actions in this codebase are stored as **Rodrigues vectors** (axis × angle, 3 dims). Two reasons why Gaussian additive noise on this representation is wrong:

1. **Non-equivariance to the manifold.** Adding `ε ~ N(0, σ²)` to the rotvec `[0, 0, π]` and `[0, 0, π + ε_3]` represents radically different rotations (the latter is "rotation by `π + ε` around z" which is *almost equivalent to* rotation by `−π + ε` around z, due to the angular wraparound). Gaussian noise is not a meaningful "small perturbation" on this manifold.
2. **Off-manifold samples.** Even for small angles where the Rodrigues representation is locally Euclidean, large additive noise can put the sample outside the regular charts of the manifold. Composition `Q · R` *always* yields a valid rotation.

The principled alternative is to sample directly on `SO(3)`:

```
axis n ~ Uniform(S²)
angle θ ~ Uniform([0, max_angle_rad])
Q = expmap(θ · n)  ∈ SO(3)
R_aug = Q · R       # left-multiplication
```

This generates a uniformly-distributed perturbation up to a max angle, while always producing a valid rotation. For the encoder, this means the rotation slice of the input distribution now spans more of the manifold than what's in the LIBERO demos — pushing it to learn codes that distinguish *relative* rotation patterns rather than memorize the *absolute* rotation cone seen in training.

### 6.6 Why `mode = left_noise` (and `augment_position = false`)

The three modes correspond to different physical interpretations:

| Mode | Formula | Interpretation |
|---|---|---|
| `left_noise` | `Q · R` | Perturb the world / reference frame. The robot's *intended* gripper orientation in its own frame is unchanged; what changes is the world that the action is described relative to. |
| `right_noise` | `R · Q` | Perturb the end-effector's local frame. The action *means something different* (e.g., a different roll for the gripper). |
| `conjugate` | `Q · R · Qᵀ` | Symmetric, frame-invariant rotation of the rotation itself. More aggressive. |

`left_noise` is the most task-preserving choice — it simulates "what if the camera/table were tilted a bit relative to where the demo was recorded" without changing the semantic intent of the action. This is *exactly* the variation that the tokenizer should be robust to: a code that means "rotate the gripper towards the can" should look the same whether the can is on the left or the right of the table.

Consistent with this interpretation, `augment_position = false`: under a pure rotation of the *reference frame*, the position deltas would in principle also rotate, but in the current action representation (relative deltas in a fixed task frame), the position slice is left untouched. This decoupling forces the encoder to learn **rotation invariance for the orientation slice specifically**, without conflating it with translation invariance.

If `augment_position` were set to `true`, the full action delta vector would be rotated by `Q`, modeling a full SE(3) frame change — a different (stronger) augmentation. The choice to leave position fixed is deliberate.

### 6.7 Why `p = 0.6` and `max_angle_deg = 60°`

Two configured magnitudes, both larger than typical augmentation defaults:

- **`p = 0.6`**: aggressive — the augmented distribution is larger than the un-augmented one. But `p < 1.0` keeps 40 % of batches as anchors, so reconstruction loss is still well-targeted on un-augmented data and the test/eval distribution (which has no augmentation) doesn't drift.
- **`max_angle_deg = 60°`**: large by RL-augmentation standards. The yaml shows two commented-out alternatives (`10°`, `30°`) — `60°` is presumably the result of a sweep. `60°` per chunk is much larger than the rotational motion *within* a single 16-step chunk (which would be small for smooth demos), but matches the rotational variation *across* demos (where the same skill is performed from different angles).

The combination is consistent with treating SO(3) augmentation as **codebook-broadening** rather than as a small smoothness regularizer.

### 6.8 Why the past buffer is *predicted*, not *executed*, actions

At training time, `past_action` is sliced directly from the demo trajectory — the recorded commanded action stream. At inference, the closest thing to that stream is the policy's *own previous predictions*. Specifically, `MultiStepWrapper` sends `action_pred[:, :8]` to the controller (which then runs internal subcontrollers for each step). Those 8 commanded values are exactly what *would have been* recorded if this rollout were demo data.

The choice `_past_buffer = action_pred[:, n_exec - past_n : n_exec]` (which is `action_pred[:, 1:8]` for `n_exec=8`, `past_n=7`) keeps the most recent 7 predictions for use as the next call's `past_action`. This is the *only* train-test-matched choice:

- Using `info["last_action"]` (post-controller) or differencing proprioception would introduce post-controller noise that training never sees.
- Using ground-truth demo actions is unavailable at deploy time.

The cost is that errors compound: a prediction error feeds back into the next call's condition. The hope is that the AR codes are robust enough that small action-space errors don't push the policy off-distribution. Empirically this is the standard pattern in autoregressive control (cf. ACT, Diffusion Policy with action history).

### 6.9 Why both changes — and not just one

The two changes target different failure modes that compose:

- **SO(3) augmentation** improves the *tokenizer's* generalization to rotation variation. Better codes → smaller codebook collapse → richer action vocabulary for the policy to use.
- **Enriched past** improves the *policy's* sample efficiency — it doesn't have to re-derive temporal information that's directly available.

Either change alone is meaningful but limited:

- *Just SO(3) aug*: the policy still has to infer command history from short observations. Better tokens, same conditioning bottleneck.
- *Just enriched past*: the codebook may still alias rotations seen rarely in demos, so improvements in conditioning hit a representation ceiling.

Bundling them gives compounding gains: better codebook coverage × better conditioning signal.

### 6.10 Why everything else is held constant

The rest of the architecture (encoder/decoder/quantizer hyperparameters, observation encoder, AR backbone width/depth/heads, optimizer settings, EMA, checkpoint cadence, action/observation horizons) is held identical between baseline and ours. This is the standard scientific practice for ablation: any improvement on the rollout SR metric is attributable to *this specific pair of changes*, not to incidental hyperparameter retuning.

It also means the frozen SO(3)-augmented tokenizer can be plug-replaced into the baseline policy (and vice versa), enabling clean per-axis ablation studies:

| Configuration | Tokenizer | Policy | Purpose |
|---|---|---|---|
| Baseline | `OATTok` | `OATPolicy` | reference |
| Tok-only | `OATTokSO3Aug` | `OATPolicy` | isolates SO(3) aug effect |
| Policy-only | `OATTok` | `OATPolicyWithEnrichedPast` | isolates enriched-past effect |
| Ours (full) | `OATTokSO3Aug` | `OATPolicyWithEnrichedPast` | combined |

---

## 7. What is intentionally identical

Worth listing explicitly so reviewers see the surface area of the change:

- **Action shape** (`D=7`), **horizon** (`T=16`), **executed horizon** (`n_action_steps=8`), **obs horizon** (`To=2`).
- **`RegisterEncoder`** — same `emb_dim=256`, `head_dim=64`, `depth=2`, `pdropout=0.1`, `latent_dim=4`, `num_registers=8`. Same "causal-last" attention mask.
- **`SinglePassDecoder`** — same `emb_dim=256`, `head_dim=64`, `depth=4`, `token_dropout_mode=pow2`, `use_causal_decoder=true`.
- **`FSQ`** — same `levels=[8, 5, 5, 5]`, codebook size 1000.
- **`FusedObservationEncoder`** — same RGB encoder (`RobomimicRgbEncoder` with `ResNet18Conv` + `SpatialSoftmax` + `GroupNorm`, `crop_shape=[76, 76]`) and state encoder (`ProjectionStateEncoder`, identity projection).
- **`AutoregressiveModel`** — same `embed_dim=256`, `n_layers=4`, `n_heads=4`, `dropout=0.1`. Same KV-cached generation, same top-k sampling (`temperature=1.0`, `topk=10`).
- **Optimizer / schedule / EMA** — same AdamW, `(0.9, 0.95)` betas, `weight_decay=0`, constant LR with 100 warmup steps, EMA `power=0.75, max=0.9999`.
- **Training cadence** — same `num_epochs=5001`, `val_every=10`, `sample_every=10`, `checkpoint_every=10`, `rollout_every=200`, `max_grad_norm=1.0`, `allow_bf16=True`.
- **Checkpointing** — same monitors and top-3 strategy.

---

## 8. Caveats and open questions

1. **Past buffer holds predicted, not executed, actions.** This is the correct matched-distribution choice (see § 6.8), but it means prediction errors *can* feed back into the condition. Robustness depends on the AR codes being insensitive to small numerical perturbations in `_past_buffer`. Empirically untested in isolation; would benefit from a controlled study.

2. **SO(3) augmentation only on rotation slice.** Position remains in its original frame even though the rotation slice is perturbed. This is a deliberate decoupling (see § 6.6) but means the augmented samples are not consistent SE(3) frame transforms of the originals — they are *partial* frame transforms. If position is in any way coupled to rotation in the underlying task (e.g., end-effector trajectories that depend on orientation), this could create a small mismatch.

3. **`mode='left_noise'` perturbs the world frame.** If the task semantics depend on the absolute world frame (e.g., gravity-aligned actions), `left_noise` is *too* strong — it implicitly assumes the task is gravity/world-rotation invariant. For LIBERO benchmarks this is broadly true (tabletop manipulation), but might not transfer to other domains.

4. **Numerical stability of `so3_log_map` near identity.** The `logmap` formula has a divide-by-zero near `θ ≈ 0`. The implementation handles this via Taylor expansion for `θ² < 1e-8`, but very small (or very near-identity) `R_aug` could in principle introduce numerical noise. The implementation upcasts to fp32 for the SO(3) ops then casts back — this should be sufficient in practice.

5. **`max_angle_deg=60°` is large.** The yaml shows it was sweep-tuned (10°, 30°, 60° all present, only 60° uncommented). For tasks with finer rotational structure, `60°` may be too coarse and a smaller value should be tried.

6. **Loss signal is unchanged.** Both methods optimize MSE (tokenizer) and cross-entropy (policy). No new auxiliary losses (e.g. equivariance, contrastive, or smoothness terms) are added. The whole improvement is via the *training distribution* and the *conditioning signal*, not the loss.

7. **`task_uid` as a state channel.** Worth flagging that `task_uid` (shape `[1]`) is encoded alongside other state ports through `ProjectionStateEncoder`. This is shared by baseline and ours, but it's worth knowing because it makes the encoder *task-aware* — the model has access to a one-dim task identifier even without language tokens.
