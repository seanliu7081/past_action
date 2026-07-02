# Trajectory Smoothness Evaluation

How the OAT LIBERO smoothness eval works — from the executed policy actions to
the numbers in `eval_log.json` — and how to read and use the metrics.

All file paths below are relative to the repo root (`/workspace/oat`).

---

## 1. TL;DR

- Smoothness is measured on the **action stream the policy actually executes**
  during a LIBERO rollout (not the observed robot state).
- Each episode's action trajectory is split into a **position** part (dims 0–2)
  and a **rotation** part (dims 3–5); the gripper dim (6) is ignored.
- For each part, 4 metrics are computed → **8 metrics total**:
  `vel`, `acc`, `jerk` (time-domain finite differences) and `sparc`
  (frequency-domain Spectral Arc Length).
- Metrics are averaged per task and over all episodes, then written to
  `eval_log.json` alongside the success rate.
- The numbers are only meaningful as **relative comparisons** (policy A vs B,
  checkpoint vs checkpoint, task vs task) — not as absolute physical quantities.

---

## 2. Files in the pipeline

| File | Role |
|---|---|
| `scripts/eval_policy_sim_smoothness.py` | Thin CLI wrapper. Computes **no** smoothness itself — it only swaps the env runner and runs the normal eval. |
| `oat/env_runner/libero_runner_smoothness.py` | `LiberoRunnerSmoothness`: captures executed actions during the rollout and reassembles per-episode trajectories. |
| `oat/eval/trajectory_smoothness.py` | The actual math: velocity / acceleration / jerk / SPARC. |
| `oat/env_runner/libero_runner.py` | Base `LiberoRunner`: the unchanged rollout loop (chunked parallel envs, one `policy.reset()` per chunk). |
| `oat/tokenizer/oat/augment/so3_action_chunk_aug.py` | `so3_exp_map` / `so3_log_map`, used only by the optional `geodesic` rotation mode. |

---

## 3. How the code evaluates smoothness

### Step 1 — The wrapper redirects the runner

`scripts/eval_policy_sim_smoothness.py` is a copy of `eval_policy_sim.py`. Its
only added logic is `_swap_runner_to_smoothness`, which mutates the checkpoint's
Hydra config in memory so the env runner's `_target_` points at
`LiberoRunnerSmoothness` instead of the plain `LiberoRunner`, and injects three
kwargs:

- `smoothness_rot_mode` (`euclidean` | `geodesic`)
- `smoothness_pos_slice` (default `(0, 3)`)
- `smoothness_rot_slice` (default `(3, 6)`)

The checkpoint on disk is untouched. Everything else — load policy, instantiate
runner, `env_runner.run(policy)` × `num_exp`, dump JSON — is identical to the
standard eval.

### Step 2 — Capture the executed actions (monkey-patching)

`LiberoRunnerSmoothness` does **not** rewrite the rollout loop. For the duration
of `run()` it wraps two policy methods (`oat/env_runner/libero_runner_smoothness.py`):

- **`policy.reset`** → bumps a `current_chunk` counter. This works because the
  base runner calls `policy.reset()` **exactly once per chunk** of parallel envs.
- **`policy.predict_action`** → grabs every returned `action` chunk
  (`[n_envs, T_chunk, D]`), detaches it to a NumPy copy, and files it under the
  current chunk index.

So `captured[chunk_idx]` accumulates the action chunks the policy emitted for
that group of parallel environments. The patches are restored in a `finally`.

### Step 3 — Reassemble per-episode trajectories

`_reassemble_episodes` stacks each chunk's captured calls into
`[n_calls, n_envs, T_chunk, D]`, then for each active env slices out that env's
stream and flattens time across all prediction calls → one `[T_ep, D]` array per
episode. Episode `i` maps to `chunk_idx * n_envs + env_local`, which is exactly
how the base runner slices envs into global indices. Padding envs (duplicates
used to fill a short final chunk) are excluded — only the first `k_active` envs
per chunk are read.

### Step 4 — Compute the 8 metrics per episode

For each episode with **≥ 4 timesteps**, `compute_smoothness_metrics`
(`oat/eval/trajectory_smoothness.py`) splits the action vector into the position
slice and rotation slice and computes 4 metrics for each. See §4.

### Step 5 — Aggregate and write out

`_populate_smoothness_metrics`:

- averages each metric over all scored episodes → `rollout_<key>`
- also averages within each task → `<task>/rollout_<key>`
- records `rollout_n_episodes_scored` (episodes with ≥ 4 steps)

These are merged into the same `log_data` dict as the success rates. The wrapper
then averages every numeric key across `num_exp` runs (adding `_std`/`_stderr`
if `num_exp > 1`) and writes `eval_log.json`.

---

## 4. The metrics

Two motion components, evaluated separately:

- **position** (`_pos`): translation, action dims 0–2
- **rotation** (`_rot`): orientation, action dims 3–5 (axis-angle)

Let `x_t` be one component's trajectory, `dt = 1/fs` with `fs = 20 Hz`, and let
`Δⁿ` be the n-th finite difference along time.

| Metric | Definition | Direction |
|---|---|---|
| **`vel_*`**   | `mean_t ‖Δ¹ x_t‖ / dt`     | mean speed — lower = calmer |
| **`acc_*`**   | `mean_t ‖Δ² x_t‖ / dt²`    | mean acceleration — lower = smoother |
| **`jerk_*`**  | `mean_t ‖Δ³ x_t‖ / dt³`    | mean jerk — **lower = smoother** (sharpest signal) |
| **`sparc_*`** | Spectral Arc Length of the speed profile | **closer to 0 = smoother**, more negative = jerkier |

**SPARC (Spectral Arc Length, Balasubramanian et al.)** is the frequency-domain
metric and the odd one out:

1. Build the speed profile `v_t = ‖Δ¹ x_t‖ / dt`.
2. FFT it (zero-padded) and take the magnitude spectrum.
3. Normalize the spectrum by its peak (makes it amplitude-invariant).
4. Keep frequencies ≤ `fc = 10 Hz` whose normalized amplitude ≥ `amp_th = 0.05`.
5. SPARC = **−(arc length of that normalized spectrum curve)**.

A jerky motion has a broader / more wrinkled spectrum → longer arc → more
negative SPARC. Because it is scale-invariant, it captures the *shape* of the
motion rather than its magnitude.

The 8 keys (`SMOOTH_KEYS`):

```
vel_pos   acc_pos   jerk_pos   sparc_pos
vel_rot   acc_rot   jerk_rot   sparc_rot
```

---

## 5. How the metrics are used — reading `eval_log.json`

Every metric appears at two scopes:

- **Overall** (bare keys): `rollout_vel_pos_mean`, … — averaged over all scored
  episodes. Use these as the single-policy summary.
- **Per task** (task-name prefix): `KITCHEN_SCENE3_.../rollout_vel_pos_mean`, … —
  averaged within one task. Use these to see *where* the policy moves roughly.

The `_mean` suffix exists because the wrapper averages across `num_exp` runs.
With `num_exp = 1`, `_mean` is just the single value and there are no
`_std`/`_stderr` keys.

Bookkeeping keys:

| Key | Meaning |
|---|---|
| `checkpoint` | which model was evaluated |
| `num_exp` | number of eval passes (error bars only if > 1) |
| `smoothness_rot_mode` | `euclidean` (correct for LIBERO delta actions) |
| `rollout_n_episodes_scored_mean` | episodes with ≥ 4 steps that were scored (coverage check) |
| `mean_success_rate_mean` | overall task success on this run |

### Reading rules

1. **Direction:** lower `vel`/`acc`/`jerk` = smoother; `sparc` closer to 0 =
   smoother.
2. **Never compare `_pos` vs `_rot`** — different units (translation
   action-units vs axis-angle radians). Compare pos-to-pos and rot-to-rot only.
3. **Only relative comparisons are meaningful.** Values are in the policy's
   action units (LIBERO OSC_POSE deltas, roughly [-1, 1] per step) scaled by an
   assumed 20 Hz — not SI m/s. They matter only against another policy /
   checkpoint evaluated identically.
4. **Smoothness ≠ success.** These metrics measure motion quality (relevant for
   real-robot transfer, actuator wear, safety), not task competence. A jerky
   policy can still succeed, and a smooth one can still fail.

### Typical use cases

- **Compare two checkpoints / policies:** run the same eval for each, then
  compare `rollout_jerk_*` (lower better) and `rollout_sparc_*` (closer to 0
  better). Jerk and SPARC are the two standard headline smoothness numbers.
- **Regression check:** confirm a new training recipe didn't make motion jerkier
  while chasing success rate.
- **Per-task diagnosis:** find tasks with unusually high jerk / very negative
  SPARC and inspect the rendered rollout videos (`<task>/video_<seed>_*` paths in
  the log) to see what that looks like.

---

## 6. Caveats & gotchas (verified against the source)

- **`vel`/`acc`/`jerk` are mean-of-norms, not true RMS.** Despite names like
  `velocity_rms`, the helper `_norm_rms` computes `x.norm(-1).mean()` (mean of
  per-step Euclidean norms), not `sqrt(mean(‖·‖²))`. Read them as mean
  speed/accel/jerk. Internally consistent, so fine for ranking.
- **`fs` is effectively hard-coded to 20 Hz.** `LiberoRunner` never assigns
  `self.fps`, so the `hasattr(self, "fps")` check always falls back to `20.0`.
  This matches LIBERO's real control rate, so the numbers are valid — but if the
  control rate ever changes, or you compare against a differently-configured
  run, the `vel`/`acc`/`jerk` scales won't line up (they scale as `1/dtⁿ`).
  SPARC is unaffected in relative terms.
- **Use `euclidean` rotation mode for this repo.** LIBERO OSC_POSE actions are
  *delta* rotations already in the tangent space, so plain finite-differencing is
  correct. The `geodesic` mode (compose on SO(3), then log-map) is
  mathematically sound but only meaningful for **absolute-orientation** action
  spaces; feeding delta rotations into it produces meaningless numbers, and the
  code does not guard against this.
- **Early-success tail noise (documented, not a bug).** All envs in a chunk keep
  stepping until every env is done, so an episode that succeeds early includes a
  few trailing actions in its smoothness trajectory. Mild, unbiased noise.
- **Episodes with < 4 timesteps are skipped** (finite differences up to jerk
  need at least 4 samples). `rollout_n_episodes_scored` reports how many counted.

---

## 7. Running it

```bash
python scripts/eval_policy_sim_smoothness.py \
    -c path/to/policy.ckpt \
    -o path/to/output_dir \
    -n 1 \
    --rot-mode euclidean        # euclidean (default) is correct for LIBERO
    # --pos-start 0 --pos-end 3 --rot-start 3 --rot-end 6   # action slice overrides
```

Output: `path/to/output_dir/eval_log.json` (structure described in §5) plus
rendered rollout videos under `path/to/output_dir/media/`.
