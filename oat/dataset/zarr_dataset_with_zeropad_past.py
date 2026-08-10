import numpy as np
import torch
from typing import Dict, List, Optional

from oat.dataset.zarr_dataset_with_past import ZarrDatasetWithPastAction


class ZarrDatasetWithZeroPadPastAction(ZarrDatasetWithPastAction):
    """
    ZarrDatasetWithPastAction variant that ZERO-pads `past_action` at episode
    starts, instead of the SequenceSampler's edge repetition.

    Why this exists
    ---------------
    `SequenceSampler` pads short windows by replicating the boundary frame
    (`data[:sample_start_idx] = sample[0]`, seq_sampler.py:151).  For
    `past_action` that means the episode's first ground-truth action is copied
    backwards, so at training time the model is told "the robot was already
    commanding a_0" before the episode began.

    At rollout time no such value exists — `a_0` is exactly what the policy is
    about to predict — so `OATPolicyWithEnrichedPast.reset()` seeds
    `_past_buffer` with zeros.  Every eval episode therefore opens in a state
    that training has essentially never produced.

    The mismatch cannot be fixed from the inference side (the value is
    unknowable there), so it is fixed here: padded `past_action` entries become
    raw zeros, matching `reset()` exactly.

    Consistency notes
    -----------------
    * Zeros are written in RAW (unnormalized) action space, because that is
      what `_past_buffer` holds; both paths then go through
      `action_normalizer["action"].normalize(...)` inside `_build_condition`,
      so the normalized conditioning vectors match too.
    * `obs` is deliberately left edge-padded: the rollout wrapper already
      edge-pads obs at t=0 (`stack_last_n_obs`, multistep_wrapper.py:63-65),
      so that channel is consistent as-is.
    * `action` (the prediction target) is never touched.

    Padding geometry (To = n_obs_steps, past_n, from the parent's
    `_sample_to_data`):

        action_start = past_n + max(To - 1, 0)
        past_action  = sample[action_start - past_n : action_start]
                     = sample[max(To - 1, 0) : action_start]

    A window's front padding occupies sample indices [0, sample_start_idx), so
    the number of padded `past_action` entries is

        n_pad = clip(sample_start_idx - max(To - 1, 0), 0, past_n)

    With To=2, past_n=7 the slice is sample[1:8] and `sample_start_idx` ranges
    over [0, pad_before=8], giving n_pad in [0, 7].

    zero_whole_past
    ---------------
    With n_action_steps >= past_n the inference buffer is fully overwritten
    after the first chunk, so rollout only ever sees an all-zero past or a
    fully-populated one — never a partially-zero one.  Set
    `zero_whole_past=True` to reproduce that exactly (any padding at all zeroes
    the entire `past_action`), at the cost of discarding a few real actions in
    the ~past_n windows at each episode start.  Default False keeps the
    minimal, entry-wise correction.
    """

    def __init__(
        self,
        past_n: int = 7,
        zero_whole_past: bool = False,
        # all ZarrDatasetWithPastAction / ZarrDataset args
        zarr_path: str = "",
        obs_keys: List[str] = [],
        action_key: str = "action",
        n_obs_steps: int = 2,
        n_action_steps: int = 16,
        seed: int = 42,
        val_ratio: float = 0.0,
        max_train_episodes: Optional[int] = None,
    ):
        super().__init__(
            past_n=past_n,
            zarr_path=zarr_path,
            obs_keys=obs_keys,
            action_key=action_key,
            n_obs_steps=n_obs_steps,
            n_action_steps=n_action_steps,
            seed=seed,
            val_ratio=val_ratio,
            max_train_episodes=max_train_episodes,
        )
        self.zero_whole_past = zero_whole_past

    # ── Helpers ─────────────────────────────────────────────────────────────

    def past_pad_len(self, idx: int) -> int:
        """
        Number of leading `past_action` entries that are SequenceSampler
        padding rather than real recorded actions.
        """
        # (buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx)
        sample_start_idx = int(self.seq_sampler.indices[idx][2])
        past_start = max(self.n_obs_steps - 1, 0)
        n_pad = sample_start_idx - past_start
        return int(min(max(n_pad, 0), self.past_n))

    # ── Dataset interface ───────────────────────────────────────────────────

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data = super().__getitem__(idx)

        n_pad = self.past_pad_len(idx)
        if n_pad > 0:
            # `past_action` came from `.astype(np.float32)` in the parent's
            # `_sample_to_data`, i.e. a fresh array — writing in place here
            # cannot alias the replay buffer.
            if self.zero_whole_past:
                data["past_action"].zero_()
            else:
                data["past_action"][:n_pad] = 0.0

        return data
