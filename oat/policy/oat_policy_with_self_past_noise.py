import torch
import torch.nn.functional as F
from typing import Dict, Optional

from oat.policy.oat_policy_with_self_past import OATPolicyWithSelfPast
from oat.tokenizer.oat.tokenizer import OATTok
from oat.perception.base_obs_encoder import BaseObservationEncoder


class OATPolicyWithSelfPastNoise(OATPolicyWithSelfPast):
    """
    OATPolicyWithSelfPast plus Gaussian noise on the past-action conditioning.

    Training-time past_action pipeline:

        ground truth  ->  self-generated (policy re-run on prev_obs)  ->  + noise
        [dataset]         [OATPolicyWithSelfPast]                        [here]

    Everything else is inherited untouched: architecture, condition layout,
    loss, and the whole inference path.

    Where the noise is applied
    --------------------------
    In NORMALIZED action space, not raw space.  `LinearNormalizer` is min-max
    over the demo set, so normalized actions occupy exactly [-1, 1] on every
    dimension and a single `past_noise_std` means the same thing for all of
    them.  Raw dimensions do not share a scale (translation deltas reach
    +-0.94 while rotation deltas stay inside +-0.38), so raw-space noise would
    hit the rotation channels several times harder.

    Implementation keeps `_build_condition` untouched: the past is normalized,
    perturbed, and un-normalized here, and `_build_condition` re-normalizes it
    downstream.  That affine round trip is exact, so the model sees exactly
    `normalize(past) + noise`.

    Because `acc` and `jerk` are differences of normalized past actions inside
    `_build_condition`, the noise propagates into them automatically — and is
    amplified, as differencing always does: std_acc = sqrt(2)*std,
    std_jerk = sqrt(6)*std for i.i.d. noise.  That is intended; those features
    are exactly the ones that are fragile to an imperfect past.

    Where the noise is NOT applied
    ------------------------------
    * Not at inference — `predict_action` is inherited unchanged.
    * Not inside `_generate_prev_past`'s inner rollout simulation.  That call
      reproduces what the policy actually did on the previous chunk, and at
      rollout its past buffer is not noised.  Noise is added only to the past
      that conditions the main training forward.

    Choosing past_noise_std
    -----------------------
    Per-dimension std of NORMALIZED actions in libero10_N500:

        [0.308, 0.368, 0.380, 0.131, 0.162, 0.238, 0.996]
         --------- xyz ------  ------ rot ------  gripper

    So an absolute std of 0.05 is ~16% of signal std on the translation axes
    but ~38% on the tightest rotation axis.  Set `past_noise_relative=True` to
    scale the noise per dimension by the batch's own normalized std instead,
    which equalizes the relative perturbation across channels.

    Knobs
    -----
    past_noise_std           noise std in normalized units (0 disables)
    past_noise_relative      scale per dim by batch std instead of absolute
    past_noise_p             per-sample probability of receiving noise
    past_noise_warmup_steps  train steps before noise starts
    past_noise_clip          clamp the noised past back into [-1, 1], the
                             exact range min-max normalization produces
    """

    def __init__(
        self,
        shape_meta: Dict,
        obs_encoder: BaseObservationEncoder,
        action_tokenizer: OATTok,
        n_action_steps: int,
        n_obs_steps: int,
        past_n: int = 7,
        # policy model params
        embed_dim: int = 512,
        n_layers: int = 8,
        n_heads: int = 8,
        dropout: float = 0.1,
        # policy inference params
        temperature: float = 1.0,
        topk: int = 10,
        # self-past params
        self_past_p: float = 1.0,
        self_past_warmup_steps: int = 500,
        self_past_temperature: Optional[float] = None,
        self_past_topk: Optional[int] = None,
        # ── past-action noise params ────────────────────────────────────
        past_noise_std: float = 0.05,
        past_noise_relative: bool = False,
        past_noise_p: float = 1.0,
        past_noise_warmup_steps: int = 0,
        past_noise_clip: bool = True,
    ):
        super().__init__(
            shape_meta=shape_meta,
            obs_encoder=obs_encoder,
            action_tokenizer=action_tokenizer,
            n_action_steps=n_action_steps,
            n_obs_steps=n_obs_steps,
            past_n=past_n,
            embed_dim=embed_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout,
            temperature=temperature,
            topk=topk,
            self_past_p=self_past_p,
            self_past_warmup_steps=self_past_warmup_steps,
            self_past_temperature=self_past_temperature,
            self_past_topk=self_past_topk,
        )

        self.past_noise_std = past_noise_std
        self.past_noise_relative = past_noise_relative
        self.past_noise_p = past_noise_p
        self.past_noise_warmup_steps = past_noise_warmup_steps
        self.past_noise_clip = past_noise_clip

        print(
            f"  past noise   : std={past_noise_std}"
            f"{' (relative)' if past_noise_relative else ' (absolute, normalized units)'}, "
            f"p={past_noise_p}, warmup={past_noise_warmup_steps} steps, "
            f"clip={past_noise_clip}\n"
        )

    def get_policy_name(self):
        base_name = "oatpolicy_selfpastnoise_"
        for modality in self.modalities:
            if modality != "state":
                base_name += modality + "|"
        return base_name[:-1]

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _maybe_add_noise(self, past_actions: torch.Tensor) -> torch.Tensor:
        """
        Add Gaussian noise to `past_actions` in normalized space.

        Args:
            past_actions: (B, past_n, action_dim) raw (unnormalized)

        Returns:
            (B, past_n, action_dim) raw (unnormalized), perturbed
        """
        if self.past_noise_std <= 0.0 or self.past_noise_p <= 0.0:
            return past_actions
        if self._train_step < self.past_noise_warmup_steps:
            return past_actions

        normalizer = self.action_normalizer["action"]
        norm_past = normalizer.normalize(past_actions)

        std = self.past_noise_std
        if self.past_noise_relative:
            # per-dim std over (batch, past_n); clamped so a degenerate
            # channel cannot silently zero out its noise
            std = std * norm_past.std(dim=(0, 1), keepdim=True).clamp_min(1e-6)

        noise = torch.randn_like(norm_past) * std

        if self.past_noise_p < 1.0:
            keep = (
                torch.rand(
                    norm_past.shape[0], 1, 1, device=norm_past.device
                ) < self.past_noise_p
            )
            noise = noise * keep

        noisy = norm_past + noise
        if self.past_noise_clip:
            noisy = noisy.clamp(-1.0, 1.0)

        return normalizer.unnormalize(noisy)

    # ── Training ────────────────────────────────────────────────────────────

    def forward(self, batch) -> torch.Tensor:
        # tokenize ground-truth actions (frozen tokenizer)
        with torch.no_grad():
            action_tokens = self.action_tokenizer.tokenize(batch["action"])

        B = batch["action"].shape[0]
        device = batch["action"].device

        # encode observation
        features = self.obs_encoder(batch["obs"])       # (B, To, d)

        # ── past actions: policy's own output, then perturbed ─────────────
        past_actions = batch["past_action"]              # (B, past_n, action_dim)
        past_actions = self._maybe_self_past(batch, past_actions)
        past_actions = self._maybe_add_noise(past_actions)

        # ── build extended condition ──────────────────────────────────────
        cond = self._build_condition(features, past_actions)

        # prepend <BOS> token
        action_tokens = torch.cat([
            torch.full(
                (B, 1), self.bos_id,
                dtype=torch.long, device=device,
            ),
            action_tokens,
        ], dim=1)

        # forward model
        logits = self.model(action_tokens[:, :-1], cond=cond)

        # compute loss
        vocab_size = logits.size(-1)
        loss = F.cross_entropy(
            logits.reshape(-1, vocab_size),
            action_tokens[:, 1:].reshape(-1),
        )

        # increment step counter (used by _maybe_self_past / _maybe_add_noise)
        self._train_step += 1

        return loss
