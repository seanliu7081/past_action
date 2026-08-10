import contextlib
import torch
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from oat.policy.oat_policy_with_enriched_past import OATPolicyWithEnrichedPast
from oat.tokenizer.oat.tokenizer import OATTok
from oat.perception.base_obs_encoder import BaseObservationEncoder


class OATPolicyWithSelfPast(OATPolicyWithEnrichedPast):
    """
    OATPolicyWithEnrichedPast that conditions on its OWN generated actions as
    `past_action` during training, instead of the dataset's ground truth.

    Everything else — architecture, condition layout, loss, inference path —
    is inherited unchanged.  The only difference from the baseline is where
    the 7 past actions come from.

    How the self-generated past is produced
    ---------------------------------------
    At rollout, chunk k's past buffer is exactly what the policy produced on
    chunk k-1 (oat_policy_with_enriched_past.py:337-343).  So each training
    step re-runs the policy on the observation window one execution stride
    back — supplied by `ZarrDatasetWithPrevWindow` as `prev_obs` /
    `prev_past_action` — and applies the same buffer-update rule:

        prev_pred = detokenize(generate(cond(prev_obs, prev_past_action)))
        past      = prev_pred[:, n_exec - past_n : n_exec]          # n_exec >= past_n

    That is one level of unrolling: the inner call still uses ground-truth
    past, so this closes the "own predictions" gap at depth 1, not to
    convergence.  Going deeper would need k nested generations per step.

    Faithfulness details
    --------------------
    * The inner generation runs in eval mode (dropout off, center crop rather
      than random crop) and with the policy's own `temperature` / `topk`,
      because that is what rollout does.  Module modes are restored afterwards.
    * It runs under `torch.inference_mode()` and is detached before entering
      the condition — no gradient flows through generation, which is discrete
      and non-differentiable anyway.
    * `self.action_tokenizer`'s train/eval mode is restored to whatever it was,
      deliberately preserving the baseline's behaviour (the workspace's
      `model.train()` flips the frozen tokenizer back into train mode) so this
      variant stays comparable to `train_oatpolicy_with_enriched_past`.
    * Boundary windows are NOT zero-padded — this variant keeps the parent's
      edge repetition, matching the baseline config.

    Cost
    ----
    Each training step adds one obs-encoder forward over `prev_obs` plus an
    autoregressive generation of `latent_horizon` tokens.  Measured at
    batch_size=64 on a single RTX 4090 with bf16 autocast: 60.6 -> 88.9
    ms/step, i.e. 1.47x.  `self_past_warmup_steps` skips the extra work
    entirely, so those steps run at baseline speed.

    Knobs
    -----
    self_past_p             per-sample probability of using the generated past
                            (1.0 = always, the intended experiment; <1.0 gives
                            scheduled-sampling style mixing with ground truth)
    self_past_warmup_steps  train steps before substitution starts
    self_past_temperature   sampling temperature for the inner generation
                            (None -> self.temperature)
    self_past_topk          top-k for the inner generation (None -> self.topk)
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
        # ── self-past params ────────────────────────────────────────────
        self_past_p: float = 1.0,
        self_past_warmup_steps: int = 500,
        self_past_temperature: Optional[float] = None,
        self_past_topk: Optional[int] = None,
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
        )

        self.self_past_p = self_past_p
        self.self_past_warmup_steps = self_past_warmup_steps
        self.self_past_temperature = (
            temperature if self_past_temperature is None else self_past_temperature
        )
        self.self_past_topk = topk if self_past_topk is None else self_past_topk

        self._train_step = 0  # incremented in forward()

        print(
            f"  self-past    : p={self_past_p}, warmup={self_past_warmup_steps} steps, "
            f"temp={self.self_past_temperature}, topk={self.self_past_topk}\n"
        )

    def get_policy_name(self):
        base_name = "oatpolicy_selfpast_"
        for modality in self.modalities:
            if modality != "state":
                base_name += modality + "|"
        return base_name[:-1]

    # ── Helpers ─────────────────────────────────────────────────────────────

    @contextlib.contextmanager
    def _rollout_mode(self):
        """Temporarily put every module used by generation into eval mode."""
        modules = [
            self.obs_encoder, self.model, self.action_tokenizer,
            self.acc_proj, self.jerk_proj, self.raw_proj,
        ]
        was_training = [m.training for m in modules]
        try:
            for m in modules:
                m.eval()
            yield
        finally:
            for m, mode in zip(modules, was_training):
                m.train(mode)

    @contextlib.contextmanager
    def _clean_autocast_cache(self):
        """
        Drop autocast's weight cache around the inner generation.

        Under `accelerator.autocast()` the training forward runs inside an
        autocast region that caches each parameter's bf16 cast.  A cast
        performed inside `torch.inference_mode()` produces an INFERENCE tensor;
        if it lands in that cache, the outer grad-tracked forward reuses it and
        dies with "Inference tensors cannot be saved for backward" — every
        module shared by both passes (obs_encoder, model, *_proj) is affected.

        Clearing on both sides keeps the two passes from sharing cast weights.
        Outside autocast this is a no-op.
        """
        torch.clear_autocast_cache()
        try:
            yield
        finally:
            torch.clear_autocast_cache()

    def _generate_prev_past(self, batch) -> torch.Tensor:
        """
        Re-run the policy on the previous window and return the past buffer it
        would have left behind.

        Returns:
            (B, past_n, action_dim) raw (unnormalized) actions, detached
        """
        prev_past = batch["prev_past_action"]
        B = prev_past.shape[0]

        with self._rollout_mode(), self._clean_autocast_cache():
            with torch.inference_mode():
                prev_features = self.obs_encoder(batch["prev_obs"])
                prev_cond = self._build_condition(prev_features, prev_past)

                tokens = torch.full(
                    (B, 1), self.bos_id,
                    dtype=torch.long, device=prev_past.device,
                )
                tokens = self.model.generate(
                    tokens,
                    cond=prev_cond,
                    max_new_tokens=self.max_seq_len,
                    temperature=self.self_past_temperature,
                    top_k=self.self_past_topk,
                )[:, 1:]    # drop <BOS>
                tokens = tokens.clamp(0, self.bos_id - 1)

                prev_pred = self.action_tokenizer.detokenize(tokens=tokens)

                # Same buffer-update rule as predict_action.
                n_exec = self.n_action_steps
                past_n = self.past_n
                if n_exec >= past_n:
                    generated = prev_pred[:, n_exec - past_n: n_exec]
                else:
                    generated = torch.cat([
                        prev_past[:, n_exec:],
                        prev_pred[:, :n_exec],
                    ], dim=1)

        # Lift out of inference mode so the result can feed the autograd graph.
        with torch.inference_mode(False):
            generated = generated.detach().clone()

        return generated.to(dtype=prev_past.dtype)

    def _maybe_self_past(self, batch, past_actions: torch.Tensor) -> torch.Tensor:
        """
        Replace ground-truth `past_action` with the policy's own generated past,
        per sample with probability `self_past_p`, after warmup.
        """
        if self.self_past_p <= 0.0:
            return past_actions
        if self._train_step < self.self_past_warmup_steps:
            return past_actions
        if "prev_obs" not in batch:
            raise KeyError(
                "OATPolicyWithSelfPast needs 'prev_obs' / 'prev_past_action'; "
                "use oat.dataset.zarr_dataset_with_prev_window.ZarrDatasetWithPrevWindow"
            )

        generated = self._generate_prev_past(batch)

        if self.self_past_p >= 1.0:
            return generated

        use_self = (
            torch.rand(
                past_actions.shape[0], 1, 1, device=past_actions.device
            ) < self.self_past_p
        )
        return torch.where(use_self, generated, past_actions)

    # ── Training ────────────────────────────────────────────────────────────

    def forward(self, batch) -> torch.Tensor:
        # tokenize ground-truth actions (frozen tokenizer)
        with torch.no_grad():
            action_tokens = self.action_tokenizer.tokenize(batch["action"])

        B = batch["action"].shape[0]
        device = batch["action"].device

        # encode observation
        features = self.obs_encoder(batch["obs"])       # (B, To, d)

        # ── past actions: policy's own output instead of ground truth ─────
        past_actions = batch["past_action"]              # (B, past_n, action_dim)
        past_actions = self._maybe_self_past(batch, past_actions)

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

        # increment step counter (used by _maybe_self_past)
        self._train_step += 1

        return loss
