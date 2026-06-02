"""
OAT policy with Gaussian soft-label loss.

A drop-in replacement for `oat.policy.oatpolicy.OATPolicy` that swaps the
one-hot cross-entropy loss for a Gaussian-smoothed soft-label loss over the
action codebook (optionally plus an EMD^2 regularizer). Everything else —
observation encoding, autoregressive transformer, token sampling at inference
time, evaluation rollouts — is inherited unchanged.

Because `OATPolicy.forward` returns a scalar loss used directly by
`TrainPolicyWorkspace`, we override only that method. The codebook is pulled
from the tokenizer attached to the policy (`self.action_tokenizer.quantizer.
implicit_codebook`), so no changes to the training workspace are needed.
"""

from typing import Dict

import torch
import torch.nn.functional as F

from oat.policy.oatpolicy import OATPolicy
from oat.policy.loss.gaussian_soft_label_loss import (
    GaussianSoftLabelLoss,
    GaussianSoftCEWithEMD,
)


class GaussianSoftLabelPolicy(OATPolicy):
    """OATPolicy variant that replaces CE loss with Gaussian soft-label loss.

    Extra arguments:
        sigma: Gaussian temperature. Small sigma ~ CE; larger sigma puts more
            target mass on codebook neighbors.
        use_index_distance: If True, distance is |i - j| over token indices
            instead of codebook L2. Meaningful only when the codebook is
            locality-preserving (e.g. Hilbert-ordered like ZHill). In that
            case sigma should be much larger (indices run 0..K-1).
        lambda_emd: Weight on the per-dimension EMD^2 regularizer. 0 disables.
    """

    def __init__(
        self,
        *args,
        sigma: float = 1.0,
        use_index_distance: bool = False,
        lambda_emd: float = 0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.sigma = float(sigma)
        self.use_index_distance = bool(use_index_distance)
        self.lambda_emd = float(lambda_emd)

        codebook = self.action_tokenizer.quantizer.implicit_codebook.detach()
        if self.lambda_emd > 0.0:
            levels = self.action_tokenizer.quantizer._levels
            self.criterion = GaussianSoftCEWithEMD(
                codebook=codebook,
                levels=levels,
                sigma=self.sigma,
                use_index_distance=self.use_index_distance,
                lambda_emd=self.lambda_emd,
            )
        else:
            self.criterion = GaussianSoftLabelLoss(
                codebook=codebook,
                sigma=self.sigma,
                use_index_distance=self.use_index_distance,
            )

        print(
            f"  GaussianSoftLabelPolicy: sigma={self.sigma} "
            f"use_index_distance={self.use_index_distance} "
            f"lambda_emd={self.lambda_emd} "
            f"codebook_size={codebook.shape[0]} codebook_dim={codebook.shape[1]}"
        )

    def get_policy_name(self) -> str:
        base_name = 'gslpolicy_'
        for modality in self.modalities:
            if modality != 'state':
                base_name += modality + '|'
        return base_name[:-1]

    def forward(self, batch) -> torch.Tensor:
        # tokenize trajectory
        with torch.inference_mode():
            action_tokens = self.action_tokenizer.tokenize(batch['action'])

        B = batch['action'].shape[0]
        device = batch['action'].device

        # encode observation
        features = self.obs_encoder(batch['obs'])   # [B, To, d]

        # prepend <BOS>
        action_tokens = torch.cat([
            torch.full((B, 1), self.bos_id, dtype=torch.long, device=device),
            action_tokens,
        ], dim=1)

        logits = self.model(action_tokens[:, :-1], cond=features)  # (B, T, V=K+1)
        targets = action_tokens[:, 1:]                             # (B, T) in [0, K)

        out = self.criterion(logits, targets)
        if isinstance(out, dict):
            loss = out["loss"]
            self._last_loss_components = {
                k: v.detach() for k, v in out.items() if torch.is_tensor(v)
            }
        else:
            loss = out
            self._last_loss_components = {"gaussian_loss": loss.detach()}

        # Also track standard CE for comparison (no gradient contribution).
        with torch.no_grad():
            V = logits.size(-1)
            ce_loss = F.cross_entropy(
                logits.reshape(-1, V).float(),
                targets.reshape(-1),
            )
            self._last_loss_components["ce_loss"] = ce_loss.detach()

        return loss

    def get_last_loss_components(self) -> Dict[str, torch.Tensor]:
        """Return the most recent loss components for logging (gaussian, emd2, ce)."""
        return getattr(self, "_last_loss_components", {})
