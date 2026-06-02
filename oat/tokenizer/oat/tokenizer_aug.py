"""OATTok with SO(2) direct data augmentation.

During training, random SO(2) rotations are applied to the (dx, dy) and
(droll, dpitch) action pairs. The tokenizer is then trained to reconstruct
these rotated actions, effectively multiplying dataset diversity.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Union, List, Optional

from oat.model.common.normalizer import LinearNormalizer
from oat.tokenizer.oat.tokenizer import OATTok
from oat.tokenizer.oat.encoder.register_encoder import RegisterEncoder
from oat.tokenizer.oat.decoder.single_pass_decoder import SinglePassDecoder
from oat.tokenizer.oat.quantizer.fsq import FSQ


class OATTokAug(OATTok):
    """OATTok with SO(2) rotation data augmentation.

    Args:
        encoder: RegisterEncoder for temporal compression.
        decoder: SinglePassDecoder for reconstruction.
        quantizer: FSQ quantizer.
        n_aug_samples: Number of random SO(2) rotations per batch during training.
            Set to 0 to disable augmentation (falls back to base OATTok).
        aug_weight: Weight for the augmentation loss relative to the original
            reconstruction loss.
    """

    def __init__(
        self,
        encoder: RegisterEncoder,
        decoder: SinglePassDecoder,
        quantizer: FSQ,
        n_aug_samples: int = 4,
        aug_weight: float = 1.0,
    ):
        super().__init__(encoder=encoder, decoder=decoder, quantizer=quantizer)
        self.n_aug_samples = n_aug_samples
        self.aug_weight = aug_weight

    @staticmethod
    def _rotate_actions(
        actions: torch.Tensor, cos_phi: torch.Tensor, sin_phi: torch.Tensor,
    ) -> torch.Tensor:
        """Apply SO(2) rotation to (dx, dy) and (droll, dpitch) doublets.

        Leaves dz (index 2), dyaw (index 5), and grip (index 6) unchanged.
        """
        rotated = actions.clone()
        # Rotate (dx, dy)
        rotated[..., 0] = cos_phi * actions[..., 0] - sin_phi * actions[..., 1]
        rotated[..., 1] = sin_phi * actions[..., 0] + cos_phi * actions[..., 1]
        # Rotate (droll, dpitch)
        rotated[..., 3] = cos_phi * actions[..., 3] - sin_phi * actions[..., 4]
        rotated[..., 4] = sin_phi * actions[..., 3] + cos_phi * actions[..., 4]
        return rotated

    def _aug_loss(self, nactions: torch.Tensor) -> torch.Tensor:
        """Compute reconstruction loss on SO(2)-rotated actions."""
        device = nactions.device
        loss = torch.tensor(0.0, device=device)

        for _ in range(self.n_aug_samples):
            phi = torch.rand(1, device=device) * 2 * torch.pi
            cos_phi, sin_phi = torch.cos(phi), torch.sin(phi)

            rotated = self._rotate_actions(nactions, cos_phi, sin_phi)
            latents = self.encoder(rotated)
            latents, _ = self.quantizer(latents)
            recons = self.decoder(latents)
            loss = loss + F.mse_loss(recons, rotated)

        return loss / self.n_aug_samples

    def forward(self, batch) -> torch.Tensor:
        samples = batch['action']

        # normalize
        nsamples = self.normalizer['action'].normalize(samples)

        # original reconstruction
        latents = self.encoder(nsamples)
        latents, _ = self.quantizer(latents)
        recons = self.decoder(latents)
        loss = F.mse_loss(recons, nsamples)

        # SO(2) augmentation (training only)
        if self.training and self.n_aug_samples > 0:
            loss_aug = self._aug_loss(nsamples)
            loss = loss + self.aug_weight * loss_aug

        return loss
