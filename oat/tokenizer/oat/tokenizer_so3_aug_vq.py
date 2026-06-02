import torch
import torch.nn.functional as F

from oat.tokenizer.oat.tokenizer_so3_aug import OATTokSO3Aug


class OATTokSO3AugVQ(OATTokSO3Aug):
    """SO(3)-augmented OATTok variant that also adds the vanilla-VQ
    commitment loss to the reconstruction objective. The quantizer is
    expected to expose its last commitment loss via
    `self.quantizer.last_aux_loss` (see
    oat.tokenizer.oat.quantizer.vq.VQ)."""

    def forward(self, batch) -> torch.Tensor:
        samples = batch["action"]

        if self.action_aug is not None:
            samples = self.action_aug(samples)

        nsamples = self.normalizer["action"].normalize(samples)
        latents = self.encoder(nsamples)
        latents, _ = self.quantizer(latents)
        recons = self.decoder(latents)
        loss = F.mse_loss(recons, nsamples)

        aux = getattr(self.quantizer, "last_aux_loss", None)
        if aux is not None:
            loss = loss + aux
        return loss
