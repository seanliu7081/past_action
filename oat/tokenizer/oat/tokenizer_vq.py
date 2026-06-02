import torch
import torch.nn.functional as F

from oat.tokenizer.oat.tokenizer import OATTok


class OATTokVQ(OATTok):
    """OATTok variant that adds the vanilla-VQ commitment loss to the
    reconstruction objective. The quantizer is expected to expose its
    last commitment loss via `self.quantizer.last_aux_loss` (see
    oat.tokenizer.oat.quantizer.vq.VQ)."""

    def forward(self, batch) -> torch.Tensor:
        samples = batch["action"]
        nsamples = self.normalizer["action"].normalize(samples)
        latents = self.encoder(nsamples)
        latents, _ = self.quantizer(latents)
        recons = self.decoder(latents)
        loss = F.mse_loss(recons, nsamples)

        aux = getattr(self.quantizer, "last_aux_loss", None)
        if aux is not None:
            loss = loss + aux
        return loss
