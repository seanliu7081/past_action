import torch
import torch.nn.functional as F

from oat.tokenizer.oat.tokenizer import OATTok


class OATTokSO3Aug(OATTok):
    """OATTok variant that applies raw-action SO(3) augmentation in training forward."""

    def __init__(self, encoder, decoder, quantizer, action_aug=None):
        super().__init__(encoder=encoder, decoder=decoder, quantizer=quantizer)
        self.action_aug = action_aug

    def forward(self, batch) -> torch.Tensor:
        samples = batch["action"]

        if self.action_aug is not None:
            samples = self.action_aug(samples)

        nsamples = self.normalizer["action"].normalize(samples)
        latents = self.encoder(nsamples)
        latents, _ = self.quantizer(latents)
        recons = self.decoder(latents)
        loss = F.mse_loss(recons, nsamples)
        return loss
