import torch

from oat.tokenizer.bin.tokenizer import BinTok


class BinTokSO3Aug(BinTok):
    """BinTok variant that applies raw-action SO(3) augmentation for evaluation.

    The bin tokenizer is parameter-free, so there is nothing to train. The
    augmentation is instead applied to the raw evaluation actions so that the
    reported reconstruction MSE measures how the fixed bins handle SO(3)-rotated
    action chunks, comparable to the augmented OATTok eval.
    """

    def __init__(self,
        num_bins: int = 256,
        min_val: float = -1.0,
        max_val: float = 1.0,
        action_aug=None,
    ):
        super().__init__(num_bins=num_bins, min_val=min_val, max_val=max_val)
        self.action_aug = action_aug

    def augment(self, samples: torch.Tensor) -> torch.Tensor:
        if self.action_aug is None:
            return samples
        # SO3ActionChunkAug is a no-op outside train mode; force it on for eval.
        was_training = self.action_aug.training
        self.action_aug.train()
        try:
            return self.action_aug(samples)
        finally:
            self.action_aug.train(was_training)
