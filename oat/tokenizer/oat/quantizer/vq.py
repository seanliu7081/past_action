# --------------------------------------------------------------------
# Vanilla Vector Quantization (VQ) — ablation against FSQ / BSQ.
#
# Reference:
#   van den Oord et al., "Neural Discrete Representation Learning"
#   (VQ-VAE), NeurIPS 2017. arXiv:1711.00937
#   EMA codebook update follows VQ-VAE-2 (Razavi et al., 2019,
#   arXiv:1906.00446) and the original sonnet implementation
#   (deepmind/sonnet/src/nets/vqvae.py).
#
# Drop-in compatible with FSQ / BSQ at the call sites that only consume
# `(quant, tokens)`. Because vanilla VQ has an auxiliary commitment loss
# that must be added to the reconstruction loss, this module also
# exposes the last computed loss via `self.last_aux_loss`. A tokenizer
# subclass aware of this attribute is responsible for adding it to the
# training objective (see oat.tokenizer.oat.tokenizer_vq.OATTokVQ and
# oat.tokenizer.oat.tokenizer_so3_aug_vq.OATTokSO3AugVQ).
# --------------------------------------------------------------------

import math
from functools import partial
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.amp import autocast

from oat.tokenizer.oat.util.packed_ops import packed_call


__all__ = ["VQ"]


class VQ(nn.Module):
    """Vanilla VQ-VAE quantizer with EMA codebook updates.

    Interface matches FSQ at the call-site:
        forward(latents) -> (quant, tokens)
        indices_to_embedding(indices) -> embeddings
        codebook_size, dim, implicit_codebook

    The commitment loss (already scaled by `commitment_beta`) is stashed
    on `self.last_aux_loss` after every forward call. Tokenizers should
    add it to the reconstruction loss during training.

    Args:
        codebook_size: Number of discrete codes K.
        dim: Embedding dimension D of each code (== latent dim).
        commitment_beta: Weight of the encoder commitment loss term.
        ema_decay: Exponential moving average decay for codebook updates.
        ema_eps: Laplace smoothing epsilon for EMA cluster sizes.
        dead_code_threshold: Reinitialize codes whose EMA cluster size
            drops below this from random batch entries. Set to 0 to
            disable. Only active in training mode with `use_ema=True`.
        use_ema: When True, codebook is a non-trainable buffer updated
            via EMA. When False, codebook is an nn.Parameter trained
            with a codebook loss (original VQ-VAE formulation).
        packed_call: Use packed_call for list inputs (matches FSQ).
    """

    def __init__(
        self,
        codebook_size: int,
        dim: int,
        commitment_beta: float = 0.25,
        ema_decay: float = 0.99,
        ema_eps: float = 1.0e-5,
        dead_code_threshold: float = 1.0,
        use_ema: bool = True,
        packed_call: bool = True,
    ):
        super().__init__()

        self.codebook_size = int(codebook_size)
        self.dim = int(dim)
        self.commitment_beta = float(commitment_beta)
        self.ema_decay = float(ema_decay)
        self.ema_eps = float(ema_eps)
        self.dead_code_threshold = float(dead_code_threshold)
        self.use_ema = bool(use_ema)
        self.packed_call = bool(packed_call)

        # Codebook initialization: scale roughly to unit-variance encoder
        # outputs. 1/sqrt(D) keeps per-component magnitude on the same
        # order as a standard-normal input projected through a small MLP.
        codebook = torch.randn(self.codebook_size, self.dim) / math.sqrt(self.dim)

        if self.use_ema:
            self.register_buffer("codebook", codebook)
            self.register_buffer("ema_cluster_size", torch.zeros(self.codebook_size))
            self.register_buffer("ema_embed", codebook.clone())
        else:
            self.codebook = nn.Parameter(codebook)

        # Convenience alias used by FSQ-style callers (e.g. corrupt paths
        # in other quantizers). Same tensor as `codebook`.
        # Defined as a property below so it always tracks the live codebook.

        # last commitment loss (scaled by beta). Read by tokenizer.
        self.last_aux_loss: Optional[Tensor] = None

    @property
    def implicit_codebook(self) -> Tensor:
        return self.codebook

    def __repr__(self) -> str:
        return (
            f"VQ(\n"
            f"  codebook_size={self.codebook_size},\n"
            f"  dim={self.dim},\n"
            f"  commitment_beta={self.commitment_beta},\n"
            f"  ema_decay={self.ema_decay},\n"
            f"  use_ema={self.use_ema},\n"
            f"  dead_code_threshold={self.dead_code_threshold},\n"
            f")"
        )

    # ---------------------------------------------------------------
    # Core quantization
    # ---------------------------------------------------------------

    def _nearest_indices(self, z_flat: Tensor) -> Tensor:
        """Return [N] long indices of nearest codebook entry per row."""
        # ||z - e||^2 = ||z||^2 - 2 z·e + ||e||^2
        z_sq = z_flat.pow(2).sum(dim=-1, keepdim=True)            # [N, 1]
        e_sq = self.codebook.pow(2).sum(dim=-1).unsqueeze(0)      # [1, K]
        cross = z_flat @ self.codebook.t()                        # [N, K]
        dists = z_sq - 2.0 * cross + e_sq
        return dists.argmin(dim=-1)

    @torch.no_grad()
    def _ema_update(self, z_flat: Tensor, tokens: Tensor) -> None:
        """Update EMA cluster sizes / sums and refresh the codebook."""
        one_hot = F.one_hot(tokens, num_classes=self.codebook_size).type_as(z_flat)
        new_cluster_size = one_hot.sum(dim=0)                     # [K]
        new_embed_sum = one_hot.t() @ z_flat                      # [K, D]

        self.ema_cluster_size.mul_(self.ema_decay).add_(
            new_cluster_size, alpha=1.0 - self.ema_decay
        )
        self.ema_embed.mul_(self.ema_decay).add_(
            new_embed_sum, alpha=1.0 - self.ema_decay
        )

        # Laplace-smoothed cluster size; prevents division by zero
        # and pulls dead codes towards the codebook mean.
        n = self.ema_cluster_size.sum()
        cluster_size = (
            (self.ema_cluster_size + self.ema_eps)
            / (n + self.codebook_size * self.ema_eps)
            * n
        )
        self.codebook.copy_(self.ema_embed / cluster_size.unsqueeze(-1))

        # Optional dead-code reinit from random batch rows.
        if self.dead_code_threshold > 0.0:
            dead_mask = self.ema_cluster_size < self.dead_code_threshold
            n_dead = int(dead_mask.sum().item())
            if n_dead > 0 and z_flat.shape[0] > 0:
                rand_idx = torch.randint(
                    0, z_flat.shape[0], (n_dead,), device=z_flat.device
                )
                replacements = z_flat[rand_idx].detach()
                self.codebook[dead_mask] = replacements
                self.ema_embed[dead_mask] = replacements
                self.ema_cluster_size[dead_mask] = 1.0

    @autocast(device_type="cuda", enabled=False)
    def forward_z(self, z: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Quantize a single tensor input.

        Args:
            z: [..., D] continuous latent vectors.
        Returns:
            quant: [..., D] quantized vectors with STE gradient.
            tokens: [...] integer (long) token indices.
            aux_loss: scalar tensor — commitment loss already scaled by
                `commitment_beta`. When use_ema=False, also includes the
                codebook loss term.
        """
        assert z.shape[-1] == self.dim, (
            f"expected last dim {self.dim}, got {z.shape[-1]}"
        )
        z = z.float()
        orig_shape = z.shape
        z_flat = z.reshape(-1, self.dim)

        tokens = self._nearest_indices(z_flat)
        e = self.codebook[tokens]                                 # [N, D]

        # Auxiliary loss: commitment always; codebook loss only when
        # the codebook is trained via gradient (use_ema=False).
        commitment_loss = F.mse_loss(z_flat, e.detach())
        aux_loss = self.commitment_beta * commitment_loss
        if not self.use_ema:
            codebook_loss = F.mse_loss(e, z_flat.detach())
            aux_loss = aux_loss + codebook_loss

        # Straight-through estimator: forward = e, backward passes
        # gradient through to z.
        quant_flat = z_flat + (e - z_flat).detach()

        if self.training and self.use_ema:
            self._ema_update(z_flat, tokens)

        quant = quant_flat.reshape(orig_shape)
        tokens = tokens.reshape(orig_shape[:-1]).long()
        return quant, tokens, aux_loss

    @torch.compiler.disable
    def forward(self, latents) -> Tuple[Tensor, Tensor]:
        """Quantize latents. Returns (quant, tokens); commitment loss is
        accessible afterwards via `self.last_aux_loss`."""
        if self.packed_call:
            quant, tokens, aux = packed_call(partial(self.forward_z), latents)
            # packed_call leaves scalar aux untouched whether `latents`
            # is a list or a single tensor.
        elif isinstance(latents, list):
            quant, tokens, aux_list = [], [], []
            for z_i in latents:
                q_i, t_i, a_i = self.forward_z(z_i)
                quant.append(q_i)
                tokens.append(t_i)
                aux_list.append(a_i)
            aux = torch.stack(aux_list).mean()
        else:
            quant, tokens, aux = self.forward_z(latents)

        self.last_aux_loss = aux
        return quant, tokens

    # ---------------------------------------------------------------
    # Index <-> embedding
    # ---------------------------------------------------------------

    def indices_to_embedding(self, indices: Tensor) -> Tensor:
        return self.codebook[indices.long()]
