# --------------------------------------------------------------------
# A2 hexagonal lattice FSQ quantizer for A2Lex tokenizer.
# Replaces the cubic Z^d quantization grid in FSQ with A2^{d/2}
# hexagonal product lattice for lower quantization distortion.
# Lexicographic ordering on lattice coordinates.
# --------------------------------------------------------------------

import math
import random
from functools import partial

import torch
import torch.nn as nn
from einops import repeat
from torch import Tensor, int32
from torch.amp import autocast
from typing import List, Optional, Tuple

from oat.tokenizer.oat.util.packed_ops import packed_call


__all__ = ["A2LexFSQ"]


SQRT3 = math.sqrt(3)
SQRT3_OVER_2 = SQRT3 / 2  # ~0.8660
INV_SQRT3 = 1.0 / SQRT3   # ~0.5774


def round_ste(z: Tensor) -> Tensor:
    """Round with straight through gradients."""
    zhat = z.round()
    return z + (zhat - z).detach()


class A2LexFSQ(nn.Module):
    """FSQ with A2 hexagonal product lattice quantization and lexicographic ordering.

    Instead of quantizing each dimension independently on a cubic grid (Z^d),
    dimensions are paired and each pair is quantized to the A2 hexagonal lattice,
    which has provably lower quantization distortion. The embedding passed to the
    decoder is in normalized Cartesian coordinates of the lattice points.

    Args:
        levels: List of FSQ levels per dimension.
        drop_quant_p: During training, skip quantization with this probability per sample.
        corrupt_tokens_p: During training, corrupt this percentage of tokens to random indices.
        min_corrupt_tokens_p: Minimum corruption percentage (sampled uniformly with max).
        apply_corrupt_tokens_p: Probability of activating token corruption per sample.
        packed_call: Pack list of examples and quantize jointly.
    """

    def __init__(
        self,
        levels: List[int],
        drop_quant_p: float = 0.0,
        corrupt_tokens_p: float = 0.0,
        min_corrupt_tokens_p: Optional[float] = None,
        apply_corrupt_tokens_p: float = 0.2,
        packed_call: bool = True,
    ):
        super().__init__()

        _levels = torch.tensor(levels, dtype=int32)
        self.register_buffer("_levels", _levels, persistent=False)

        self.dim = len(levels)
        self.num_pairs = self.dim // 2
        self.has_remainder = (self.dim % 2 == 1)
        self.codebook_size = _levels.prod().item()

        # Pair-level information
        pair_sizes = []
        for i in range(self.num_pairs):
            pair_sizes.append(levels[2 * i] * levels[2 * i + 1])
        if self.has_remainder:
            pair_sizes.append(levels[-1])
        self.register_buffer(
            "_pair_sizes", torch.tensor(pair_sizes, dtype=int32), persistent=False
        )

        # Cross-pair mixed-radix basis
        _pair_basis = torch.cumprod(
            torch.tensor([1] + pair_sizes[:-1], dtype=int32), dim=0
        )
        self.register_buffer("_pair_basis", _pair_basis, persistent=False)

        # Cartesian normalization constants
        cart_scale = torch.zeros(self.dim)
        cart_offset = torch.zeros(self.dim)

        for i in range(self.num_pairs):
            L0, L1 = levels[2 * i], levels[2 * i + 1]
            x_max = (L0 - 1) + (L1 - 1) * 0.5
            y_max = (L1 - 1) * SQRT3_OVER_2

            if x_max > 0:
                cart_scale[2 * i] = 2.0 / x_max
                cart_offset[2 * i] = -1.0
            else:
                cart_scale[2 * i] = 0.0
                cart_offset[2 * i] = 0.0

            if y_max > 0:
                cart_scale[2 * i + 1] = 2.0 / y_max
                cart_offset[2 * i + 1] = -1.0
            else:
                cart_scale[2 * i + 1] = 0.0
                cart_offset[2 * i + 1] = 0.0

        if self.has_remainder:
            L_last = levels[-1]
            if L_last > 1:
                cart_scale[-1] = 2.0 / (L_last - 1)
                cart_offset[-1] = -1.0
            else:
                cart_scale[-1] = 0.0
                cart_offset[-1] = 0.0

        self.register_buffer("_cart_scale", cart_scale, persistent=False)
        self.register_buffer("_cart_offset", cart_offset, persistent=False)

        # Build implicit codebook
        implicit_codebook = self.indices_to_embedding(torch.arange(self.codebook_size))
        self.register_buffer("implicit_codebook", implicit_codebook, persistent=False)

        self.drop_quant_p = drop_quant_p
        self.corrupt_tokens_p = corrupt_tokens_p
        self.min_corrupt_tokens_p = min_corrupt_tokens_p or corrupt_tokens_p
        self.apply_corrupt_tokens_p = apply_corrupt_tokens_p
        self.packed_call = packed_call

    def __repr__(self):
        cls_name = self.__class__.__name__
        return (
            f"{cls_name}(\n"
            f"  levels={self._levels.tolist()!r},\n"
            f"  codebook_size={self.codebook_size!r},\n"
            f"  drop_quant_p={self.drop_quant_p!r},\n"
            f"  lattice='A2^{self.num_pairs}',\n"
            ")"
        )

    # ------------------------------------------------------------------
    # Bounding (identical to FSQ)
    # ------------------------------------------------------------------

    def bound(self, z: Tensor, eps: float = 1e-3) -> Tensor:
        """Bound `z`, an array of shape (..., d)."""
        half_l = (self._levels - 1) * (1 + eps) / 2
        offset = torch.where(self._levels % 2 == 0, 0.5, 0.0)
        shift = (offset / half_l).atanh()
        return (z + shift).tanh() * half_l - offset

    # ------------------------------------------------------------------
    # A2 hex rounding
    # ------------------------------------------------------------------

    def _hex_round_pair(self, u: Tensor, v: Tensor) -> Tuple[Tensor, Tensor]:
        """Nearest A2 lattice point via 3-candidate selection.

        The A2 metric in lattice coordinates is:
            d^2(du, dv) = du^2 + du*dv + dv^2

        This corresponds to Euclidean distance in Cartesian space where
        the lattice has basis e1=(1,0), e2=(1/2, sqrt3/2).
        """
        u0 = torch.floor(u)
        v0 = torch.floor(v)
        fu = u - u0
        fv = v - v0

        lower_triangle = (fu + fv) < 1.0

        ca_u = torch.where(lower_triangle, u0, u0 + 1)
        ca_v = torch.where(lower_triangle, v0, v0 + 1)
        cb_u, cb_v = u0 + 1, v0
        cc_u, cc_v = u0, v0 + 1

        def a2_dist_sq(du, dv):
            return du * du + du * dv + dv * dv

        da = a2_dist_sq(u - ca_u, v - ca_v)
        db = a2_dist_sq(u - cb_u, v - cb_v)
        dc = a2_dist_sq(u - cc_u, v - cc_v)

        best_a = (da <= db) & (da <= dc)
        best_b = (~best_a) & (db <= dc)

        result_u = torch.where(best_a, ca_u, torch.where(best_b, cb_u, cc_u))
        result_v = torch.where(best_a, ca_v, torch.where(best_b, cb_v, cc_v))

        return result_u, result_v

    def _hex_round_all_pairs(self, bounded: Tensor) -> Tensor:
        """Non-differentiable hex rounding. Returns integer lattice coords (detached)."""
        parts = []
        for i in range(self.num_pairs):
            u_r, v_r = self._hex_round_pair(bounded[..., 2 * i], bounded[..., 2 * i + 1])
            parts.extend([u_r.unsqueeze(-1), v_r.unsqueeze(-1)])
        if self.has_remainder:
            parts.append(bounded[..., -1:].round())
        return torch.cat(parts, dim=-1)

    def _hex_round_ste_with_dropout(self, bounded: Tensor) -> Tensor:
        """Hex-round with STE gradient and optional per-sample dropout."""
        drop_p = self.drop_quant_p if self.training else 0.0

        hex_rounded = self._hex_round_all_pairs(bounded)

        if drop_p == 0.0:
            return bounded + (hex_rounded - bounded).detach()
        else:
            batch_size = bounded.shape[0]
            mask = torch.bernoulli(torch.full((batch_size,), drop_p, device=bounded.device))
            mask = mask.view(batch_size, *([1] * (bounded.ndim - 1)))
            return bounded + ((1 - mask) * (hex_rounded - bounded)).detach()

    # ------------------------------------------------------------------
    # Coordinate conversions
    # ------------------------------------------------------------------

    def _centered_to_nonneg(self, centered: Tensor) -> Tensor:
        """Convert centered lattice coords to 0-based [0, L-1]."""
        half_width = self._levels // 2
        return centered + half_width.float()

    def _nonneg_to_centered(self, nonneg: Tensor) -> Tensor:
        """Convert 0-based [0, L-1] back to centered coords."""
        half_width = self._levels // 2
        return nonneg - half_width.float()

    def _to_cartesian(self, nonneg: Tensor) -> Tensor:
        """Convert 0-based integer lattice coords to Cartesian coordinates, per pair."""
        parts = []
        for i in range(self.num_pairs):
            a = nonneg[..., 2 * i]
            b = nonneg[..., 2 * i + 1]
            x = a + b * 0.5
            y = b * SQRT3_OVER_2
            parts.extend([x.unsqueeze(-1), y.unsqueeze(-1)])
        if self.has_remainder:
            parts.append(nonneg[..., -1:])
        return torch.cat(parts, dim=-1)

    def _cartesian_to_lattice_nonneg(self, cartesian: Tensor) -> Tensor:
        """Inverse of _to_cartesian."""
        parts = []
        for i in range(self.num_pairs):
            x = cartesian[..., 2 * i]
            y = cartesian[..., 2 * i + 1]
            b = y / SQRT3_OVER_2
            a = x - b * 0.5
            parts.extend([a.round().unsqueeze(-1), b.round().unsqueeze(-1)])
        if self.has_remainder:
            parts.append(cartesian[..., -1:].round())
        return torch.cat(parts, dim=-1)

    def _normalize_cartesian(self, cartesian: Tensor) -> Tensor:
        """Map raw Cartesian [0, max] to normalized [-1, 1] per dimension."""
        return cartesian * self._cart_scale + self._cart_offset

    def _denormalize_cartesian(self, normalized: Tensor) -> Tensor:
        """Map normalized [-1, 1] back to raw Cartesian [0, max]."""
        safe_scale = self._cart_scale.clamp(min=1e-8)
        return (normalized - self._cart_offset) / safe_scale

    # ------------------------------------------------------------------
    # Index encoding / decoding (lexicographic across pairs)
    # ------------------------------------------------------------------

    def _nonneg_to_indices(self, nonneg: Tensor) -> Tensor:
        """Convert 0-based lattice coords to flat token indices."""
        nonneg_long = nonneg.long()

        index = torch.zeros(nonneg.shape[:-1], dtype=torch.long, device=nonneg.device)
        for i in range(self.num_pairs):
            a = nonneg_long[..., 2 * i]
            b = nonneg_long[..., 2 * i + 1]
            L0 = self._levels[2 * i].item()
            pair_idx = a + b * L0
            index = index + pair_idx * self._pair_basis[i]

        if self.has_remainder:
            index = index + nonneg_long[..., -1] * self._pair_basis[-1]

        return index

    def _indices_to_nonneg(self, indices: Tensor) -> Tensor:
        """Convert flat token indices back to 0-based lattice coords."""
        coords = []

        for i in range(self.num_pairs):
            pair_size = self._pair_sizes[i].item()
            pair_idx = (indices // self._pair_basis[i].item()) % pair_size
            L0 = self._levels[2 * i].item()
            a = pair_idx % L0
            b = pair_idx // L0
            coords.extend([a.unsqueeze(-1).float(), b.unsqueeze(-1).float()])

        if self.has_remainder:
            r_idx = (indices // self._pair_basis[-1].item()) % self._levels[-1].item()
            coords.append(r_idx.unsqueeze(-1).float())

        return torch.cat(coords, dim=-1)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def codes_to_indices(self, quant: Tensor) -> Tensor:
        """Convert normalized Cartesian embedding to token indices."""
        cartesian = self._denormalize_cartesian(quant)
        nonneg = self._cartesian_to_lattice_nonneg(cartesian)
        upper = (self._levels - 1).float()
        nonneg = torch.max(nonneg, torch.zeros_like(upper))
        nonneg = torch.min(nonneg, upper)
        return self._nonneg_to_indices(nonneg).to(int32)

    def indices_to_embedding(self, indices: Tensor) -> Tensor:
        """Convert token indices to normalized Cartesian embedding."""
        nonneg = self._indices_to_nonneg(indices)
        cartesian = self._to_cartesian(nonneg)
        return self._normalize_cartesian(cartesian)

    def corrupt_quant(self, quant: Tensor) -> Tensor:
        """Randomly corrupt some entries of the quantized Tensor."""
        quant_shape, quant_device = quant.shape[:-1], quant.device
        random_indices = torch.randint(
            low=0, high=self.codebook_size, size=quant_shape, device=quant_device
        )
        random_quant = self.implicit_codebook[random_indices]
        sample_corrupt_tokens_p = random.uniform(self.min_corrupt_tokens_p, self.corrupt_tokens_p)
        corruption_mask = torch.rand(quant_shape, device=quant_device) < sample_corrupt_tokens_p
        corruption_mask = repeat(corruption_mask, "... -> ... d", d=quant.shape[-1])
        return torch.where(corruption_mask, random_quant, quant)

    @autocast(device_type="cuda", enabled=False)
    def forward_z(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.LongTensor]:
        assert (
            z.shape[-1] == self.dim
        ), f"expected dimension of {self.dim} but found dimension of {z.shape[-1]}"

        z = z.float()

        # 1. Bound in lattice coords (identical to FSQ)
        bounded = self.bound(z)

        # 2. Hex-round with STE + dropout
        rounded = self._hex_round_ste_with_dropout(bounded)

        # 3. Convert centered -> 0-based lattice coords
        nonneg = self._centered_to_nonneg(rounded)

        # 4. Clamp to valid range (safety net, out-of-place for autograd)
        upper = (self._levels - 1).float()
        nonneg = torch.max(nonneg, torch.zeros_like(upper))
        nonneg = torch.min(nonneg, upper)

        # 5. Convert to Cartesian
        cartesian = self._to_cartesian(nonneg)

        # 6. Normalize to [-1, 1]
        quant = self._normalize_cartesian(cartesian)

        # 7. Token index from 0-based lattice coords
        tokens = self._nonneg_to_indices(nonneg)

        # 8. Optional token corruption
        if (
            self.training
            and self.corrupt_tokens_p > 0.0
            and random.random() < self.apply_corrupt_tokens_p
        ):
            quant = self.corrupt_quant(quant)

        return quant, tokens.long()  # type: ignore

    @torch.compiler.disable
    def forward(self, latents: torch.Tensor) -> Tuple[torch.Tensor, torch.LongTensor]:
        if self.packed_call:
            fsq_fn = partial(self.forward_z)
            quant, tokens = packed_call(fsq_fn, latents)
        elif isinstance(latents, list):
            quant, tokens = [], []
            for z_i in latents:
                quant_i, tokens_i = self.forward_z(z_i)
                quant.append(quant_i)
                tokens.append(tokens_i)
        else:
            quant, tokens = self.forward_z(latents)

        return quant, tokens
