"""
Gaussian soft label loss for OAT autoregressive policy training.

Standard cross-entropy is blind to codebook geometry: predicting a token whose
codeword is near the ground truth in action space is penalized identically to
predicting one on the opposite side of the action manifold. This module replaces
one-hot targets with Gaussian-smoothed soft labels over the codebook:

    q(k | v*) = softmax_k( -d(k, v*)^2 / (2 * sigma^2) )

where d(k, v*) is either the L2 distance between codeword embeddings or the
absolute index distance |k - v*| when the codebook is ordered (e.g. Hilbert).

Small sigma collapses q back to one-hot and recovers standard CE. Large sigma
spreads mass uniformly. This is the HL-Gauss approach (Farebrother et al.,
ICML 2024) applied to autoregressive action tokens, closely related to SORD
(Diaz & Marathe, CVPR 2019) for ordinal regression.

The loss is a drop-in replacement for F.cross_entropy in the OAT policy:

    codebook = tokenizer.quantizer.implicit_codebook     # (K, D)
    criterion = GaussianSoftLabelLoss(codebook=codebook, sigma=1.0)
    logits = policy(observations, past_tokens)           # (B, T, V) with V = K or K+1
    targets = tokenizer.tokenize(actions)                # (B, T) in [0, K)
    loss = criterion(logits, targets)

If logits have V > K (e.g. K+1 because of a BOS token appended to the vocab),
the soft target is zero-padded to width V, so the extra logits receive no
target mass but are still normalized into the softmax (matching standard CE).
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def build_codebook_distance_matrix(codebook: torch.Tensor) -> torch.Tensor:
    """Precompute pairwise L2 distances between codebook entries.

    Args:
        codebook: (K, D) tensor of codeword embeddings.
    Returns:
        (K, K) tensor where entry [i, j] = ||c_i - c_j||_2.
    """
    assert codebook.ndim == 2, f"codebook must be (K, D), got {tuple(codebook.shape)}"
    return torch.cdist(codebook, codebook, p=2)


def build_index_distance_matrix(K: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """Precompute |i - j| for all index pairs."""
    idx = torch.arange(K, device=device, dtype=dtype)
    return (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()


def gaussian_soft_targets(distances: torch.Tensor, sigma: float) -> torch.Tensor:
    """Convert per-sample distances to Gaussian soft probability distributions.

    Uses the numerically-stable log-sum-exp form of
        q(k) = softmax_k( -distances[..., k]^2 / (2 * sigma^2) ).

    Args:
        distances: (..., K) tensor of non-negative distances to each codebook
            entry (or to all pairs, broadcast-compatible).
        sigma: Gaussian temperature; smaller -> more peaked, larger -> more uniform.
    Returns:
        (..., K) probability distribution (sums to 1 over last dim).
    """
    assert sigma > 0, f"sigma must be positive, got {sigma}"
    return F.softmax(-(distances.pow(2)) / (2.0 * sigma * sigma), dim=-1)


def _infer_per_dim_indices(codebook: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
    """Infer integer per-dim indices from a (K, D) codebook and per-dim level counts.

    Works generically for any product-FSQ-family quantizer (FSQ, A2Lex, ZHill):
    each scalar dimension of the codebook takes only `levels[d]` unique values,
    and their sorted rank gives a stable integer index in [0, levels[d]).

    Returns:
        (K, D) long tensor of per-dim indices.
    """
    K, D = codebook.shape
    levels = levels.to(codebook.device)
    per_dim = torch.zeros(K, D, dtype=torch.long, device=codebook.device)
    for d in range(D):
        col = codebook[:, d].contiguous()
        uniq = torch.unique(col)
        assert uniq.numel() == int(levels[d].item()), (
            f"Expected {int(levels[d].item())} unique values along dim {d}, "
            f"got {uniq.numel()}. Codebook/levels mismatch."
        )
        per_dim[:, d] = torch.searchsorted(uniq, col)
    return per_dim


# ---------------------------------------------------------------------------
# Gaussian soft label cross-entropy
# ---------------------------------------------------------------------------


class GaussianSoftLabelLoss(nn.Module):
    """Cross-entropy against Gaussian-smoothed soft labels over a discrete codebook.

    Args:
        codebook: (K, D) codeword embeddings. Obtained from
            `tokenizer.quantizer.implicit_codebook`.
        sigma: Gaussian temperature in the space where distance is measured.
            For L2 distance on a normalized codebook (values roughly in [-1, 1]),
            sigma ~ 0.5-2.0 is a reasonable sweep range. For index distance
            (integers in [0, K)), sigma must be much larger (e.g. 10-100).
        use_index_distance: If True, use |i - j| instead of ||c_i - c_j||_2.
            Useful for locality-preserving codebooks (e.g. Hilbert ordering).
        ignore_index: Target indices equal to this value are skipped in the
            loss (mirrors F.cross_entropy's ignore_index; default -100 = no skip).

    Inputs:
        logits: (B, T, V) raw logits from the policy head. V may be equal to K
            (plain codebook softmax) or K + delta (e.g. +1 for a BOS token in
            the output vocab). Any extra vocab entries beyond K receive zero
            target mass.
        targets: (B, T) ground-truth codebook indices in [0, K).

    Returns:
        Scalar loss (mean over all non-ignored positions).
    """

    def __init__(
        self,
        codebook: torch.Tensor,
        sigma: float = 1.0,
        use_index_distance: bool = False,
        ignore_index: int = -100,
    ):
        super().__init__()
        assert codebook.ndim == 2, f"codebook must be (K, D), got {tuple(codebook.shape)}"
        assert sigma > 0, f"sigma must be positive, got {sigma}"

        K = codebook.shape[0]
        self.codebook_size = K
        self.sigma = float(sigma)
        self.use_index_distance = bool(use_index_distance)
        self.ignore_index = int(ignore_index)

        if use_index_distance:
            dist = build_index_distance_matrix(K, device=codebook.device, dtype=codebook.dtype)
        else:
            dist = build_codebook_distance_matrix(codebook.float()).to(codebook.dtype)

        # Precompute the full (K, K) soft-target matrix once. For K=1000, this is
        # 4 MB in fp32 — negligible and saves a softmax every forward pass.
        soft_targets = gaussian_soft_targets(dist, self.sigma)  # (K, K)

        self.register_buffer("soft_targets", soft_targets, persistent=False)
        self.register_buffer("codebook", codebook.detach().clone(), persistent=False)

    def extra_repr(self) -> str:
        return (
            f"codebook_size={self.codebook_size}, sigma={self.sigma}, "
            f"use_index_distance={self.use_index_distance}"
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        assert logits.ndim == 3, f"logits must be (B, T, V), got {tuple(logits.shape)}"
        assert targets.ndim == 2, f"targets must be (B, T), got {tuple(targets.shape)}"
        B, T, V = logits.shape
        K = self.codebook_size
        assert V >= K, f"logits vocab {V} must be >= codebook size {K}"
        assert targets.shape == (B, T), "targets shape mismatch"

        flat_logits = logits.reshape(B * T, V)
        flat_targets = targets.reshape(B * T)

        valid_mask = flat_targets != self.ignore_index
        safe_targets = flat_targets.clamp(min=0, max=K - 1)

        # (N, K) soft target rows looked up from the (K, K) precomputed table.
        q = self.soft_targets.index_select(0, safe_targets)  # (N, K)

        # Zero-pad to full vocab width (extra entries like BOS get zero mass but
        # still participate in softmax normalization).
        if V > K:
            pad = q.new_zeros(q.shape[0], V - K)
            q = torch.cat([q, pad], dim=-1)

        log_probs = F.log_softmax(flat_logits, dim=-1)  # (N, V)
        per_pos_loss = -(q * log_probs).sum(dim=-1)     # (N,)

        if valid_mask.all():
            return per_pos_loss.mean()
        per_pos_loss = per_pos_loss * valid_mask.float()
        denom = valid_mask.float().sum().clamp(min=1.0)
        return per_pos_loss.sum() / denom


# ---------------------------------------------------------------------------
# Factored EMD^2 regularizer
# ---------------------------------------------------------------------------


class GaussianSoftCEWithEMD(nn.Module):
    """GaussianSoftLabelLoss + optional EMD^2 regularizer over factored dims.

    EMD^2 (1-D Wasserstein-2, squared) is computed independently per scalar
    quantization dimension then averaged:

        L_emd = mean_d  sum_k ( CDF_pred_d(k) - CDF_target_d(k) )^2

    The per-dim predicted distribution is obtained by marginalizing the joint
    K-way softmax over the product codebook, which only requires a
    (K, D) precomputed per-dim index map. This works for any FSQ-family
    quantizer (FSQ, A2Lex, ZHill) given its `_levels` buffer.

        L_total = L_gauss + lambda_emd * L_emd

    Args:
        codebook: (K, D) codeword embeddings.
        levels: (D,) int tensor with per-dim level counts (e.g. FSQ's
            `quantizer._levels`). Required to factor out per-dim distributions.
        sigma: Gaussian soft-label temperature.
        use_index_distance: See GaussianSoftLabelLoss.
        lambda_emd: Weight on the EMD^2 term. 0 disables the regularizer.
        ignore_index: See GaussianSoftLabelLoss.
    """

    def __init__(
        self,
        codebook: torch.Tensor,
        levels: torch.Tensor,
        sigma: float = 1.0,
        use_index_distance: bool = False,
        lambda_emd: float = 0.0,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.gauss = GaussianSoftLabelLoss(
            codebook=codebook,
            sigma=sigma,
            use_index_distance=use_index_distance,
            ignore_index=ignore_index,
        )
        self.lambda_emd = float(lambda_emd)
        self.ignore_index = int(ignore_index)

        levels = levels.to(torch.long).cpu()
        self.register_buffer("levels", levels, persistent=False)
        K = codebook.shape[0]
        assert int(levels.prod().item()) == K, (
            f"prod(levels)={int(levels.prod().item())} must equal codebook_size={K}"
        )

        per_dim = _infer_per_dim_indices(codebook, levels)  # (K, D)
        self.register_buffer("per_dim_indices", per_dim, persistent=False)

        # Per-dim one-hot scatter rows: a (K, L_d) indicator that maps a joint
        # token index to a one-hot over its per-dim index. Precomputed as
        # sparse-friendly dense tensors for easy einsum-style marginalization.
        D = codebook.shape[1]
        scatter_list = []
        for d in range(D):
            Ld = int(levels[d].item())
            oh = F.one_hot(per_dim[:, d], num_classes=Ld).to(codebook.dtype)  # (K, Ld)
            scatter_list.append(oh)
        # Store as a list of buffers with fixed names.
        self._num_dims = D
        for d, sc in enumerate(scatter_list):
            self.register_buffer(f"scatter_d{d}", sc, persistent=False)

    def extra_repr(self) -> str:
        return (
            f"sigma={self.gauss.sigma}, use_index_distance={self.gauss.use_index_distance}, "
            f"lambda_emd={self.lambda_emd}, levels={self.levels.tolist()}"
        )

    def _emd2(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Per-dim EMD^2 between marginalized predicted distribution and one-hot target."""
        B, T, V = logits.shape
        K = self.gauss.codebook_size
        flat_logits_codebook = logits[..., :K].reshape(B * T, K)     # drop extras (e.g. BOS)
        probs = F.softmax(flat_logits_codebook, dim=-1)              # (N, K)

        flat_targets = targets.reshape(B * T).clamp(min=0, max=K - 1)

        total = probs.new_zeros(())
        for d in range(self._num_dims):
            scatter = getattr(self, f"scatter_d{d}")                 # (K, Ld)
            Ld = scatter.shape[1]
            # Marginalize: p_d(l) = sum_k probs(k) * 1{per_dim[k,d] == l}.
            p_d = probs @ scatter                                    # (N, Ld)
            # Target marginal is one-hot at per_dim[target, d].
            tgt_idx_d = self.per_dim_indices[flat_targets, d]        # (N,)
            t_d = F.one_hot(tgt_idx_d, num_classes=Ld).to(p_d.dtype) # (N, Ld)
            cdf_p = torch.cumsum(p_d, dim=-1)
            cdf_t = torch.cumsum(t_d, dim=-1)
            total = total + (cdf_p - cdf_t).pow(2).sum(dim=-1).mean()
        return total / self._num_dims

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> dict:
        """Returns a dict with the combined loss and the individual components.

        Keys:
            loss: total loss used for backprop.
            gaussian_loss: the Gaussian soft-label CE term.
            emd2: the per-dim EMD^2 term (zero tensor when lambda_emd == 0).
        """
        gauss_loss = self.gauss(logits, targets)
        if self.lambda_emd > 0.0:
            emd = self._emd2(logits, targets)
            total = gauss_loss + self.lambda_emd * emd
        else:
            emd = logits.new_zeros(())
            total = gauss_loss
        return {"loss": total, "gaussian_loss": gauss_loss, "emd2": emd}


# ---------------------------------------------------------------------------
# Demo / smoke test
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    torch.manual_seed(0)

    # Build a dummy FSQ-style codebook with levels=[8, 5, 5, 5] -> K=1000.
    levels = torch.tensor([8, 5, 5, 5], dtype=torch.long)
    K = int(levels.prod().item())
    D = levels.numel()

    # Construct a codebook the same way FSQ.indices_to_embedding does so that
    # each scalar dim takes exactly levels[d] unique values in [-1, 1].
    basis = torch.cumprod(torch.cat([torch.tensor([1]), levels[:-1]]), dim=0)
    indices = torch.arange(K).unsqueeze(-1)
    codes_noncentered = (indices // basis) % levels
    half_width = levels // 2
    codebook = (codes_noncentered - half_width).float() / half_width.float()  # (K, D)

    B, T = 4, 8
    V = K + 1  # +1 for BOS, mirroring OATPolicy
    logits = torch.randn(B, T, V, requires_grad=True)
    targets = torch.randint(0, K, (B, T))

    print("== GaussianSoftLabelLoss ==")
    for sigma in [1e-3, 0.1, 1.0, 5.0]:
        crit = GaussianSoftLabelLoss(codebook=codebook, sigma=sigma)
        loss = crit(logits, targets)
        ref_ce = F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1))
        # Check soft target rows sum to 1.
        row_sum = crit.soft_targets.sum(dim=-1)
        assert torch.allclose(row_sum, torch.ones_like(row_sum), atol=1e-4), row_sum
        print(f"  sigma={sigma:>6g}  gauss_loss={loss.item():.4f}  "
              f"ref_CE={ref_ce.item():.4f}  |diff|={(loss - ref_ce).abs().item():.4g}")

    # Sigma -> 0 should match CE closely (soft target collapses to one-hot).
    crit_tiny = GaussianSoftLabelLoss(codebook=codebook, sigma=1e-4)
    loss_tiny = crit_tiny(logits, targets)
    ref_ce = F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1))
    print(f"\n  sanity: sigma->0 loss={loss_tiny.item():.6f}  vs  CE={ref_ce.item():.6f}")
    assert (loss_tiny - ref_ce).abs().item() < 1e-3, "sigma->0 should recover CE"

    # Gradient flows.
    loss_tiny.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    print("  gradient flow: OK")

    print("\n== Index-distance mode ==")
    crit_idx = GaussianSoftLabelLoss(codebook=codebook, sigma=50.0, use_index_distance=True)
    loss_idx = crit_idx(logits.detach().requires_grad_(True), targets)
    print(f"  sigma=50 (index dist)  loss={loss_idx.item():.4f}")

    print("\n== GaussianSoftCEWithEMD ==")
    crit_emd = GaussianSoftCEWithEMD(
        codebook=codebook, levels=levels, sigma=1.0, lambda_emd=0.1
    )
    logits2 = logits.detach().clone().requires_grad_(True)
    out = crit_emd(logits2, targets)
    print(f"  total={out['loss'].item():.4f}  "
          f"gauss={out['gaussian_loss'].item():.4f}  "
          f"emd2={out['emd2'].item():.4f}")
    out["loss"].backward()
    assert logits2.grad is not None and torch.isfinite(logits2.grad).all()
    print("  gradient flow: OK")

    print("\nall checks passed.")
