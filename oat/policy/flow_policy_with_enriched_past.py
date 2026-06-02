import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from oat.policy.base_policy import BasePolicy
from oat.perception.base_obs_encoder import BaseObservationEncoder
from oat.model.diffusion.transformer_for_diffusion import TransformerForDiffusion
from oat.model.common.normalizer import LinearNormalizer


class FlowPolicyWithEnrichedPast(BasePolicy):
    """
    Rectified flow-matching policy with enriched-past conditioning.

    Instead of generating the action chunk from pure Gaussian noise, the flow's
    source distribution is a *warm-start prior* anchored on the most recent past
    action, and the velocity field is conditioned on the enriched-past sequence:
      [ obs (To) | acc | jerk | raw past (past_n) ]   -> (B, To + 2 + past_n, d)

    Training (rectified flow):
      x1   = normalized GT action chunk          (B, H, A)
      mu   = normalize(a_{t-1}) broadcast to H   (B, H, A)
      x0   = mu + sigma * noise                  (warm-start prior, NOT pure noise)
      t    ~ U(0, 1)
      xt   = (1 - t) * x0 + t * x1
      loss = MSE( model(xt, t, cond), x1 - x0 )

    Sampling (Euler, N steps):
      x = mu + sigma * noise; for i in 0..N-1: x += (1/N) * model(x, i/N, cond)
    """

    N_EXPLICIT_FEATURES = 2  # acc, jerk

    def __init__(
        self,
        shape_meta: Dict,
        obs_encoder: BaseObservationEncoder,
        horizon: int,
        n_action_steps: int,
        n_obs_steps: int,
        past_n: int = 7,
        # backbone model params
        embed_dim: int = 256,
        n_layers: int = 4,
        n_heads: int = 4,
        dropout: float = 0.1,
        # flow-matching params
        num_inference_steps: int = 10,
        prior_noise_scale: float = 1.0,
    ):
        super().__init__()

        modalities = obs_encoder.modalities()
        obs_feature_dim = obs_encoder.output_feature_dim()
        action_shape = shape_meta["action"]["shape"]
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_key_shapes = dict()
        obs_ports = []
        for key, attr in shape_meta["obs"].items():
            shape = attr["shape"]
            obs_key_shapes[key] = list(shape)
            _type = attr["type"]
            if _type in modalities:
                obs_ports.append(key)

        # ── Explicit feature projections (independent, different scales) ──
        acc_proj = nn.Sequential(
            nn.Linear(action_dim, obs_feature_dim),
            nn.GELU(),
            nn.Linear(obs_feature_dim, obs_feature_dim),
        )
        jerk_proj = nn.Sequential(
            nn.Linear(action_dim, obs_feature_dim),
            nn.GELU(),
            nn.Linear(obs_feature_dim, obs_feature_dim),
        )

        # ── Raw past action projection (shared across all past_n steps) ──
        raw_proj = nn.Sequential(
            nn.Linear(action_dim, obs_feature_dim),
            nn.GELU(),
            nn.Linear(obs_feature_dim, obs_feature_dim),
        )

        # ── Velocity field backbone ──────────────────────────────────────
        # The conditioning sequence has length max_cond_len; pass it as the
        # backbone's n_obs_steps so cond_pos_emb is sized (1, 1 + max_cond_len, d).
        # causal_attn=False -> no causal/memory mask: the whole action chunk is
        # denoised jointly with full cross-attention to all conditioning tokens.
        max_cond_len = n_obs_steps + self.N_EXPLICIT_FEATURES + past_n
        model = TransformerForDiffusion(
            input_dim=action_dim,
            output_dim=action_dim,
            horizon=horizon,
            n_obs_steps=max_cond_len,
            cond_dim=obs_feature_dim,
            n_layer=n_layers,
            n_head=n_heads,
            n_emb=embed_dim,
            p_drop_emb=dropout,
            p_drop_attn=dropout,
            causal_attn=False,
            time_as_cond=True,
            obs_as_cond=True,
        )

        self.modalities = modalities
        self.obs_key_shapes = obs_key_shapes
        self.obs_ports = obs_ports
        self.obs_encoder = obs_encoder
        self.acc_proj = acc_proj
        self.jerk_proj = jerk_proj
        self.raw_proj = raw_proj
        self.model = model
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.past_n = past_n
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.max_cond_len = max_cond_len
        self.num_inference_steps = num_inference_steps
        self.prior_noise_scale = prior_noise_scale
        self._t_scale = 1000.0  # map flow time [0,1] into SinusoidalPosEmb range

        # Inference-time past buffer
        self._past_buffer: Optional[torch.Tensor] = None

        # ── Report ───────────────────────────────────────────────────────
        num_obs_params = sum(p.numel() for p in obs_encoder.parameters())
        num_trainable_obs = sum(
            p.numel() for p in obs_encoder.parameters() if p.requires_grad
        )
        num_model_params = sum(p.numel() for p in model.parameters())
        num_proj_params = sum(
            p.numel()
            for m in (acc_proj, jerk_proj, raw_proj)
            for p in m.parameters()
        )
        print(
            f"{self.get_policy_name()} initialized with\n"
            f"  obs enc : {num_obs_params / 1e6:.1f}M "
            f"({num_trainable_obs / max(num_obs_params, 1):.5%} trainable)\n"
            f"  policy  : {num_model_params / 1e6:.1f}M\n"
            f"  proj    : {num_proj_params / 1e3:.1f}K (acc + jerk + raw)\n"
            f"  cond_len={n_obs_steps}+{self.N_EXPLICIT_FEATURES}+{past_n}"
            f"={max_cond_len}, flow_steps={num_inference_steps}, "
            f"sigma={prior_noise_scale}\n"
        )

    # ── BasePolicy interface ────────────────────────────────────────────────

    def get_observation_encoder(self):
        return self.obs_encoder

    def get_observation_modalities(self):
        return self.modalities

    def get_observation_ports(self):
        return self.obs_ports

    def get_policy_name(self):
        base_name = "flowpolicy_enriched_"
        for modality in self.modalities:
            if modality != "state":
                base_name += modality + "|"
        return base_name[:-1]

    def create_dummy_observation(
        self,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.Tensor]:
        return super().create_dummy_observation(
            batch_size=batch_size,
            horizon=self.n_obs_steps,
            obs_key_shapes=self.obs_key_shapes,
            device=device,
        )

    def set_normalizer(self, normalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())
        self.obs_encoder.set_normalizer(normalizer)

    def get_optimizer(
        self,
        policy_lr: float,
        obs_enc_lr: float,
        weight_decay: float,
        betas: Tuple[float, float],
    ) -> torch.optim.Optimizer:
        encoder_decay, encoder_nodecay = [], []
        for name, param in self.obs_encoder.named_parameters():
            if not param.requires_grad:
                continue
            (encoder_decay if param.dim() >= 2 else encoder_nodecay).append(param)

        policy_decay, policy_nodecay = [], []
        policy_modules = [self.model, self.acc_proj, self.jerk_proj, self.raw_proj]
        for module in policy_modules:
            for name, param in module.named_parameters():
                if not param.requires_grad:
                    continue
                (policy_decay if param.dim() >= 2 else policy_nodecay).append(param)

        optim_groups = [
            {"params": policy_decay,    "lr": policy_lr,  "weight_decay": weight_decay},
            {"params": policy_nodecay,  "lr": policy_lr,  "weight_decay": 0.0},
            {"params": encoder_decay,   "lr": obs_enc_lr, "weight_decay": weight_decay},
            {"params": encoder_nodecay, "lr": obs_enc_lr, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(optim_groups, betas=betas)

    def reset(self):
        """Called by env_runner at the start of each rollout episode."""
        self._past_buffer = None

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _build_condition(
        self,
        obs_features: torch.Tensor,
        past_actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Build the full condition sequence from obs features and past actions.

        Args:
            obs_features: (B, To, obs_feature_dim)
            past_actions: (B, past_n, action_dim) raw (unnormalized) actions
                          ordered as [a_{t-past_n}, ..., a_{t-1}]

        Returns:
            (B, To + 2 + past_n, obs_feature_dim)
        """
        norm_past = self.normalizer["action"].normalize(past_actions)

        # ── Explicit features from near 3 steps ──────────────────────────
        a_t1 = norm_past[:, -1]   # a_{t-1}
        a_t2 = norm_past[:, -2]   # a_{t-2}
        a_t3 = norm_past[:, -3]   # a_{t-3}

        acc = a_t1 - a_t2                  # acceleration-level
        jerk = a_t1 - 2.0 * a_t2 + a_t3    # jerk-level

        acc_feat = self.acc_proj(acc)      # [B, d]
        jerk_feat = self.jerk_proj(jerk)   # [B, d]

        explicit = torch.stack([acc_feat, jerk_feat], dim=1)  # [B, 2, d]

        # ── Raw history (all past_n steps, shared projection) ────────────
        raw_feat = self.raw_proj(norm_past)  # [B, past_n, d]

        # ── Concatenate: [obs, explicit, raw] ────────────────────────────
        return torch.cat([obs_features, explicit, raw_feat], dim=1)

    def _warm_start_prior(self, past_actions: torch.Tensor) -> torch.Tensor:
        """
        Warm-start mean: normalized most-recent past action broadcast over the
        action horizon. Returns (B, horizon, action_dim).
        """
        last = self.normalizer["action"].normalize(past_actions[:, -1])  # (B, A)
        return last.unsqueeze(1).expand(-1, self.horizon, -1)

    def _scale_t(self, t: torch.Tensor) -> torch.Tensor:
        """Map flow time t in [0,1] into the SinusoidalPosEmb frequency range."""
        return t * self._t_scale

    # ── Inference ───────────────────────────────────────────────────────────

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # encode observation
        features = self.obs_encoder(obs_dict)       # (B, To, d)
        B = features.shape[0]

        # ── get or initialise past buffer ─────────────────────────────────
        if (
            self._past_buffer is None
            or self._past_buffer.shape[0] != B
            or self._past_buffer.device != self.device
        ):
            self._past_buffer = torch.zeros(
                B, self.past_n, self.action_dim,
                device=self.device, dtype=features.dtype,
            )

        # ── build extended condition + warm-start prior ───────────────────
        cond = self._build_condition(features, self._past_buffer)
        mu = self._warm_start_prior(self._past_buffer)          # (B, H, A)
        x = mu + self.prior_noise_scale * torch.randn(
            mu.shape, device=mu.device, dtype=mu.dtype
        )

        # ── Euler integration of the velocity field ───────────────────────
        N = self.num_inference_steps
        dt = 1.0 / N
        for i in range(N):
            t = torch.full((B,), i * dt, device=mu.device, dtype=mu.dtype)
            x = x + dt * self.model(x, self._scale_t(t), cond)

        # unnormalize prediction
        action_pred = self.normalizer["action"].unnormalize(x)

        # receding horizon
        action = action_pred[:, : self.n_action_steps]

        # ── update past buffer ────────────────────────────────────────────
        n_exec = self.n_action_steps
        past_n = self.past_n
        if n_exec >= past_n:
            self._past_buffer = action_pred[:, n_exec - past_n: n_exec].detach().clone()
        else:
            self._past_buffer = torch.cat([
                self._past_buffer[:, n_exec:],
                action_pred[:, :n_exec].detach().clone(),
            ], dim=1)

        return {
            "action": action,
            "action_pred": action_pred,
        }

    # ── Training ────────────────────────────────────────────────────────────

    def forward(self, batch) -> torch.Tensor:
        # normalize target action chunk
        x1 = self.normalizer["action"].normalize(batch["action"])   # (B, H, A)
        B = x1.shape[0]
        device = x1.device

        # encode observation
        features = self.obs_encoder(batch["obs"])                   # (B, To, d)

        # ── build extended condition + warm-start prior ───────────────────
        past_actions = batch["past_action"]                          # (B, past_n, A)
        cond = self._build_condition(features, past_actions)
        mu = self._warm_start_prior(past_actions)                    # (B, H, A)

        noise = torch.randn_like(x1)
        x0 = mu + self.prior_noise_scale * noise                     # warm-start prior

        # ── rectified-flow interpolation ──────────────────────────────────
        t = torch.rand(B, device=device, dtype=x1.dtype)             # (B,)
        t_b = t[:, None, None]
        xt = (1.0 - t_b) * x0 + t_b * x1
        v_target = x1 - x0

        v_pred = self.model(xt, self._scale_t(t), cond)
        loss = F.mse_loss(v_pred, v_target)
        return loss
