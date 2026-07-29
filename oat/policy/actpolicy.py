import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from oat.policy.base_policy import BasePolicy
from oat.model.act.detr_vae import build_act_model, kl_divergence
from oat.model.common.normalizer import LinearNormalizer


class ACTPolicy(BasePolicy):
    """
    ACT baseline (Action Chunking with Transformers, Zhao et al. 2023,
    https://github.com/tonyzhaozh/act): a conditional VAE that decodes a chunk of
    future actions from raw camera images + proprioception with a DETR-style
    transformer.

    Training (CVAE):
      z ~ q(z | a_{1:H}, qpos)   via a transformer encoder over [CLS, qpos, actions]
      a_hat = Decoder(ResNet18 image tokens, qpos token, z token)   (B, H, A)
      loss  = masked-L1(a, a_hat) + kl_weight * KL(q || N(0, I))

    Inference (deterministic):
      z = 0 (the prior mean), receding horizon: predict H, execute n_action_steps.
      Optionally ACT's temporal ensembling (requires n_action_steps == 1): query
      every step and exponentially blend overlapping chunks, oldest prediction
      weighted highest (w_i ∝ exp(-k * i), i ordered oldest-first).

    oat adaptations vs. upstream ACT (functionally documented deviations):
      * actions/proprio normalized with the dataset ``LinearNormalizer`` ([-1, 1]
        limits) instead of upstream's mean/std stats; images are /255 +
        ImageNet-normalized inside the policy, as upstream.
      * ``is_pad`` is all-False: oat sequence windows are edge-padded to full
        length, so no pad mask exists (the masked-mean loss expression is kept).
      * proprio = concat of the ``state`` ports incl. ``task_uid`` (task
        conditioning parity with the other oat baselines).
      * ``dec_layers`` defaults to 1: upstream configures 7 decoder layers but only
        the first layer's output is used or trained (the ``hs[0]`` quirk), and dead
        parameters would break DDP with ``find_unused_parameters=False``.
      * EMA / grad clipping / constant-lr warmup come from the shared training
        workspace (upstream has none; all config-overridable).
    """

    def __init__(
        self,
        shape_meta: Dict,
        horizon: int,
        n_action_steps: int,
        n_obs_steps: int,
        # ACT / DETR-VAE params (ACT README defaults, chunk size == horizon)
        hidden_dim: int = 512,
        dim_feedforward: int = 3200,
        enc_layers: int = 4,
        dec_layers: int = 1,
        nheads: int = 8,
        dropout: float = 0.1,
        latent_dim: int = 32,
        kl_weight: float = 10.0,
        backbone: str = 'resnet18',
        pre_norm: bool = False,
        # ACT temporal ensembling (eval-time only; requires n_action_steps == 1)
        temporal_ensemble: bool = False,
        temporal_ensemble_k: float = 0.01,
    ):
        super().__init__()

        modalities = ['rgb', 'state']
        action_shape = shape_meta["action"]["shape"]
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_key_shapes = dict()
        rgb_ports = []
        state_ports = []
        for key, attr in shape_meta["obs"].items():
            shape = attr["shape"]
            obs_key_shapes[key] = list(shape)
            _type = attr["type"]
            if _type == 'rgb':
                rgb_ports.append(key)
            elif _type == 'state':
                state_ports.append(key)
        proprio_dim = sum(obs_key_shapes[key][0] for key in state_ports)

        if temporal_ensemble:
            assert n_action_steps == 1, \
                "temporal ensembling queries the policy every env step"

        # ── DETR-VAE model (raw images -> ResNet18 spatial tokens) ───────────
        model = build_act_model(
            num_cameras=len(rgb_ports),
            proprio_dim=proprio_dim,
            action_dim=action_dim,
            num_queries=horizon,
            hidden_dim=hidden_dim,
            dim_feedforward=dim_feedforward,
            enc_layers=enc_layers,
            dec_layers=dec_layers,
            nheads=nheads,
            dropout=dropout,
            latent_dim=latent_dim,
            pre_norm=pre_norm,
            backbone_name=backbone,
        )

        # ImageNet normalization constants, as upstream ACT (policy.py)
        self.register_buffer(
            'imagenet_mean',
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False)
        self.register_buffer(
            'imagenet_std',
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False)

        self.modalities = modalities
        self.obs_key_shapes = obs_key_shapes
        self.rgb_ports = rgb_ports
        self.state_ports = state_ports
        self.model = model
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.proprio_dim = proprio_dim
        self.action_dim = action_dim
        self.kl_weight = kl_weight
        self.temporal_ensemble = temporal_ensemble
        self.temporal_ensemble_k = temporal_ensemble_k
        self._chunk_history = []  # past action chunks for temporal ensembling

        # ── Report ───────────────────────────────────────────────────────────
        num_backbone_params = sum(
            p.numel() for name, p in model.named_parameters() if 'backbone' in name
        )
        num_model_params = sum(p.numel() for p in model.parameters())
        print(
            f"{self.get_policy_name()} initialized with\n"
            f"  backbone: {num_backbone_params / 1e6:.1f}M ({backbone}, ImageNet init)\n"
            f"  total   : {num_model_params / 1e6:.1f}M\n"
            f"  chunk={horizon}, n_action_steps={n_action_steps}, "
            f"kl_weight={kl_weight}, latent_dim={latent_dim}\n"
        )

    # ── BasePolicy interface ────────────────────────────────────────────────

    def get_observation_encoder(self):
        # ACT has no oat-style obs encoder; the DETR backbone is the vision path.
        return self.model.backbone

    def get_observation_modalities(self):
        return self.modalities

    def get_observation_ports(self):
        return self.rgb_ports + self.state_ports

    def get_policy_name(self):
        base_name = "actpolicy_"
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

    def reset(self):
        self._chunk_history = []

    def get_optimizer(
        self,
        policy_lr: float,
        obs_enc_lr: float,
        weight_decay: float,
        betas: Tuple[float, float],
    ) -> torch.optim.Optimizer:
        # Faithful to upstream ACT (detr/main.py): two AdamW param groups split on
        # the "backbone" substring, weight decay applied to ALL params (upstream
        # has no no-decay list). obs_enc_lr plays the role of ACT's lr_backbone.
        backbone_params, other_params = [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            (backbone_params if 'backbone' in name else other_params).append(param)
        return torch.optim.AdamW(
            [
                {"params": other_params, "lr": policy_lr},
                {"params": backbone_params, "lr": obs_enc_lr},
            ],
            lr=policy_lr, weight_decay=weight_decay, betas=betas,
        )

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _preprocess_images(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """[B, To, H, W, 3] per rgb port (uint8 or float, 0-255) ->
        [B, num_cam, 3, H, W], /255 + ImageNet-normalized (as upstream ACT)."""
        images = []
        for port in self.rgb_ports:
            x = obs_dict[port][:, -1]                     # last frame: [B, H, W, 3]
            x = x.to(self.imagenet_mean.dtype).permute(0, 3, 1, 2) / 255.0
            x = (x - self.imagenet_mean) / self.imagenet_std
            images.append(x)
        return torch.stack(images, dim=1)                 # [B, num_cam, 3, H, W]

    def _assemble_proprio(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Normalize each state port and concat -> [B, proprio_dim]."""
        feats = [
            self.normalizer[port].normalize(obs_dict[port][:, -1])
            for port in self.state_ports
        ]
        return torch.cat(feats, dim=-1)

    # ── Inference ───────────────────────────────────────────────────────────

    def predict_action(self, obs_dict: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        qpos = self._assemble_proprio(obs_dict)
        images = self._preprocess_images(obs_dict)

        # no actions -> z = 0 (prior mean), deterministic
        a_hat, _ = self.model(qpos, images)

        # unnormalize prediction
        action_pred = self.normalizer["action"].unnormalize(a_hat)  # (B, H, A)

        if self.temporal_ensemble:
            action = self._ensemble_step(action_pred)     # (B, 1, A)
        else:
            # receding horizon
            action = action_pred[:, : self.n_action_steps]

        return {
            "action": action,
            "action_pred": action_pred,
        }

    def _ensemble_step(self, action_pred: torch.Tensor) -> torch.Tensor:
        """ACT temporal ensembling (imitate_episodes.py eval loop): blend the
        current step's action across all live chunks with w_i ∝ exp(-k*i), i
        ordered oldest-first — the OLDEST surviving prediction gets the highest
        weight. Called once per env step (n_action_steps == 1)."""
        self._chunk_history.append(action_pred)
        if len(self._chunk_history) > self.horizon:
            self._chunk_history.pop(0)
        n = len(self._chunk_history)
        # _chunk_history[i] was predicted (n-1-i) steps ago, so its action for the
        # current step sits at chunk index (n-1-i).
        stacked = torch.stack(
            [chunk[:, n - 1 - i] for i, chunk in enumerate(self._chunk_history)]
        )                                                  # (n, B, A), i=0 oldest
        weights = torch.exp(
            -self.temporal_ensemble_k
            * torch.arange(n, device=stacked.device, dtype=stacked.dtype))
        weights = weights / weights.sum()
        action = (stacked * weights.view(n, 1, 1)).sum(dim=0)  # (B, A)
        return action.unsqueeze(1)                         # (B, 1, A)

    # ── Training ────────────────────────────────────────────────────────────

    def forward(self, batch) -> torch.Tensor:
        actions = self.normalizer["action"].normalize(batch["action"])  # (B, H, A)
        actions = actions[:, : self.horizon]

        qpos = self._assemble_proprio(batch["obs"])
        images = self._preprocess_images(batch["obs"])

        # oat windows are edge-padded to full length -> no pad mask; the upstream
        # masked-mean expression is kept so a real mask would just work.
        is_pad = torch.zeros(
            actions.shape[:2], dtype=torch.bool, device=actions.device)

        a_hat, (mu, logvar) = self.model(qpos, images, actions=actions, is_pad=is_pad)

        all_l1 = F.l1_loss(actions, a_hat, reduction='none')
        l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
        kld = kl_divergence(mu.float(), logvar.float())    # fp32 for bf16 stability
        loss = l1 + self.kl_weight * kld
        return loss
