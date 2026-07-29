# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR-VAE: the CVAE transformer model behind ACT (Action Chunking with Transformers).

Adapted from ACT (Zhao et al. 2023, https://github.com/tonyzhaozh/act, MIT license,
detr/models/detr_vae.py + detr/main.py + policy.py), which derives from DETR
(Facebook AI Research, Apache-2.0 license).
Modifications for oat:
    * the hardcoded ALOHA `state_dim = 14` is split into `proprio_dim` (robot state
      input) and `action_dim` (action output), and `latent_dim` is a constructor arg
    * `is_pad_head` is removed — upstream computes it but never uses it in any loss,
      and dead parameters break DDP with `find_unused_parameters=False`
    * the state-only (`backbones is None`) branch and CNNMLP variant are dropped
    * `reparametrize` uses `torch.randn_like` instead of the deprecated
      `torch.autograd.Variable` (identical distribution)
    * the inference-time zero latent takes its dtype from the input (bf16-friendly)
      instead of hardcoded float32
    * `build_act_model` replaces the argparse/sys.argv-driven
      `build_ACT_model_and_optimizer` with explicit keyword arguments and does not
      call `.cuda()` (device placement is the caller's job)
    * `kl_divergence` returns the scalar `total_kld[0]` directly (the only value
      upstream uses); the unused dimension-wise/mean variants are dropped
"""
import numpy as np
import torch
from torch import nn

from oat.model.act.backbone import build_backbone
from oat.model.act.transformer import (
    Transformer, TransformerEncoder, TransformerEncoderLayer)


def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    return mu + std * torch.randn_like(std)


def kl_divergence(mu, logvar):
    """KL(q(z|x) || N(0, I)): sum over latent dims, mean over batch.

    Equals `total_kld[0]` of upstream ACT's kl_divergence (policy.py).
    """
    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    return klds.sum(1).mean(0)


def get_sinusoid_encoding_table(n_position, d_hid):
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class DETRVAE(nn.Module):
    """ACT's CVAE: a DETR-style transformer that decodes a chunk of future actions.

    Train time: a separate transformer encoder embeds [CLS, proprio, action chunk]
    and the CLS output parameterizes q(z | actions, proprio); z is reparametrized.
    Test time: z = 0 (the prior mean), making inference deterministic.
    """

    def __init__(self, backbone, transformer, encoder, num_queries: int,
                 num_cameras: int, proprio_dim: int, action_dim: int,
                 latent_dim: int = 32):
        super().__init__()
        self.num_queries = num_queries
        self.num_cameras = num_cameras
        self.transformer = transformer
        self.encoder = encoder
        hidden_dim = transformer.d_model
        self.action_head = nn.Linear(hidden_dim, action_dim)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        # NOTE: the attribute name must contain "backbone" — the optimizer splits
        # param groups on that substring, exactly like upstream ACT (detr/main.py).
        self.backbone = backbone
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.input_proj_robot_state = nn.Linear(proprio_dim, hidden_dim)

        # encoder extra parameters
        self.latent_dim = latent_dim  # final size of latent z
        self.cls_embed = nn.Embedding(1, hidden_dim)  # extra cls token embedding
        self.encoder_action_proj = nn.Linear(action_dim, hidden_dim)  # project action to embedding
        self.encoder_joint_proj = nn.Linear(proprio_dim, hidden_dim)  # project qpos to embedding
        self.latent_proj = nn.Linear(hidden_dim, self.latent_dim*2)  # project hidden state to latent std, var
        self.register_buffer('pos_table', get_sinusoid_encoding_table(1+1+num_queries, hidden_dim))  # [CLS], qpos, a_seq

        # decoder extra parameters
        self.latent_out_proj = nn.Linear(self.latent_dim, hidden_dim)  # project latent sample to embedding
        self.additional_pos_embed = nn.Embedding(2, hidden_dim)  # learned position embedding for proprio and latent

    def forward(self, qpos, image, actions=None, is_pad=None):
        """
        qpos: batch, proprio_dim
        image: batch, num_cam, channel, height, width  (already ImageNet-normalized)
        actions: batch, seq, action_dim  (None at inference)
        is_pad: batch, seq  bool
        """
        is_training = actions is not None  # train or val
        bs, _ = qpos.shape
        ### Obtain latent z from action sequence
        if is_training:
            # project action sequence to embedding dim, and concat with a CLS token
            action_embed = self.encoder_action_proj(actions)  # (bs, seq, hidden_dim)
            qpos_embed = self.encoder_joint_proj(qpos)  # (bs, hidden_dim)
            qpos_embed = torch.unsqueeze(qpos_embed, axis=1)  # (bs, 1, hidden_dim)
            cls_embed = self.cls_embed.weight  # (1, hidden_dim)
            cls_embed = torch.unsqueeze(cls_embed, axis=0).repeat(bs, 1, 1)  # (bs, 1, hidden_dim)
            encoder_input = torch.cat([cls_embed, qpos_embed, action_embed], axis=1)  # (bs, seq+1, hidden_dim)
            encoder_input = encoder_input.permute(1, 0, 2)  # (seq+1, bs, hidden_dim)
            # do not mask cls token
            cls_joint_is_pad = torch.full((bs, 2), False).to(qpos.device)  # False: not a padding
            is_pad = torch.cat([cls_joint_is_pad, is_pad], axis=1)  # (bs, seq+1)
            # obtain position embedding
            pos_embed = self.pos_table.clone().detach()
            pos_embed = pos_embed.permute(1, 0, 2)  # (seq+1, 1, hidden_dim)
            # query model
            encoder_output = self.encoder(encoder_input, pos=pos_embed, src_key_padding_mask=is_pad)
            encoder_output = encoder_output[0]  # take cls output only
            latent_info = self.latent_proj(encoder_output)
            mu = latent_info[:, :self.latent_dim]
            logvar = latent_info[:, self.latent_dim:]
            latent_sample = reparametrize(mu, logvar)
            latent_input = self.latent_out_proj(latent_sample)
        else:
            mu = logvar = None
            latent_sample = torch.zeros([bs, self.latent_dim], dtype=qpos.dtype, device=qpos.device)
            latent_input = self.latent_out_proj(latent_sample)

        # Image observation features and position embeddings
        all_cam_features = []
        all_cam_pos = []
        for cam_id in range(self.num_cameras):
            # single backbone SHARED across cameras, as upstream (`self.backbones[0]`)
            features, pos = self.backbone(image[:, cam_id])
            features = features[0]  # take the last layer feature
            pos = pos[0]
            all_cam_features.append(self.input_proj(features))
            all_cam_pos.append(pos)
        # proprioception features
        proprio_input = self.input_proj_robot_state(qpos)
        # fold camera dimension into width dimension
        src = torch.cat(all_cam_features, axis=3)
        pos = torch.cat(all_cam_pos, axis=3)
        # NOTE(upstream quirk, kept for fidelity): the transformer returns the stack
        # of ALL decoder layers' outputs and `[0]` selects the FIRST layer's output
        # (github.com/tonyzhaozh/act/issues/25). With dec_layers > 1 the later layers
        # would never receive gradients, which is why `build_act_model` defaults to a
        # single decoder layer — functionally identical to upstream's 7-layer config.
        hs = self.transformer(src, None, self.query_embed.weight, pos, latent_input, proprio_input, self.additional_pos_embed.weight)[0]
        a_hat = self.action_head(hs)
        return a_hat, (mu, logvar)


def build_cvae_encoder(hidden_dim: int, dropout: float, nheads: int,
                       dim_feedforward: int, enc_layers: int,
                       pre_norm: bool) -> TransformerEncoder:
    """The CVAE 'style' encoder q(z | actions, proprio) — port of upstream
    `build_encoder(args)`; shares the layer count with the main encoder."""
    activation = "relu"
    encoder_layer = TransformerEncoderLayer(hidden_dim, nheads, dim_feedforward,
                                            dropout, activation, pre_norm)
    encoder_norm = nn.LayerNorm(hidden_dim) if pre_norm else None
    encoder = TransformerEncoder(encoder_layer, enc_layers, encoder_norm)
    return encoder


def build_act_model(
    num_cameras: int,
    proprio_dim: int,
    action_dim: int,
    num_queries: int,
    hidden_dim: int = 512,
    dim_feedforward: int = 3200,
    enc_layers: int = 4,
    dec_layers: int = 1,
    nheads: int = 8,
    dropout: float = 0.1,
    latent_dim: int = 32,
    pre_norm: bool = False,
    backbone_name: str = 'resnet18',
) -> DETRVAE:
    """Explicit-kwargs replacement for upstream `build_ACT_model_and_optimizer`.

    Defaults follow the ACT README config (hidden 512, ffn 3200, enc 4, heads 8),
    except `dec_layers=1`: upstream configures 7 decoder layers but only the first
    layer's output is ever used or trained (the `hs[0]` quirk, see DETRVAE.forward),
    so a single layer is functionally identical and safe for DDP with
    `find_unused_parameters=False`.
    """
    backbone = build_backbone(hidden_dim, backbone_name)
    transformer = Transformer(
        d_model=hidden_dim,
        dropout=dropout,
        nhead=nheads,
        dim_feedforward=dim_feedforward,
        num_encoder_layers=enc_layers,
        num_decoder_layers=dec_layers,
        normalize_before=pre_norm,
        return_intermediate_dec=True,
    )
    encoder = build_cvae_encoder(hidden_dim, dropout, nheads, dim_feedforward,
                                 enc_layers, pre_norm)
    model = DETRVAE(
        backbone,
        transformer,
        encoder,
        num_queries=num_queries,
        num_cameras=num_cameras,
        proprio_dim=proprio_dim,
        action_dim=action_dim,
        latent_dim=latent_dim,
    )
    return model
