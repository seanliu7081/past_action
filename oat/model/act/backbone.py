# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
ResNet backbone with frozen BatchNorm + 2D sine position embedding, as used by ACT.

Adapted from ACT (Zhao et al. 2023, https://github.com/tonyzhaozh/act, MIT license,
detr/models/backbone.py + detr/models/position_encoding.py), which derives from DETR
(Facebook AI Research, Apache-2.0 license).
Modifications for oat:
    * FrozenBatchNorm2d comes from torchvision.ops — numerically identical to ACT's
      vendored copy (same eps=1e-5 added before rsqrt, same `num_batches_tracked`
      deletion on state-dict load)
    * `pretrained=is_main_process()` replaced with the modern
      `weights=ResNet18_Weights.IMAGENET1K_V1` API (same ImageNet checkpoint)
    * dropped the unused dilation/masks/interm-layers plumbing, the learned position
      embedding variant, the `util.misc` NestedTensor annotations, and `IPython.embed`
    * `build_backbone` takes explicit arguments instead of an argparse namespace

As in upstream ACT, the backbone weights remain TRAINABLE (upstream's freeze loop is
commented out); they are simply trained at the lower backbone learning rate. Only the
BatchNorm statistics/affines are frozen.
"""
import math
from typing import List

import torch
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops import FrozenBatchNorm2d


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor):
        x = tensor
        # NOTE(ACT): mask-free variant — the pos grid has batch dim 1 and is
        # broadcast/repeated downstream, so it is identical for every camera.
        not_mask = torch.ones_like(x[0, [0]])
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, num_channels: int):
        super().__init__()
        # NOTE(ACT): upstream's partial-freeze loop is commented out — all backbone
        # weights stay trainable (at the backbone learning rate); only BN is frozen.
        self.body = IntermediateLayerGetter(backbone, return_layers={'layer4': "0"})
        self.num_channels = num_channels

    def forward(self, tensor):
        xs = self.body(tensor)
        return xs


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str):
        weights = {
            'resnet18': torchvision.models.ResNet18_Weights.IMAGENET1K_V1,
            'resnet34': torchvision.models.ResNet34_Weights.IMAGENET1K_V1,
            'resnet50': torchvision.models.ResNet50_Weights.IMAGENET1K_V1,
        }[name]
        backbone = getattr(torchvision.models, name)(
            weights=weights, norm_layer=FrozenBatchNorm2d)
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, num_channels)


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor):
        xs = self[0](tensor)
        out: List[torch.Tensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.dtype))

        return out, pos


def build_backbone(hidden_dim: int, name: str = 'resnet18') -> Joiner:
    position_embedding = PositionEmbeddingSine(hidden_dim // 2, normalize=True)
    backbone = Backbone(name)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model
