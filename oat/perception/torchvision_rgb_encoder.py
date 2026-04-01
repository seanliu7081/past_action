from typing import Dict, List, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
import robomimic.models.base_nets as rmbn

from oat.common.pytorch_util import replace_submodules
from oat.perception.base_obs_encoder import BaseObservationEncoder
from oat.model.common.normalizer import LinearNormalizer, _normalize


class TorchvisionRgbEncoder(BaseObservationEncoder):
    """
    RGB encoder using torchvision ConvNeXt-Tiny backbone with SpatialSoftmax pooling.
    Trains end-to-end. ConvNeXt uses LayerNorm natively (not BatchNorm), so no
    GroupNorm swap is needed for small-batch stability.

    Assumes rgb input: B, To, H, W, C (channels-last, values in [0, 255]).
    """

    def __init__(
        self,
        shape_meta: dict,
        crop_shape: Union[Tuple[int, int], None] = None,
        pretrained: bool = True,
        num_kp: int = 32,
        share_rgb_model: bool = False,
        eval_fixed_crop: bool = False,
        feature_stage: int = 6,
    ):
        super().__init__()

        # Parse rgb ports from shape_meta
        rgb_ports = []
        port_shape = {}
        for key, attr in shape_meta['obs'].items():
            if attr.get('type', '') == 'rgb':
                rgb_ports.append(key)
                shape = attr['shape']
                port_shape[key] = (shape[2], shape[0], shape[1])  # H,W,C -> C,H,W
        self.rgb_ports = rgb_ports

        # Resolve crop shape per port
        if isinstance(crop_shape, dict):
            self.crop_shape = crop_shape
        else:
            self.crop_shape = {port: crop_shape for port in rgb_ports}

        # Build per-port (or shared) backbone + pooling
        def _make_crop_randomizer(input_shape, cs):
            if cs is None:
                return None
            return rmbn.CropRandomizer(
                input_shape=input_shape,
                crop_height=cs[0],
                crop_width=cs[1],
                num_crops=1,
                pos_enc=False,
            )

        # ConvNeXt-Tiny stages:
        #   0-1: stem + stage1 (96ch,  stride 4)   → 76x76 input → 19x19
        #   2-3: stage2        (192ch, stride 8)    → 9x9
        #   4-5: stage3        (384ch, stride 16)   → 4x4
        #   6-7: stage4        (768ch, stride 32)   → 2x2
        # Use feature_stage to truncate. Default=6 (stage3, 4x4) for good
        # spatial resolution with SpatialSoftmax on typical robot image sizes.
        stage_channels = {0: 96, 1: 96, 2: 192, 3: 192, 4: 384, 5: 384, 6: 768, 7: 768}

        def _make_backbone_and_pool(input_shape, cs):
            # ConvNeXt-Tiny feature extractor (remove classifier)
            weights = tv_models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
            convnext = tv_models.convnext_tiny(weights=weights)
            backbone = convnext.features[:feature_stage]
            feat_channels = stage_channels[feature_stage - 1]

            # Compute exact spatial dims via dummy forward pass
            if cs is not None:
                h_in, w_in = cs
            else:
                h_in, w_in = input_shape[1], input_shape[2]
            with torch.no_grad():
                dummy = torch.zeros(1, input_shape[0], h_in, w_in)
                dummy_out = backbone(dummy)
                _, _, h_out, w_out = dummy_out.shape

            pool = rmbn.SpatialSoftmax(
                input_shape=[feat_channels, h_out, w_out],
                num_kp=num_kp,
                temperature=1.0,
                noise_std=0.0,
            )
            return backbone, pool, feat_channels

        self.backbones = nn.ModuleDict()
        self.pools = nn.ModuleDict()
        self.crop_randomizers = nn.ModuleDict()
        self._shared_from = {}

        if share_rgb_model:
            first = rgb_ports[0]
            shape = port_shape[first]
            cs = self.crop_shape[first]
            backbone, pool, feat_channels = _make_backbone_and_pool(shape, cs)
            cr = _make_crop_randomizer(shape, cs)
            self.backbones[first] = backbone
            self.pools[first] = pool
            if cr is not None:
                self.crop_randomizers[first] = cr
            for port in rgb_ports[1:]:
                assert port_shape[port] == shape, \
                    f"share_rgb_model requires identical shapes, got {port_shape[port]} vs {shape}"
                self._shared_from[port] = first
        else:
            for port in rgb_ports:
                shape = port_shape[port]
                cs = self.crop_shape[port]
                backbone, pool, feat_channels = _make_backbone_and_pool(shape, cs)
                cr = _make_crop_randomizer(shape, cs)
                self.backbones[port] = backbone
                self.pools[port] = pool
                if cr is not None:
                    self.crop_randomizers[port] = cr

        self.feat_channels = feat_channels
        self.num_kp = num_kp

        if eval_fixed_crop:
            from oat.perception.crop_randomizer import CropRandomizer as FixedCropRandomizer
            replace_submodules(
                root_module=self,
                predicate=lambda x: isinstance(x, rmbn.CropRandomizer),
                func=lambda x: FixedCropRandomizer(
                    input_shape=x.input_shape,
                    crop_height=x.crop_height,
                    crop_width=x.crop_width,
                    num_crops=x.num_crops,
                    pos_enc=x.pos_enc,
                ),
            )

        self.normalizer = LinearNormalizer()

    def _get_backbone_and_pool(self, port):
        src = self._shared_from.get(port, port)
        return self.backbones[src], self.pools[src]

    def _get_crop_randomizer(self, port):
        src = self._shared_from.get(port, port)
        if src in self.crop_randomizers:
            return self.crop_randomizers[src]
        return None

    def forward(self, obs_dict: Dict) -> torch.Tensor:
        feats = []
        for port in self.rgb_ports:
            x = obs_dict[port]  # [B, To, H, W, C]

            # Normalize
            params = self.normalizer.params_dict.get(port, None)
            if params is not None:
                x = _normalize(x, params, forward=True)

            B, To, H, W, C = x.shape
            x = x.reshape(B * To, H, W, C).permute(0, 3, 1, 2).float()  # [B*To, C, H, W]

            # Crop augmentation
            cr = self._get_crop_randomizer(port)
            if cr is not None:
                x = cr.forward_in(x)

            # Backbone + pool
            backbone, pool = self._get_backbone_and_pool(port)
            spatial = backbone(x)       # [B*To, 768, h, w]
            feat = pool(spatial)        # [B*To, num_kp * 2]

            feat = feat.reshape(B, To, -1)  # [B, To, num_kp * 2]
            feats.append(feat)

        return torch.cat(feats, dim=-1)  # [B, To, num_kp * 2 * N_cameras]

    @torch.no_grad()
    def output_feature_dim(self) -> int:
        return self.num_kp * 2 * len(self.rgb_ports)

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def modalities(self) -> List[str]:
        return ['rgb']
