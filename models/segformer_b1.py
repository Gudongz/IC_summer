"""SegFormer-B1 with a binary lesion segmentation decoder."""

from __future__ import annotations

import re

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class SegFormerB1(nn.Module):
    """ImageNet-pretrained MiT-B1 encoder plus a lightweight MLP decoder."""

    checkpoint_name = "nvidia/mit-b1"

    def __init__(self, out_channels: int = 1, pretrained: bool = True, decoder_channels: int = 256) -> None:
        super().__init__()
        try:
            from transformers import SegformerConfig, SegformerModel
        except ModuleNotFoundError as exc:
            raise ImportError("SegFormer-B1 requires transformers. Install it with: python -m pip install transformers") from exc
        if pretrained:
            self.encoder = SegformerModel.from_pretrained(self.checkpoint_name)
        else:
            config = SegformerConfig(
                num_channels=3, depths=[2, 2, 2, 2], hidden_sizes=[64, 128, 320, 512],
                num_attention_heads=[1, 2, 5, 8], sr_ratios=[8, 4, 2, 1], mlp_ratios=[4, 4, 4, 4],
                drop_path_rate=0.1,
            )
            self.encoder = SegformerModel(config)
        self.register_buffer("input_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("input_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.normalize_input = pretrained
        hidden_sizes = self.encoder.config.hidden_sizes
        self.projections = nn.ModuleList([nn.Conv2d(channels, decoder_channels, 1) for channels in hidden_sizes])
        self.fuse = nn.Sequential(
            nn.Conv2d(decoder_channels * len(hidden_sizes), decoder_channels, 1, bias=False),
            nn.BatchNorm2d(decoder_channels), nn.ReLU(inplace=True), nn.Dropout2d(0.1),
        )
        self.head = nn.Conv2d(decoder_channels, out_channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        input_size = x.shape[-2:]
        if self.normalize_input:
            x = (x - self.input_mean) / self.input_std
        outputs = self.encoder(pixel_values=x, output_hidden_states=True, return_dict=True)
        features = outputs.hidden_states
        target_size = features[0].shape[-2:]
        decoded = [F.interpolate(proj(feature), size=target_size, mode="bilinear", align_corners=False) for proj, feature in zip(self.projections, features)]
        logits = self.head(self.fuse(torch.cat(decoded, dim=1)))
        return F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)

    def load_compatible_state_dict(self, state_dict: dict[str, Tensor]) -> None:
        """Load checkpoints saved by the legacy Hugging Face SegFormer layout.

        Transformers renamed MiT encoder modules between versions. The tensor
        shapes did not change, so old project checkpoints only need key
        translation rather than retraining.
        """
        expected_keys = self.state_dict().keys()
        if any(key.startswith("encoder.encoder.") for key in expected_keys):
            # Older transformers releases still use the exact checkpoint
            # naming scheme, so translating it would be incorrect.
            self.load_state_dict(state_dict)
            return
        if not any(key.startswith("encoder.encoder.") for key in state_dict):
            self.load_state_dict(state_dict)
            return

        translated: dict[str, Tensor] = {}
        for key, value in state_dict.items():
            key = re.sub(
                r"^encoder\.encoder\.patch_embeddings\.(\d+)\.",
                r"encoder.stages.\1.patch_embeddings.",
                key,
            )
            key = re.sub(
                r"^encoder\.encoder\.block\.(\d+)\.(\d+)\.layer_norm_1\.",
                r"encoder.stages.\1.blocks.\2.layernorm_before.",
                key,
            )
            key = re.sub(
                r"^encoder\.encoder\.block\.(\d+)\.(\d+)\.layer_norm_2\.",
                r"encoder.stages.\1.blocks.\2.layernorm_after.",
                key,
            )
            key = re.sub(
                r"^encoder\.encoder\.block\.(\d+)\.(\d+)\.attention\.self\.(query|key|value)\.",
                r"encoder.stages.\1.blocks.\2.attention.\3_proj.",
                key,
            )
            key = re.sub(
                r"^encoder\.encoder\.block\.(\d+)\.(\d+)\.attention\.output\.dense\.",
                r"encoder.stages.\1.blocks.\2.attention.o_proj.",
                key,
            )
            key = re.sub(
                r"^encoder\.encoder\.block\.(\d+)\.(\d+)\.attention\.self\.sr\.",
                r"encoder.stages.\1.blocks.\2.attention.sequence_reduction.sequence_reduction.",
                key,
            )
            key = re.sub(
                r"^encoder\.encoder\.block\.(\d+)\.(\d+)\.attention\.self\.layer_norm\.",
                r"encoder.stages.\1.blocks.\2.attention.sequence_reduction.layer_norm.",
                key,
            )
            key = re.sub(
                r"^encoder\.encoder\.block\.(\d+)\.(\d+)\.mlp\.dense1\.",
                r"encoder.stages.\1.blocks.\2.mlp.fc1.",
                key,
            )
            key = re.sub(
                r"^encoder\.encoder\.block\.(\d+)\.(\d+)\.mlp\.dense2\.",
                r"encoder.stages.\1.blocks.\2.mlp.fc2.",
                key,
            )
            key = re.sub(
                r"^encoder\.encoder\.block\.(\d+)\.(\d+)\.mlp\.dwconv\.dwconv\.",
                r"encoder.stages.\1.blocks.\2.mlp.dwconv.dwconv.",
                key,
            )
            key = re.sub(
                r"^encoder\.encoder\.layer_norm\.(\d+)\.",
                r"encoder.stages.\1.layer_norm.",
                key,
            )
            key = key.replace(".attention.query_proj.", ".attention.q_proj.")
            key = key.replace(".attention.key_proj.", ".attention.k_proj.")
            key = key.replace(".attention.value_proj.", ".attention.v_proj.")
            translated[key] = value
        self.load_state_dict(translated)
