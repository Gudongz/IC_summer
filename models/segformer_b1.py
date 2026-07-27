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

    def load_compatible_state_dict(self, state_dict: dict[str, Tensor]) -> bool:
        """Load checkpoints saved by either Hugging Face SegFormer layout.

        Transformers renamed MiT encoder modules between versions. The tensor
        shapes did not change, so checkpoints only need key translation rather
        than retraining when the installed version uses the other layout.
        """
        expected_keys = self.state_dict().keys()
        expected_legacy = any(key.startswith("encoder.encoder.") for key in expected_keys)
        checkpoint_legacy = any(key.startswith("encoder.encoder.") for key in state_dict)
        expected_short_attention_names = any(".attention.q_proj." in key for key in expected_keys)
        checkpoint_keys = set(state_dict)
        if expected_legacy == checkpoint_legacy and checkpoint_keys == set(expected_keys):
            self.load_state_dict(state_dict)
            return False

        translated: dict[str, Tensor] = {}
        for key, value in state_dict.items():
            if expected_legacy != checkpoint_legacy:
                key = self._translate_encoder_key(key, to_legacy=expected_legacy)
            key = self._translate_attention_projection_alias(key, use_short_names=expected_short_attention_names)
            translated[key] = value
        self.load_state_dict(translated)
        return True

    @staticmethod
    def _translate_encoder_key(key: str, to_legacy: bool) -> str:
        """Translate one encoder key between ``encoder.encoder`` and ``stages`` layouts."""
        if to_legacy:
            rules = (
                (r"^encoder\.stages\.(\d+)\.patch_embeddings\.", r"encoder.encoder.patch_embeddings.\1."),
                (r"^encoder\.stages\.(\d+)\.blocks\.(\d+)\.layernorm_before\.", r"encoder.encoder.block.\1.\2.layer_norm_1."),
                (r"^encoder\.stages\.(\d+)\.blocks\.(\d+)\.layernorm_after\.", r"encoder.encoder.block.\1.\2.layer_norm_2."),
                (r"^encoder\.stages\.(\d+)\.blocks\.(\d+)\.attention\.q_proj\.", r"encoder.encoder.block.\1.\2.attention.self.query."),
                (r"^encoder\.stages\.(\d+)\.blocks\.(\d+)\.attention\.k_proj\.", r"encoder.encoder.block.\1.\2.attention.self.key."),
                (r"^encoder\.stages\.(\d+)\.blocks\.(\d+)\.attention\.v_proj\.", r"encoder.encoder.block.\1.\2.attention.self.value."),
                (r"^encoder\.stages\.(\d+)\.blocks\.(\d+)\.attention\.o_proj\.", r"encoder.encoder.block.\1.\2.attention.output.dense."),
                (r"^encoder\.stages\.(\d+)\.blocks\.(\d+)\.attention\.sequence_reduction\.sequence_reduction\.", r"encoder.encoder.block.\1.\2.attention.self.sr."),
                (r"^encoder\.stages\.(\d+)\.blocks\.(\d+)\.attention\.sequence_reduction\.layer_norm\.", r"encoder.encoder.block.\1.\2.attention.self.layer_norm."),
                (r"^encoder\.stages\.(\d+)\.blocks\.(\d+)\.mlp\.fc1\.", r"encoder.encoder.block.\1.\2.mlp.dense1."),
                (r"^encoder\.stages\.(\d+)\.blocks\.(\d+)\.mlp\.fc2\.", r"encoder.encoder.block.\1.\2.mlp.dense2."),
                (r"^encoder\.stages\.(\d+)\.blocks\.(\d+)\.mlp\.dwconv\.dwconv\.", r"encoder.encoder.block.\1.\2.mlp.dwconv.dwconv."),
                (r"^encoder\.stages\.(\d+)\.layer_norm\.", r"encoder.encoder.layer_norm.\1."),
            )
        else:
            rules = (
                (r"^encoder\.encoder\.patch_embeddings\.(\d+)\.", r"encoder.stages.\1.patch_embeddings."),
                (r"^encoder\.encoder\.block\.(\d+)\.(\d+)\.layer_norm_1\.", r"encoder.stages.\1.blocks.\2.layernorm_before."),
                (r"^encoder\.encoder\.block\.(\d+)\.(\d+)\.layer_norm_2\.", r"encoder.stages.\1.blocks.\2.layernorm_after."),
                (r"^encoder\.encoder\.block\.(\d+)\.(\d+)\.attention\.self\.(query|key|value)\.", r"encoder.stages.\1.blocks.\2.attention.\3_proj."),
                (r"^encoder\.encoder\.block\.(\d+)\.(\d+)\.attention\.output\.dense\.", r"encoder.stages.\1.blocks.\2.attention.o_proj."),
                (r"^encoder\.encoder\.block\.(\d+)\.(\d+)\.attention\.self\.sr\.", r"encoder.stages.\1.blocks.\2.attention.sequence_reduction.sequence_reduction."),
                (r"^encoder\.encoder\.block\.(\d+)\.(\d+)\.attention\.self\.layer_norm\.", r"encoder.stages.\1.blocks.\2.attention.sequence_reduction.layer_norm."),
                (r"^encoder\.encoder\.block\.(\d+)\.(\d+)\.mlp\.dense1\.", r"encoder.stages.\1.blocks.\2.mlp.fc1."),
                (r"^encoder\.encoder\.block\.(\d+)\.(\d+)\.mlp\.dense2\.", r"encoder.stages.\1.blocks.\2.mlp.fc2."),
                (r"^encoder\.encoder\.block\.(\d+)\.(\d+)\.mlp\.dwconv\.dwconv\.", r"encoder.stages.\1.blocks.\2.mlp.dwconv.dwconv."),
                (r"^encoder\.encoder\.layer_norm\.(\d+)\.", r"encoder.stages.\1.layer_norm."),
            )
        for pattern, replacement in rules:
            key = re.sub(pattern, replacement, key)
        return key

    @staticmethod
    def _translate_attention_projection_alias(key: str, use_short_names: bool) -> str:
        """Handle Transformers' ``query_proj`` ↔ ``q_proj`` rename."""
        if use_short_names:
            for old, new in (("query_proj", "q_proj"), ("key_proj", "k_proj"), ("value_proj", "v_proj")):
                key = key.replace(f".attention.{old}.", f".attention.{new}.")
        else:
            for old, new in (("q_proj", "query_proj"), ("k_proj", "key_proj"), ("v_proj", "value_proj")):
                key = key.replace(f".attention.{old}.", f".attention.{new}.")
        return key
