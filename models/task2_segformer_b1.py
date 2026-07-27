"""Task 2 five-attribute segmentation with a SegFormer-B1 backbone."""

from __future__ import annotations

import re
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .task2_resnet34_unet import AttributeAttentionHead, LesionAwareFusion


class Task2SegFormerB1(nn.Module):
    """MiT-B1 with a lesion-prior branch, SegFormer decoder, and five heads."""

    checkpoint_name = "nvidia/mit-b1"
    task1_model_name = "segformer_b1"

    def __init__(
        self,
        pretrained: bool = True,
        num_attributes: int = 5,
        decoder_channels: int = 256,
    ) -> None:
        super().__init__()
        if num_attributes <= 0:
            raise ValueError("num_attributes must be positive")
        try:
            from transformers import SegformerConfig, SegformerModel
        except ModuleNotFoundError as exc:
            raise ImportError("Task2SegFormerB1 requires transformers.") from exc
        if pretrained:
            self.encoder = SegformerModel.from_pretrained(self.checkpoint_name)
        else:
            self.encoder = SegformerModel(
                SegformerConfig(
                    num_channels=3, depths=[2, 2, 2, 2], hidden_sizes=[64, 128, 320, 512],
                    num_attention_heads=[1, 2, 5, 8], sr_ratios=[8, 4, 2, 1], mlp_ratios=[4, 4, 4, 4],
                    drop_path_rate=0.1,
                )
            )
        self.num_attributes = num_attributes
        self.normalize_input = pretrained
        self.register_buffer("input_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("input_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        hidden_sizes = self.encoder.config.hidden_sizes
        # The binary prior is independently encoded at each MiT scale.  This
        # avoids feeding a non-RGB map into the ImageNet-pretrained encoder.
        self.prior_adapters = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(1, channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
            )
            for channels in hidden_sizes
        )
        self.lesion_fusions = nn.ModuleList(
            LesionAwareFusion(channels, channels) for channels in hidden_sizes
        )
        self.projections = nn.ModuleList(nn.Conv2d(channels, decoder_channels, 1) for channels in hidden_sizes)
        self.fuse = nn.Sequential(
            nn.Conv2d(decoder_channels * len(hidden_sizes), decoder_channels, 1, bias=False),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
        )
        self.attribute_heads = nn.ModuleList(
            AttributeAttentionHead(decoder_channels) for _ in range(num_attributes)
        )

    def load_task1_encoder(self, checkpoint_path: str | Path) -> None:
        """Load only a matching Task 1 SegFormer-B1 encoder from a checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model_name = checkpoint.get("model_name")
        if model_name != self.task1_model_name:
            raise ValueError(
                f"Checkpoint {checkpoint_path} contains {model_name!r}; "
                f"Task2SegFormerB1 requires {self.task1_model_name!r}."
            )
        state_dict = checkpoint.get("model_state_dict")
        if not isinstance(state_dict, dict):
            raise ValueError(f"Checkpoint {checkpoint_path} has no model_state_dict.")
        # ``self.encoder`` is the inner Hugging Face model, so legacy keys
        # begin with ``encoder.`` here (the Task 1 wrapper sees an additional
        # outer ``encoder.`` prefix).  Newer Transformers versions use
        # ``stages.`` instead.
        expected_legacy = any(key.startswith("encoder.") for key in self.encoder.state_dict())
        expected_short_attention_names = any(".attention.q_proj." in key for key in self.encoder.state_dict())
        encoder_state: dict[str, Tensor] = {}
        for key, value in state_dict.items():
            if key.startswith("encoder."):
                encoder_key = self._translate_encoder_key(key, to_legacy=expected_legacy)
                encoder_key = self._translate_attention_projection_alias(encoder_key, use_short_names=expected_short_attention_names)
                encoder_state[encoder_key.removeprefix("encoder.")] = value
        if not encoder_state:
            raise ValueError(f"Checkpoint {checkpoint_path} contains no encoder weights.")
        self.encoder.load_state_dict(encoder_state, strict=True)
        self.normalize_input = bool(checkpoint.get("pretrained", checkpoint.get("pretrained_encoder", False)))

    def forward(self, x: Tensor, lesion_prior: Tensor | None = None) -> Tensor:
        """Return five logits using a binary Task 1 lesion prior when supplied."""
        input_size = x.shape[-2:]
        if lesion_prior is None:
            lesion_prior = x.new_zeros((x.shape[0], 1, *input_size))
        if lesion_prior.ndim != 4 or lesion_prior.shape[1] != 1:
            raise ValueError("lesion_prior must have shape B×1×H×W")
        if lesion_prior.shape[-2:] != input_size:
            lesion_prior = F.interpolate(lesion_prior, size=input_size, mode="nearest")
        if self.normalize_input:
            x = (x - self.input_mean) / self.input_std
        outputs = self.encoder(pixel_values=x, output_hidden_states=True, return_dict=True)
        features = outputs.hidden_states
        prior_features = [
            adapter(F.interpolate(lesion_prior, size=feature.shape[-2:], mode="nearest"))
            for adapter, feature in zip(self.prior_adapters, features)
        ]
        features = [
            fusion(feature, prior_feature)
            for fusion, feature, prior_feature in zip(self.lesion_fusions, features, prior_features)
        ]
        target_size = features[0].shape[-2:]
        decoded = [
            F.interpolate(projection(feature), size=target_size, mode="bilinear", align_corners=False)
            for projection, feature in zip(self.projections, features)
        ]
        shared_features = self.fuse(torch.cat(decoded, dim=1))
        logits = torch.cat([head(shared_features) for head in self.attribute_heads], dim=1)
        return F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)

    @staticmethod
    def _translate_encoder_key(key: str, to_legacy: bool) -> str:
        """Translate the two Transformers SegFormer encoder key layouts."""
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
                (r"^encoder\.encoder\.layer_norm\.(\d+)\.", r"encoder.stages.\1.layer_norm.\1."),
            )
        for pattern, replacement in rules:
            key = re.sub(pattern, replacement, key)
        return key

    @staticmethod
    def _translate_attention_projection_alias(key: str, use_short_names: bool) -> str:
        if use_short_names:
            for old, new in (("query_proj", "q_proj"), ("key_proj", "k_proj"), ("value_proj", "v_proj")):
                key = key.replace(f".attention.{old}.", f".attention.{new}.")
        else:
            for old, new in (("q_proj", "query_proj"), ("k_proj", "key_proj"), ("v_proj", "value_proj")):
                key = key.replace(f".attention.{old}.", f".attention.{new}.")
        return key
