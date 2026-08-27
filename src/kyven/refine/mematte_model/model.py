"""Inference-only assembly of the official MEMatte ViT-S architecture."""

from __future__ import annotations

from functools import partial

import torch
from torch import nn
from torch.nn import functional as F

from kyven.refine.mematte_model.detail_capture import Detail_Capture
from kyven.refine.mematte_model.vit import ViT


class MEMatteInference(nn.Module):
    """MEMatte ViT-S without training-only teacher and criterion modules."""

    def __init__(self, max_number_token: int = 18_500, variant: str = "small") -> None:
        super().__init__()
        if variant not in ("small", "base"):
            raise ValueError(f"Unsupported MEMatte variant: {variant}")
        embed_dim = 768 if variant == "base" else 384
        num_heads = 12 if variant == "base" else 6
        self.backbone = ViT(
            in_chans=4,
            img_size=512,
            patch_size=16,
            embed_dim=embed_dim,
            depth=12,
            num_heads=num_heads,
            drop_path_rate=0,
            window_size=14,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            window_block_indexes=(0, 1, 3, 4, 6, 7, 9, 10),
            residual_block_indexes=(2, 5, 8, 11),
            use_rel_pos=True,
            out_feature="last_feat",
            topk=0.25,
            multi_score=True,
            max_number_token=max_number_token,
        )
        self.decoder = Detail_Capture(in_chans=embed_dim)
        self.register_buffer(
            "pixel_mean",
            torch.tensor((123.675 / 255.0, 116.280 / 255.0, 103.530 / 255.0)).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "pixel_std",
            torch.tensor((58.395 / 255.0, 57.120 / 255.0, 57.375 / 255.0)).view(1, 3, 1, 1),
            persistent=False,
        )

    def forward(self, image: torch.Tensor, trimap: torch.Tensor) -> torch.Tensor:
        height, width = image.shape[-2:]
        image = (image - self.pixel_mean) / self.pixel_std
        inputs = torch.cat((image, trimap), dim=1)
        pad_height = (-height) % 32
        pad_width = (-width) % 32
        if pad_height or pad_width:
            inputs = F.pad(inputs, (0, pad_width, 0, pad_height))
        features, _, _ = self.backbone(inputs)
        alpha = self.decoder(features, inputs)["phas"]
        return alpha[:, :, :height, :width]
