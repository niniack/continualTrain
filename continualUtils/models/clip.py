from dataclasses import dataclass

import torch
from transformers import CLIPModel, CLIPProcessor

from continualUtils.models import FrameworkMultiModalModel


class PretrainedCLIP(FrameworkMultiModalModel):
    """Pretrained CLIP from HuggingFace."""

    def __init__(
        self,
        clip_model: torch.nn.Module,
        clip_processor: torch.nn.Module,
        output_hidden: bool = False,
    ):
        super().__init__(
            model=clip_model,
            processor=clip_processor,
            output_hidden=output_hidden,
        )


class PretrainedCLIP_ViT_Base_Patch32(PretrainedCLIP):
    def __init__(
        self,
        output_hidden: bool = False,
    ):
        _model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        _processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32"
        )

        super().__init__(
            clip_model=_model,
            clip_processor=_processor,
            output_hidden=output_hidden,
        )
