from dataclasses import dataclass

import torch
from transformers import CLIPModel, CLIPProcessor

from continualUtils.models import FrameworkClassificationModel


class PretrainedDeiT(FrameworkClassificationModel):
    """Pretrained DeiT from HuggingFace."""

    def __init__(
        self,
        deit: torch.nn.Module,
        output_hidden: bool = False,
    ):
        super().__init__(
            _model=deit,
            output_hidden=output_hidden,
            num_classes_per_task=1000,
        )


class PretrainedDeiTSmall(PretrainedDeiT):
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
