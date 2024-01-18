import torch
from transformers import (
    DeiTConfig,
    DeiTForImageClassification,
    ViTForImageClassification,
)

from continualUtils.models import FrameworkClassificationModel


class PretrainedDeiT(FrameworkClassificationModel):
    """Pretrained DeiT on Imagenet from HuggingFace.
    It is assumed to be limited to 1K classes.
    """

    def __init__(
        self,
        device: torch.device,
        deit: str,
        output_hidden: bool = False,
    ):
        _model = ViTForImageClassification.from_pretrained(deit)

        super().__init__(
            model=_model,
            device=device,
            num_classes_per_task=1000,
            output_hidden=output_hidden,
            init_weights=False,
        )


class PretrainedDeiTSmall(PretrainedDeiT):
    """Pretrained DeiTSmall"""

    def __init__(
        self,
        device: torch.device,
        output_hidden: bool = False,
    ):
        super().__init__(
            device=device,
            deit="facebook/deit-small-patch16-224",
            output_hidden=output_hidden,
        )
