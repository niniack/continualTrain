from typing import Optional

import torch
from transformers import ResNetForImageClassification

from continualUtils.models import HuggingFaceResNet


class PretrainedResNet34(HuggingFaceResNet):
    """Pretrained ResNet34 on Imagenet"""

    def __init__(
        self,
        device: torch.device,
        num_classes_per_head: Optional[int] = None,
        output_hidden: bool = False,
        multihead: bool = False,
        seed: int = 42,
    ):
        _model = ResNetForImageClassification.from_pretrained(
            "microsoft/resnet-34"
        ).to(device)

        super().__init__(
            seed=seed,
            device=device,
            model=_model,
            output_hidden=output_hidden,
            is_multihead=multihead,
            in_features=512,
            num_classes_total=1000,
            num_classes_per_head=num_classes_per_head,
            init_weights=False,
        )
        self._model = _model
        self._hidden_layers = []
        self._num_hidden = len(self.hidden_layers)

    @property
    def model(self):
        return self._model

    @property
    def hidden_layers(self):
        return self._hidden_layers

    @property
    def num_hidden(self):
        return self._num_hidden
