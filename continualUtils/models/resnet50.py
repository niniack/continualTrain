from typing import Optional

import torch
from transformers import ResNetConfig, ResNetForImageClassification

from continualUtils.models import HuggingFaceResNet


class PretrainedResNet50(HuggingFaceResNet):
    """Pretrained ResNet34 on Imagenet"""

    def __init__(
        self,
        device: torch.device,
        num_classes_per_head: Optional[int] = None,
        output_hidden: bool = False,
        multihead: bool = False,
        seed: int = 42,
        **kwargs
    ):
        _model = ResNetForImageClassification.from_pretrained(
            "microsoft/resnet-50"
        ).to(device)

        super().__init__(
            seed=seed,
            device=device,
            model=_model,
            output_hidden=output_hidden,
            is_multihead=multihead,
            in_features=2048,
            num_classes_total=1000,
            num_classes_per_head=num_classes_per_head,
            init_weights=False,
            **kwargs
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


class CustomResNet50(HuggingFaceResNet):
    def __init__(
        self,
        device: torch.device,
        num_classes_total: int,
        num_classes_per_head: Optional[int] = None,
        output_hidden: bool = False,
        multihead: bool = False,
        seed: int = 42,
        **kwargs
    ):
        """
        Returns:
            Resnet50 model
        """

        # Initializing a model (with random weights) from
        # the resnet-50 style configuration
        configuration = ResNetConfig(
            num_channels=3,
            embedding_size=64,
            hidden_sizes=[256, 512, 1024, 2048],
            depths=[3, 4, 6, 3],
            layer_type="bottleneck",
            hidden_act="relu",
            downsample_in_first_stage=False,
        )

        _model = ResNetForImageClassification(configuration).to(device)  # type: ignore

        super().__init__(
            seed=seed,
            device=device,
            model=_model,
            output_hidden=output_hidden,
            is_multihead=multihead,
            in_features=2048,
            num_classes_total=num_classes_total,
            num_classes_per_head=num_classes_per_head,
            init_weights=True,
            **kwargs
        )

        self._model = _model
        self._hidden_layers = [
            "resnet.embedder",
            "resnet.encoder.stages.0",
            "resnet.encoder.stages.1",
            "resnet.encoder.stages.2",
            "resnet.encoder.stages.3",
        ]
        self._num_hidden = len(self.hidden_layers)

    @property
    def model(self) -> ResNetForImageClassification:
        return self._model  # type: ignore

    @property
    def hidden_layers(self) -> list:
        return self._hidden_layers

    @property
    def num_hidden(self) -> int:
        return self._num_hidden
