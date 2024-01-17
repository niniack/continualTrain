from typing import Optional

import torch
from transformers import ResNetConfig, ResNetForImageClassification

from continualUtils.models import HuggingFaceResNet


class PretrainedResNet18(HuggingFaceResNet):
    """Pretrained ResNet18 on Imagenet"""

    def __init__(
        self,
        device: torch.device,
        num_classes_per_head: Optional[int] = None,
        output_hidden: bool = False,
        multihead: bool = False,
        seed: int = 42,
    ):
        _model = ResNetForImageClassification.from_pretrained(
            "microsoft/resnet-18"
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


class CustomResNet18(HuggingFaceResNet):
    """Build a Resnet 18 model as
    described in https://arxiv.org/pdf/2007.07400.pdf
    """

    def __init__(
        self,
        device: torch.device,
        num_classes_total: int,
        num_classes_per_head: Optional[int] = None,
        output_hidden: bool = False,
        multihead: bool = False,
        seed: int = 42,
    ):
        """
        Returns:
            Resnet18 model
        """

        # Initializing a model (with random weights) from
        # the resnet-50 style configuration
        configuration = ResNetConfig(
            num_channels=3,
            embedding_size=32,
            hidden_sizes=[32, 64, 128, 256],
            depths=[2, 2, 2, 2],
            layer_type="basic",
            hidden_act="relu",
            downsample_in_first_stage=True,
        )

        _model = ResNetForImageClassification(configuration).to(device)

        super().__init__(
            seed=seed,
            device=device,
            model=_model,
            output_hidden=output_hidden,
            is_multihead=multihead,
            in_features=256,
            num_classes_total=num_classes_total,
            num_classes_per_head=num_classes_per_head,
            init_weights=True,
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
    def model(self):
        return self._model

    @property
    def hidden_layers(self):
        return self._hidden_layers

    @property
    def num_hidden(self):
        return self._num_hidden
