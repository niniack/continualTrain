from dataclasses import dataclass

import torch
from torchvision.models import resnet18
from transformers import ResNetConfig, ResNetForImageClassification

from continualUtils.models import FrameworkClassificationModel


class PretrainedResNet(FrameworkClassificationModel):
    """Pretrained ResNet on Imagenet from HuggingFace.
    It is assumed to be limited to 1K classes.
    """

    def __init__(
        self,
        device: torch.device,
        resnet: str,
        output_hidden: bool = False,
    ):
        _model = ResNetForImageClassification.from_pretrained(resnet)

        super().__init__(
            model=_model,
            device=device,
            output_hidden=output_hidden,
            num_classes_per_task=1000,
        )


class PretrainedResNet18(PretrainedResNet):
    """Pretrained ResNet18"""

    def __init__(
        self,
        output_hidden: bool = False,
    ):
        super().__init__(
            resnet="microsoft/resnet-18",
            output_hidden=output_hidden,
        )


class PretrainedResNet34(PretrainedResNet):
    """Pretrained ResNet34"""

    def __init__(
        self,
        output_hidden: bool = False,
    ):
        super().__init__(
            resnet="microsoft/resnet-34",
            output_hidden=output_hidden,
        )


class PretrainedResNet50(PretrainedResNet):
    """Pretrained ResNet50"""

    def __init__(
        self,
        output_hidden: bool = False,
    ):
        super().__init__(
            device=device,
            resnet="microsoft/resnet-50",
            output_hidden=output_hidden,
        )


class CustomResNet(FrameworkClassificationModel):
    """Custom ResNet built with HuggingFace."""

    def __init__(
        self,
        configuration: ResNetConfig,
        num_classes_per_task: int,
        output_hidden: bool = False,
        init_weights: bool = False,
        patch_batch_norm: bool = True,
        make_multihead: bool = False,
    ):
        _model = ResNetForImageClassification(configuration)
        classifier_name = "classifier"

        super().__init__(
            model=_model,
            output_hidden=output_hidden,
            init_weights=init_weights,
            patch_batch_norm=patch_batch_norm,
            num_classes_per_task=num_classes_per_task,
            classifier_name=classifier_name,
            make_multihead=make_multihead,
        )

        self._hidden_layers = [
            "resnet.embedder",
            "resnet.encoder.stages.0",
            "resnet.encoder.stages.1",
            "resnet.encoder.stages.2",
            "resnet.encoder.stages.3",
        ]


class CustomResNet18(CustomResNet):
    """Resnet 18 model"""

    def __init__(
        self,
        num_classes_per_task: int,
        output_hidden: bool = False,
        make_multihead: bool = False,
        init_weights: bool = False,
        patch_batch_norm: bool = False,
    ):
        # Initializing a model (with random weights) from
        # the resnet-18 style configuration
        configuration = ResNetConfig(
            num_channels=3,
            embedding_size=64,
            hidden_sizes=[64, 128, 256, 512],
            depths=[2, 2, 2, 2],
            layer_type="basic",
            hidden_act="relu",
            downsample_in_first_stage=False,
            num_labels=num_classes_per_task,
        )

        super().__init__(
            configuration=configuration,
            num_classes_per_task=num_classes_per_task,
            output_hidden=output_hidden,
            init_weights=init_weights,
            patch_batch_norm=patch_batch_norm,
            make_multihead=make_multihead,
        )


class CustomResNet50(CustomResNet):
    """ResNet50 Model"""

    def __init__(
        self,
        num_classes_per_task: int,
        output_hidden: bool = False,
        make_multihead: bool = False,
        init_weights: bool = False,
        patch_batch_norm: bool = False,
    ):
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
            num_labels=num_classes_per_task,
        )

        super().__init__(
            configuration=configuration,
            num_classes_per_task=num_classes_per_task,
            output_hidden=output_hidden,
            init_weights=init_weights,
            patch_batch_norm=patch_batch_norm,
            make_multihead=make_multihead,
        )
