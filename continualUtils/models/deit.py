from transformers import ViTConfig, ViTForImageClassification

from continualUtils.models import FrameworkClassificationModel


class PretrainedDeiT(FrameworkClassificationModel):
    """Pretrained DeiT from HuggingFace."""

    def __init__(
        self,
        deit: str,
        output_hidden: bool = False,
    ):
        _model = ViTForImageClassification.from_pretrained(deit)

        super().__init__(
            _model=_model,
            output_hidden=output_hidden,
            init_weights=False,
            patch_batch_norm=False,
            num_classes_per_task=1000,
            classifier_name=None,
            make_multihead=False,
        )


class PretrainedDeiTSmall(PretrainedDeiT):
    """
    Pretrained DeiT Small
    https://huggingface.co/facebook/deit-small-patch16-224
    """

    def __init__(
        self,
        output_hidden: bool = False,
    ):
        super().__init__(
            deit="facebook/deit-small-patch16-224",
            output_hidden=output_hidden,
        )


class CustomDeiT(FrameworkClassificationModel):
    """Custom DeiT built with HuggingFace."""

    def __init__(
        self,
        configuration: ViTConfig,
        num_classes_per_task: int,
        output_hidden: bool = False,
        init_weights: bool = False,
        make_multihead: bool = False,
    ):
        _model = ViTForImageClassification(configuration)
        classifier_name = "classifier"

        super().__init__(
            _model=_model,
            output_hidden=output_hidden,
            init_weights=init_weights,
            patch_batch_norm=False,
            num_classes_per_task=num_classes_per_task,
            classifier_name=classifier_name,
            make_multihead=make_multihead,
        )


class CustomDeiTSmall(CustomDeiT):
    """Custom DeiT Small"""

    def __init__(
        self,
        num_classes_per_task: int,
        output_hidden: bool = False,
        make_multihead: bool = False,
        init_weights: bool = False,
        image_size: int = 32,
    ):
        configuration = ViTConfig(
            num_labels=num_classes_per_task,
            hidden_size=384,
            num_hidden_layers=12,
            num_attention_heads=6,
            intermediate_size=1536,
            hidden_act="gelu",
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            image_size=image_size,
            patch_size=16,
            num_channel=3,
            qkv_bias=True,
            encoder_stride=16,
        )

        super().__init__(
            configuration=configuration,
            num_classes_per_task=num_classes_per_task,
            output_hidden=output_hidden,
            init_weights=init_weights,
            make_multihead=make_multihead,
        )


__all__ = ["PretrainedDeiTSmall", "CustomDeiTSmall"]
