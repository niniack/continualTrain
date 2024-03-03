from transformers import CvtConfig, CvtForImageClassification

from continualUtils.models import FrameworkClassificationModel


class PretrainedCvt(FrameworkClassificationModel):
    """Pretrained Cvt from HuggingFace."""

    def __init__(
        self,
        cvt: str,
        output_hidden: bool = False,
    ):
        _model = CvtForImageClassification.from_pretrained(cvt)

        super().__init__(
            _model=_model,
            output_hidden=output_hidden,
            init_weights=False,
            patch_batch_norm=False,
            num_classes_per_task=1000,
            classifier_name=None,
            make_multihead=False,
        )


class PretrainedCvt13(PretrainedCvt):
    """
    Pretrained Cvt 13
    https://huggingface.co/microsoft/cvt-13
    """

    def __init__(
        self,
        output_hidden: bool = False,
    ):
        super().__init__(
            cvt="microsoft/cvt-13",
            output_hidden=output_hidden,
        )


class CustomCvt(FrameworkClassificationModel):
    """Custom Cvt built with HuggingFace."""

    def __init__(
        self,
        configuration: CvtConfig,
        num_classes_per_task: int,
        output_hidden: bool = False,
        init_weights: bool = False,
        make_multihead: bool = False,
    ):
        _model = CvtForImageClassification(configuration)
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


class CustomCvt13(CustomCvt):
    """Custom Cvt 13"""

    def __init__(
        self,
        num_classes_per_task: int,
        output_hidden: bool = False,
        make_multihead: bool = False,
        init_weights: bool = False,
    ):
        configuration = CvtConfig(
            num_labels=num_classes_per_task,
        )

        super().__init__(
            configuration=configuration,
            num_classes_per_task=num_classes_per_task,
            output_hidden=output_hidden,
            init_weights=init_weights,
            make_multihead=make_multihead,
        )


__all__ = ["PretrainedCvt13", "CustomCvt13"]
