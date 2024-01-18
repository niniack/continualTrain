import collections

import torch
import torch.nn as nn
import torch.nn.functional as F

from continualUtils.models import FrameworkClassificationModel


class _SimpleMNISTCNN(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.classifier = nn.Sequential(nn.Linear(128, num_classes))
        self._model = nn.Sequential(
            collections.OrderedDict(
                [
                    (
                        "backbone",
                        nn.Sequential(
                            nn.Conv2d(1, 32, 3, 1),
                            nn.ReLU(),
                            nn.Conv2d(32, 64, 3, 1),
                            nn.ReLU(),
                            nn.MaxPool2d(2),
                            nn.Dropout(0.25),
                            nn.Flatten(start_dim=1, end_dim=-1),
                            nn.Linear(9216, 128),
                            nn.ReLU(),
                            nn.Dropout(0.5),
                        ),
                    ),
                    ("classifier", self.classifier),
                ]
            )
        )

    def forward(self, x, *args, **kwargs):
        return self._model(x)


class SimpleMNISTCNN(FrameworkClassificationModel):
    def __init__(
        self,
        device: torch.device,
        num_classes_per_task: int,
        output_hidden: bool = False,
        make_multihead: bool = False,
    ):
        _model = _SimpleMNISTCNN(num_classes=num_classes_per_task)
        classifier_name = "classifier"

        super().__init__(
            model=_model,
            device=device,
            num_classes_per_task=num_classes_per_task,
            output_hidden=output_hidden,
            init_weights=True,
            make_multihead=make_multihead,
            classifier_name=classifier_name,
        )
