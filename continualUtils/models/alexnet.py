import collections
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from continualUtils.models import FrameworkClassificationModel


class _AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.conv_layer = nn.Sequential(
            # Conv Layer block 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=3, padding=1
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Conv Layer block 2
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=128, out_channels=128, kernel_size=3, padding=1
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),
            # Conv Layer block 3
            nn.Conv2d(
                in_channels=128, out_channels=256, kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, num_classes),
        )

    def forward(self, x, *args, **kwargs):
        """Perform forward."""

        # conv layers
        x = self.conv_layer(x)

        # flatten
        x = x.view(x.size(0), -1)

        # fc layer
        x = self.fc_layer(x)
        # print("fp = ", x)
        return x


class AlexNet(FrameworkClassificationModel):
    def __init__(
        self,
        num_classes_per_task: int,
        output_hidden: bool = False,
        make_multihead: bool = False,
        init_weights: bool = False,
        patch_batch_norm: bool = False,
    ):
        classifier_name = "fc_layer"
        model = _AlexNet(num_classes=num_classes_per_task)

        super().__init__(
            model=model,
            num_classes_per_task=num_classes_per_task,
            output_hidden=output_hidden,
            init_weights=init_weights,
            make_multihead=make_multihead,
            patch_batch_norm=patch_batch_norm,
            classifier_name=classifier_name,
        )
