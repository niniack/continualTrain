import collections

import torch
import torch.nn as nn
import torch.nn.functional as F

from continualUtils.models import FrameworkClassificationModel


class _Net(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x, *args, **kwargs):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        # output = F.log_softmax(x, dim=1)
        return x


class SimpleMNISTCNN(FrameworkClassificationModel):
    def __init__(
        self,
        num_classes_per_task: int,
        output_hidden: bool = False,
        make_multihead: bool = False,
        init_weights: bool = False,
        patch_batch_norm: bool = False,
    ):
        _model = _Net(num_classes=num_classes_per_task)
        classifier_name = "classifier"

        super().__init__(
            _model=_model,
            num_classes_per_task=num_classes_per_task,
            output_hidden=output_hidden,
            init_weights=init_weights,
            make_multihead=make_multihead,
            patch_batch_norm=patch_batch_norm,
            classifier_name=classifier_name,
        )
