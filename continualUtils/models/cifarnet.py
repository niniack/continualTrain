import torch.nn as nn

from continualUtils.models import FrameworkClassificationModel


class _CIFARNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.conv_layer = nn.Sequential(
            # Conv Layer block 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=32, out_channels=32, kernel_size=3, padding=1
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.25),
            # Conv Layer block 2
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=3, padding=1
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, padding=1
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.25),
            # Conv Layer block 3
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=3, padding=1
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=128, out_channels=128, kernel_size=3, padding=1
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.25),
        )

        self.dense = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.50),
        )
        self.fc_layer = nn.Linear(256, num_classes)

    def forward(self, x, *args, **kwargs):
        """Perform forward."""

        # conv layers
        x = self.conv_layer(x)

        # flatten
        x = x.view(x.size(0), -1)

        # dense
        x = self.dense(x)

        # fc layer
        x = self.fc_layer(x)
        # print("fp = ", x)
        return x


class CIFARNet(FrameworkClassificationModel):
    def __init__(
        self,
        num_classes_per_task: int,
        output_hidden: bool = False,
        make_multihead: bool = False,
        init_weights: bool = False,
        patch_batch_norm: bool = False,
    ):
        _model = _CIFARNet(num_classes=num_classes_per_task)
        classifier_name = "fc_layer"

        super().__init__(
            _model=_model,
            num_classes_per_task=num_classes_per_task,
            output_hidden=output_hidden,
            init_weights=init_weights,
            make_multihead=make_multihead,
            patch_batch_norm=patch_batch_norm,
            classifier_name=classifier_name,
        )


__all__ = ["CIFARNet"]
