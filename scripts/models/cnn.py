import os
import torch
from pathlib import Path
from functools import reduce
from collections import OrderedDict

import torch.nn.functional as F
from torch import nn

import base
from avalanche.models import MultiHeadClassifier, MultiTaskModule


class CustomCNN(base.BaseModel):
    """Build a simple CNN"""

    def __init__(
        self, num_classes, device, seed=42, output_hidden=False, multihead=False
    ):
        """
        Returns:
            Simple CNN model
        """
        in_features = 4608

        super().__init__(
            seed=seed,
            output_hidden=output_hidden,
            is_multihead=multihead,
            device=device,
            in_features=in_features,
            out_features=num_classes,
        )

        self._model = Net(in_features=in_features, num_classes=num_classes)
        self._hidden_layers = ["features.0", "features.1"]
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

    def _save_weights_impl(self, dir_name):
        # Check for directory
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        # Save state dictionary
        torch.save(self.model.state_dict(), f"{dir_name}/model.pt")

    def _load_weights_impl(self, dir_name):
        # Load model
        self.model.load_state_dict(
            torch.load(f"{dir_name}/model.pt", map_location=self.device.type)
        )
        self.model.to(self.device)

    def forward(self, x, task_labels=None):
        # Get dictionary
        out = self.model.forward(x)
        if self.is_multihead:
            # For multihead situation, must provide task labels!
            assert (
                task_labels != None
            ), "Failed to provide task labels for multihead classifier"

            # Reshape pooler output
            pooler_out = out["pooler_output"].view(out["pooler_output"].size(0), -1)

            # Feed to multihead classifier
            classifier_out = self.multihead_classifier(pooler_out, task_labels)

        else:
            classifier_out = out["logits"]

        if self.output_hidden:
            return classifier_out, out["hidden_states"]
        else:
            return classifier_out

    def get_hidden_layer(self, id):
        return None


class Net(nn.Module):
    def __init__(self, in_features, num_classes):
        super(Net, self).__init__()
        self.num_classes = num_classes

        self.features = nn.Sequential(
            nn.Sequential(
                OrderedDict(
                    [
                        ("convolution", nn.Conv2d(1, 16, kernel_size=3)),
                        ("activation", nn.ReLU()),
                    ]
                )
            ),
            nn.Sequential(
                OrderedDict(
                    [
                        ("convolution", nn.Conv2d(16, 32, kernel_size=3)),
                        ("activation", nn.ReLU()),
                        ("dropout", nn.Dropout2d()),
                    ]
                )
            ),
        )
        self.pooler = nn.MaxPool2d(2)
        self.classifier = nn.Sequential(nn.Linear(in_features, self.num_classes))

    def forward(self, x):
        stage_one = self.features[0](x)
        stage_two = self.features[1](stage_one)
        features_output = stage_two
        pooler_output = self.pooler(stage_two)
        flattened = pooler_output.view(pooler_output.size(0), -1)
        logits = self.classifier(flattened)
        return {
            "last_hidden_state": features_output,
            "pooler_output": pooler_output,
            "hidden_states": [stage_one, stage_two],
            "logits": logits,
        }
