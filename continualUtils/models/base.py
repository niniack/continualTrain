import os
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Union

import safetensors
import torch
import torch.backends.cudnn
from avalanche.benchmarks import NCExperience
from avalanche.models import DynamicModule, MultiHeadClassifier, MultiTaskModule
from avalanche.models.utils import avalanche_forward
from functorch.experimental import replace_all_batch_norm_modules_
from torch import nn
from torch.nn.utils import parametrizations


class MissingTasksException(Exception):
    pass


class BaseModel(ABC, MultiTaskModule, DynamicModule):
    """Base model to inherit for continualTrain"""

    def __init__(
        self,
        seed: int,
        device: torch.device,
        model: torch.nn.Module,
        output_hidden: bool,
        is_multihead: bool,
        in_features: int,
        num_classes_total: int,
        num_classes_per_head: Optional[int] = None,
        init_weights: bool = False,
        patch_batch_norm: bool = True,
    ):
        super().__init__()
        self.seed = seed
        self.device = device
        self._model = model
        self.output_hidden = output_hidden
        self.is_multihead = is_multihead
        self.in_features = (in_features,)
        self.num_classes_total = num_classes_total
        self.num_classes_per_head = num_classes_per_head
        self.init_weights = init_weights
        self.patch_batch_norm = patch_batch_norm

        # Set seed for reproducibility
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # If using multi-GPU
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        if self.num_classes_per_head is None:
            self.num_classes_per_head = self.num_classes_total
        # Set classifier style
        if self.is_multihead:
            self.multihead_classifier = MultiHeadClassifier(
                in_features=in_features,
                initial_out_features=self.num_classes_per_head,
                masking=False,
            )
            self.multihead_classifier.to(device)

        # Update the module in-place to not use running stats
        # https://pytorch.org/functorch/stable/batch_norm.html
        # NOTE: Be careful with this, saliency maps require the patch
        if self.patch_batch_norm:
            self._patch_batch_norm()

        # Initialize weights with Kaiming init
        if self.init_weights:
            self._init_weights()

        self._freeze_backbone: bool = False

    def _init_weights(self):
        """
        Applies the Kaiming Normal initialization to all weights in the model.
        """

        # Set the seed for reproducibility
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_in", nonlinearity="relu"
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def toggle_hidden(self, output_hidden: bool):
        """Set whether model outputs hidden layers

        :param output_hidden: Flag for outputting hidden layers
        """
        self.output_hidden = output_hidden

    def adapt_model(self, experiences: Union[List[NCExperience], NCExperience]):
        if self.is_multihead:
            if isinstance(experiences, NCExperience):
                experiences = [experiences]
            for exp in experiences:
                self.multihead_classifier.adaptation(exp)

    def get_dir_name(self, parent_dir):
        # Build a consistent directory name
        return f"{parent_dir}/{self.__class__.__name__}_seed{self.seed}"

    def save_weights(self, parent_dir):
        # Get dir name
        dir_name = self.get_dir_name(parent_dir)

        # Call model specific implementation
        self._save_weights_impl(dir_name)

        # Handle multihead
        if self.is_multihead:
            path = Path(dir_name, "classifier.pt")
            torch.save(self.multihead_classifier.state_dict(), path)
            print(f"Classifier saved at: {path}")

    def load_weights(self, parent_dir):
        # Get dir name
        dir_name = self.get_dir_name(parent_dir)

        # Call model specific implementation
        self._load_weights_impl(dir_name)

        # Handle multihead
        if self.is_multihead:
            path = Path(dir_name, "classifier.pt")
            self.multihead_classifier.load_state_dict(
                torch.load(path, map_location=self.device.type), strict=False
            )
            self.multihead_classifier.to(self.device)

    @property
    @abstractmethod
    def model(self):
        pass

    @property
    @abstractmethod
    def hidden_layers(self):
        pass

    @property
    @abstractmethod
    def num_hidden(self):
        pass

    @abstractmethod
    def _save_weights_impl(self, dir_name):
        pass

    @abstractmethod
    def _load_weights_impl(self, dir_name):
        pass

    @abstractmethod
    def forward(self, x, task_labels):
        pass

    @abstractmethod
    def get_hidden_layer(self, *args, **kwargs):
        pass

    @abstractmethod
    def _patch_batch_norm(self):
        pass


class HuggingFaceResNet(BaseModel):
    def _save_weights_impl(self, dir_name):
        # Check if the model has a 'save_pretrained' method
        if hasattr(self._model, "save_pretrained"):
            # Create the directory if it doesn't exist
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)

            # Save the model
            self._model.save_pretrained(
                dir_name, state_dict=self._model.state_dict()
            )
            print(f"\nModel saved in directory: {dir_name}")
        else:
            print(
                "\nThe provided model does not have a 'save_pretrained' method."
            )

    def _load_weights_impl(self, dir_name):
        print(f"Loading from {dir_name}")

        # Construct the path to the .safetensors file
        file_path = os.path.join(dir_name, "model.safetensors")

        # Check if the .safetensors file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")

        # Load the model state dictionary using safeTensors
        state_dict = safetensors.torch.load_file(file_path)

        # Load the state_dict into the existing model architecture
        self._model.load_state_dict(state_dict, strict=True)

        # Move the model to the desired device
        self._model = self._model.to(self.device)

    def freeze_backbone(self, flag: bool):
        self._freeze_backbone = flag

    def get_hidden_layer(self, id):
        raise NotImplementedError("To Do!")

    def forward(self, x, task_labels=None):
        if self.is_multihead:
            if task_labels is None:
                warnings.warn(
                    "Task labels were not provided. Running forward on all tasks."
                )

            with torch.set_grad_enabled(not self._freeze_backbone):
                out = self._model.resnet(
                    x, output_hidden_states=self.output_hidden, return_dict=True
                )

            # Reshape pooler output
            flat_pooler_out = out.pooler_output.view(
                out.pooler_output.size(0), -1
            )

            # Feed to multihead classifier
            classifier_out = self.multihead_classifier(
                flat_pooler_out, task_labels
            )
        else:
            with torch.set_grad_enabled(not self._freeze_backbone):
                out = self._model(
                    x, output_hidden_states=self.output_hidden, return_dict=True
                )
            classifier_out = out.logits

        if self.output_hidden:
            return classifier_out, out.hidden_states
        else:
            return classifier_out

    def _patch_batch_norm(self):
        """
        Replace all BatchNorm modules with GroupNorm and
        apply weight normalization to all Conv2d layers.
        """

        def replace_bn_with_gn(module, module_path=""):
            for child_name, child_module in module.named_children():
                child_path = (
                    f"{module_path}.{child_name}" if module_path else child_name
                )

                if isinstance(child_module, nn.BatchNorm2d):
                    new_groupnorm = nn.GroupNorm(32, child_module.num_features)
                    setattr(module, child_name, new_groupnorm)

                else:
                    replace_bn_with_gn(child_module, child_path)

        # Apply the replacement function to the model
        replace_bn_with_gn(self._model)

        # Move to device
        self._model.to(self.device)
