import os
from abc import ABC, abstractmethod
from typing import List, Optional, Union

import safetensors
import torch
import torch.backends.cudnn
import torch.nn.functional as F
from avalanche.benchmarks import NCExperience
from avalanche.models import MultiTaskModule
from torch import nn

from continualUtils.models.utils import as_multitask


class BaseModel(ABC, torch.nn.Module):
    """Base model to be extended for continualTrain"""

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        output_hidden: bool,
        num_classes_per_task: int,
        init_weights: bool = False,
        patch_batch_norm: bool = True,
    ) -> None:
        super().__init__()
        self.model = model
        self.device = device
        self.output_hidden = output_hidden
        self.num_classes_per_task = num_classes_per_task
        self.init_weights = init_weights
        self.patch_batch_norm = patch_batch_norm

        # Update the module in-place to not use running stats
        # https://pytorch.org/functorch/stable/batch_norm.html
        # NOTE: Be careful with this, saliency maps require the patch
        if self.patch_batch_norm:
            self._patch_batch_norm()

        # Initialize weights with Kaiming init
        if self.init_weights:
            self._init_weights()

        self.model.to(device)

    def _init_weights(self) -> None:
        """
        Applies the Kaiming Normal initialization to all weights in the model.
        """

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_in", nonlinearity="relu"
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _patch_batch_norm(self) -> None:
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
        replace_bn_with_gn(self.model)

        # Move to device
        self.model.to(self.device)

    def _get_dir_name(self, parent_dir: str) -> None:
        """Get a directory name for consistency"""
        # Build a consistent directory name
        return f"{parent_dir}/{self.__class__.__name__}"

    def set_output_hidden(self, flag: bool) -> None:
        """Set whether model outputs hidden layers.
        Only relevant for HuggingFace models with this option.

        :param output_hidden: Flag for outputting hidden layers
        """
        self.output_hidden = flag

    def save_weights(self, parent_dir: str) -> None:
        """Save model weights.

        :param parent_dir: Directory to save the model weights
        """

        # Get dir name
        dir_name = self._get_dir_name(parent_dir)

        # Call model specific implementation
        self._save_weights_impl(dir_name)

    def load_weights(self, parent_dir: str) -> None:
        """Load model weights.

        :param parent_dir: Directory to find the model weights
        """
        # Get dir name
        dir_name = self._get_dir_name(parent_dir)

        # Call model specific implementation
        self._load_weights_impl(dir_name)

    def is_multihead(self):
        """Returns True if the model is a multihead model."""
        return isinstance(self.model, MultiTaskModule)

    def adapt_model(self, experiences: Union[List[NCExperience], NCExperience]):
        """Add task parameters to a model"""
        if self.is_multihead():
            if isinstance(experiences, NCExperience):
                experiences = [experiences]
            for exp in experiences:
                self.multihead_classifier.adaptation(exp)

    @abstractmethod
    def _save_weights_impl(self, dir_name: str) -> None:
        pass

    @abstractmethod
    def _load_weights_impl(self, dir_name: str) -> None:
        pass

    @abstractmethod
    def forward(
        self, x: torch.Tensor, task_labels: Optional[torch.Tensor] = None
    ) -> None:
        """Forward pass for the model

        :param x: _description_
        :param task_labels: _description_
        """
        pass


class FrameworkClassificationModel(BaseModel):
    """Extend the base model to make it usable with continualTrain.
    Classification models should inherit from this."""

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        num_classes_per_task: int,
        output_hidden: bool = False,
        init_weights: bool = True,
        make_multihead: bool = False,
        classifier_name: Optional[str] = None,
        patch_batch_norm: bool = True,
    ) -> None:
        super().__init__(
            device=device,
            model=model,
            output_hidden=output_hidden,
            num_classes_per_task=num_classes_per_task,
            init_weights=init_weights,
            patch_batch_norm=patch_batch_norm,
        )

        if make_multihead:
            if classifier_name is None:
                raise ValueError(
                    "A classifier name must be provided to build a MultiTask module."
                )
            self.model = as_multitask(model, classifier_name)
        else:
            self.model = model

    def _save_weights_impl(self, dir_name):
        # Create the directory if it doesn't exist
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        # Check if the model has a 'save_pretrained' method
        if hasattr(self.model, "save_pretrained"):
            # Save the model
            self.model.save_pretrained(
                dir_name, state_dict=self.model.state_dict()
            )

        else:
            # Construct the path to save the .safetensors file
            file_path = os.path.join(dir_name, "model.safetensors")

            # Save the model state dictionary using safeTensors
            safetensors.torch.save_model(self.model, file_path)

        print(f"\nModel saved in directory: {dir_name}")

    def _load_weights_impl(self, dir_name):
        print(f"Loading model from {dir_name}")

        # Path for the safetensors file
        safetensors_file = os.path.join(dir_name, "model.safetensors")

        if not os.path.exists(safetensors_file):
            raise FileNotFoundError(
                f"The file {safetensors_file} does not exist."
            )

        try:
            # Try loading the state_dict using safetensors
            state_dict = safetensors.torch.load_file(
                safetensors_file, device=str(self.device)
            )
            self.model.load_state_dict(state_dict, strict=True)
            print(f"Model state dictionary loaded from {safetensors_file}")
        except:
            pass

        try:
            # Try loading the entire model using safetensors
            missing, unexpected = safetensors.torch.load_model(
                self.model, safetensors_file, strict=False
            )
            print(
                f"""Entire model loaded from {safetensors_file},
                missing {missing} and unexpected {unexpected}
                """
            )
        except:
            pass

        raise FileExistsError("Failed to load the entire model")

    def forward(
        self, x: torch.Tensor, task_labels: Optional[torch.Tensor] = None
    ) -> None:
        """Forward pass for the model

        :param x: _description_
        :param task_labels: _description_, defaults to None
        """

        if not self.is_multihead() and task_labels:
            raise ValueError(
                """
                This is not a multihead module. Check if task_labels 
                are needed. If so, the model was initialized incorrectly.
                """
            )

        # Create a dictionary for the arguments
        args = {
            "output_hidden_states": self.output_hidden,
            "return_dict": False,
        }

        # Add the optional argument if the condition is met
        if self.is_multihead():
            args["task_labels"] = task_labels
            out, hidden_states = self.model(x, **args)
        else:
            out = self.model(x, **args)
            hidden_states = None
            if isinstance(out, tuple):
                hidden_states = out[1] if len(out) > 1 else None
                out = out[0]
            elif isinstance(out, dict) and "logits" in out:
                out = out.get("hidden_states", None)
                out = out["logits"]

        if self.output_hidden:
            return out, hidden_states

        return out
