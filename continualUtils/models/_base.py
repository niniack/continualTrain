import functools
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, is_dataclass
from typing import List, Optional, Union

import safetensors
import torch
import torch.backends.cudnn
from avalanche.benchmarks import NCExperience
from avalanche.models import MultiTaskModule
from torch import nn

from continualUtils.models.utils import as_multitask


def super_init_wrapper(cls):
    """
    Decorator for a dataclass to ensure super().__init__() is called in the generated __init__.
    """
    if not is_dataclass(cls):
        raise TypeError("super_init_wrapper can only be used on dataclasses")

    original_init = cls.__init__

    @functools.wraps(original_init)
    def new_init(self, *args, **kwargs):
        super(cls, self).__init__()
        original_init(self, *args, **kwargs)

    cls.__init__ = new_init
    return cls


@super_init_wrapper
@dataclass(init=False, kw_only=True, eq=False)
class BaseModel(ABC, torch.nn.Module):
    """Base model to be extended for continualTrain"""

    _model: torch.nn.Module
    output_hidden: bool = False
    init_weights: bool = False
    patch_batch_norm: bool = False

    def __post_init__(self) -> None:
        if self.patch_batch_norm:
            self._patch_batch_norm()

        if self.init_weights:
            self._init_weights()

    def _init_weights(self) -> None:
        """
        Applies the Kaiming Normal initialization to all weights in the model.
        """
        print("Initializing weights...")
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_in", nonlinearity="relu"
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _patch_batch_norm(self) -> None:
        """
        Replace all BatchNorm modules
        """
        print("Patching batch norm...")

        def replace_bn(module, module_path=""):
            for child_name, child_module in module.named_children():
                child_path = (
                    f"{module_path}.{child_name}" if module_path else child_name
                )

                if isinstance(child_module, nn.BatchNorm2d):
                    new_groupnorm = nn.GroupNorm(32, child_module.num_features)
                    setattr(module, child_name, new_groupnorm)

                else:
                    replace_bn(child_module, child_path)

        # Apply the replacement function to the model
        replace_bn(self._model)

    def _get_dir_name(self, parent_dir: str) -> str:
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

    def load_weights(self, parent_dir: str, device) -> None:
        """Load model weights.

        :param parent_dir: Directory to find the model weights
        """
        # Get dir name
        dir_name = self._get_dir_name(parent_dir)

        # Call model specific implementation
        self._load_weights_impl(dir_name, device)

    @abstractmethod
    def _save_weights_impl(self, dir_name: str) -> None:
        pass

    @abstractmethod
    def _load_weights_impl(self, dir_name: str, device) -> None:
        pass

    @abstractmethod
    def forward(
        self, x: torch.Tensor, task_labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass for the model

        :param x: _description_
        :param task_labels: _description_
        """
        pass


@dataclass(init=False, kw_only=True, eq=False)
class FrameworkModel(BaseModel):
    """Extends the base model to add in saving/loading methods continualTrain."""

    def _save_weights_impl(self, dir_name) -> None:
        # Create the directory if it doesn't exist
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        # Check if the model has a 'save_pretrained' method
        if hasattr(self._model, "save_pretrained"):
            # Save the model
            self._model.save_pretrained(
                dir_name, state_dict=self._model.state_dict()
            )

        else:
            # Construct the path to save the .safetensors file
            file_path = os.path.join(dir_name, "model.safetensors")

            # Save the model state dictionary using safeTensors
            safetensors.torch.save_model(self._model, file_path)

        print(f"\nModel saved in directory: {dir_name}")

    def _load_weights_impl(self, dir_name, device) -> None:
        print(f"Loading model from {dir_name}")

        # Path for the safetensors file
        safetensors_file = os.path.join(dir_name, "model.safetensors")

        if not os.path.exists(safetensors_file):
            raise FileNotFoundError(
                f"The file {safetensors_file} does not exist."
            )

        # Try to load the state with safetensors
        # Uses strict mode
        try:
            # Try loading the state_dict using safetensors
            state_dict = safetensors.torch.load_file(
                safetensors_file, device=device
            )
            self._model.load_state_dict(state_dict, strict=True)
            print(f"Model state dictionary loaded from {safetensors_file}")
        except:
            pass

        # Try to load the entire model with safetensors
        # Does not use strict mode
        try:
            # Try loading the entire model using safetensors
            missing, unexpected = safetensors.torch.load_model(
                self._model, safetensors_file, strict=False
            )
            self._model.to(device)
            print(
                f"""Entire model loaded from {safetensors_file},
                missing {missing} and unexpected {unexpected}
                """
            )
        except:
            raise ValueError("Failed to load the model")


@super_init_wrapper
@dataclass(kw_only=True, eq=False)
class FrameworkClassificationModel(FrameworkModel, MultiTaskModule):
    """Extends the framework model to make it usable with continualTrain.
    Classification models should inherit from this."""

    num_classes_per_task: int
    classifier_name: Optional[str] = None
    make_multihead: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.make_multihead:
            if self.classifier_name is None:
                raise ValueError(
                    "A classifier name must be provided to build a MultiTask module."
                )
            self._model = as_multitask(self._model, self.classifier_name)

    def is_multihead(self) -> None:
        """Returns True if the model is a multihead model."""
        return isinstance(self._model, MultiTaskModule)

    def adapt_model(
        self, experiences: Union[List[NCExperience], NCExperience]
    ) -> None:
        """Add task parameters to a model"""
        if self.is_multihead():
            if isinstance(experiences, NCExperience):
                experiences = [experiences]
            for exp in experiences:
                self._model.classifier.adaptation(exp)

    def forward(
        self, x: torch.Tensor, task_labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass for the model

        :param x: _description_
        :param task_labels: _description_, defaults to None
        """

        ######### Unnecessary error #########
        # if not self.is_multihead() and task_labels:
        #     raise ValueError(
        #         """
        #         This is not a multihead module. Check if task_labels
        #         are needed. If so, the model was initialized incorrectly.
        #         """
        #     )

        # Create a dictionary for the arguments
        args = {
            "output_hidden_states": self.output_hidden,
            "return_dict": False,
        }

        # Add the optional argument if the condition is met
        if self.is_multihead():
            args["task_labels"] = task_labels
            out = self._model(x, **args)
        else:
            out = self._model(x, **args)
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


@super_init_wrapper
@dataclass(kw_only=True)
class FrameworkMultiModalModel(FrameworkModel):
    """Extends the framework model to make it usable with continualTrain.
    Multimodal models should inherit from this."""

    processor: torch.nn.Module

    def __post_init__(self) -> None:
        super().__post_init__()

    def forward(
        self,
        images: torch.Tensor,
        text: List[str],
        task_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for the model

        :param images: _description_
        :param text: _description_
        :param task_labels: _description_, defaults to None
        :return: _description_
        """

        inputs = self.processor(
            text=text,
            images=images,
            return_tensors="pt",
            padding=True,
        )

        # Create a dictionary for the arguments
        args = {
            "output_hidden_states": self.output_hidden,
            "return_dict": False,
        }

        out = self._model(**inputs, **args)
        hidden_states = None
        if isinstance(out, tuple):
            out = out[0]
        elif isinstance(out, dict) and "logits" in out:
            out = out.get("hidden_states", None)
            out = out["logits_per_image"]

        if self.output_hidden:
            raise NotImplementedError(
                "Outputting hidden states not yet implemented"
            )
        return out


class FrameworkClassificationInterface(FrameworkClassificationModel):
    def __init__(
        self,
        num_classes_per_task: int,
        output_hidden: bool = False,
        make_multihead: bool = False,
        init_weights: bool = False,
    ):
        raise NotImplementedError()


__all__ = [
    "FrameworkMultiModalModel",
    "FrameworkClassificationModel",
    "FrameworkClassificationInterface",
]
