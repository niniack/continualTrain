import inspect
from typing import Callable, Optional, Tuple, Union

import torch

INPUT_IDX = 0
TASK_IDX = 1
TARGET_IDX = 2
MODEL_IDX = 3
LAYER_IDX = 4
ACTIVATIONS_IDX = 5


def get_layers_from_model(model: torch.nn.Module, module_name: str):
    """
    Retrieve a specific module from a PyTorch model by its name.

    :param model: The PyTorch model.
    :param module_name: The name of the module to retrieve.
    :return: The requested module, if found.
    """
    for name, module in model.named_modules():
        if name == module_name:
            return module
    raise AttributeError(f"Module '{module_name}' not found in the model.")


def check_pure_function(func):
    """Checks whether pure function requires the arguments
    inputs, model, and targets

    :param func: pure function to check
    :return: _description_
    """
    signature = inspect.signature(func)
    parameters = list(signature.parameters.keys())

    required_args = {"x", "task", "y", "model"}
    return required_args.issubset(parameters)


def compute_score(
    x: torch.Tensor,
    task: int,
    y: torch.Tensor,
    model: torch.nn.Module,
    delta_intermediate: Optional[torch.Tensor] = None,
    layer_name: Optional[str] = None,
) -> Union[Tuple[torch.Tensor, dict], torch.Tensor]:
    """
    Since vmap will unbatch and vectorize the computation, we
    assume that all the inputs do not have a batch dimension.
    """

    # Batch
    x = x.unsqueeze(0)
    y = y.unsqueeze(0)
    act_container = dict()

    if delta_intermediate is not None and layer_name is not None:
        # Get layer, hook layer, forward
        layer = get_layers_from_model(model, layer_name)
        hook = layer.register_forward_hook(
            get_activation(act_container, layer_name)
        )
        output = model(x, task)
        activation = act_container[layer_name].squeeze(0)
        activation = activation + delta_intermediate
        hook.remove()

        if output.shape != y.shape:
            raise OneHotException(
                "The model outputs must be the shape as the target."
            )

        score = torch.sum(output * y)
        return score, act_container

    else:
        # Simple forward
        output = model(x, task)

        if output.shape != y.shape:
            raise OneHotException(
                "The model outputs must be the shape as the target."
            )

        score = torch.sum(output * y)
        return score


def get_activation(container, key) -> Callable:
    """Store activation in container with key

    :param container: Mutable container
    :param key: Key to store with inside the container
    :return: Callable hook to be registered with torch
    """

    # the hook signature
    def hook(model, input, output):
        container[key] = output

    return hook


class OneHotException(Exception):
    """Raised when there was a one hot target expected"""

    def __init__(self, message):
        super().__init__(message)
