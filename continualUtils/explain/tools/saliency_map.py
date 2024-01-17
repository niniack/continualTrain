from typing import Callable, List, Optional

import torch
import torch.nn.functional as F
from torch.func import grad, vmap
from torchvision.transforms.functional import gaussian_blur

from continualUtils.explain.tools.utils import INPUT_IDX, check_pure_function


def compute_saliency_map(
    pure_function: Callable,
    model: torch.nn.Module,
    inputs: torch.Tensor,
    tasks: torch.Tensor,
    targets: torch.Tensor,
    grad_enabled: bool = True,
    single_channel: bool = True,
    blur_kernel: Optional[List[int]] = None,
) -> torch.Tensor:
    """
    Compute saliency map

    :param pure_function: Callable function.
    :param inputs: Model inputs.
    :param targets: Ground truth labels.
    :param grad_enabled: Keep the computational graph, defaults to True.
    :param single_channel: Reduce the map down to a single channel, defaults to True.
    :param blur_map: Applying a Gaussian blur, defaults to None. Set to None for no blurring.
    :return: Computed saliency map.
    """

    # Put the model in train mode
    model.train()

    # Check the pure function
    if not check_pure_function(pure_function):
        raise ValueError(
            f"{pure_function.__name__} must have the arguments: inputs, model, targets"
        )

    # Check if blur_kernel is valid type
    if blur_kernel is not None and not isinstance(blur_kernel, List):
        raise ValueError(
            "The blur_kernel argument must be a list of 2 odd integers. \
            The given argument is not a list"
        )

    # Check if blur_kernel is valid list
    if isinstance(blur_kernel, List) and len(blur_kernel) != 2:
        raise ValueError(
            "The blur_kernel argument must be a list of 2 odd integers. \
            The given list is not of length two"
        )

    # Set up gradient operator
    compute_single_saliency = grad(
        pure_function, argnums=INPUT_IDX, has_aux=False
    )

    # Set up vmap operator for entire batch
    # All relevant arguments must be batched (see in_dims argument)
    compute_batch_saliency = vmap(
        compute_single_saliency, in_dims=(0, None, 0, None)
    )

    # Each minibatch should have the same task
    # This is a potential issue
    task = int(tasks[0])

    # Execute the transformed function
    # vmap will automatically unbatch the arguments
    per_sample_grad = compute_batch_saliency(inputs, task, targets, model)

    # Optionally reduce the channels to get single channel heatmap
    if single_channel:
        per_sample_grad = torch.mean(per_sample_grad, dim=1, keepdim=True)

    # ReLU on the heatmap to zero out negative values
    per_sample_map = F.relu(per_sample_grad)

    # Optionally blur the heatmap, tends to make it brighter
    if blur_kernel is not None and isinstance(blur_kernel, List):
        per_sample_map = gaussian_blur(per_sample_map, kernel_size=blur_kernel)

    # If backward not required, detach graph.
    # Will not allow backpropagation
    if not grad_enabled:
        per_sample_map = per_sample_map.detach()

    return per_sample_map
