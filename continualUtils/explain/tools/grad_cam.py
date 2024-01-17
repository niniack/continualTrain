from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch.func import grad, vmap

from continualUtils.explain.tools.utils import (
    OneHotException,
    get_activation,
    get_layers_from_model,
)


def compute_grad_cam(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    tasks: torch.Tensor,
    targets: torch.Tensor,
    target_layer_name: str,
    grad_enabled: bool = True,
) -> torch.Tensor:
    """
    Compute saliency map

    :param inputs: Model inputs.
    :param tasks: Model task.
    :param targets: Ground truth labels.
    :param target_layer: Name of target layer as a string
    :param grad_enabled: Keep the computational graph, defaults to True.
    :return: Computed saliency map.
    """

    # Each minibatch should have the same task
    # This is a potential issue
    task = int(tasks[0])

    # Execute the transformed function
    # vmap will automatically unbatch the arguments
    act_container = dict()
    delta_intermediates = torch.zeros(
        (inputs.shape[0], 2048, 7, 7), requires_grad=True
    )

    layer = get_layers_from_model(model, target_layer_name)
    hook = layer.register_forward_hook(
        get_activation(act_container, target_layer_name)
    )

    # Set up gradient operator
    compute_single_saliency = grad(compute_score, argnums=4, has_aux=False)

    # Set up vmap operator for entire batch
    # All relevant arguments must be batched (see in_dims argument)
    compute_batch_saliency = vmap(
        compute_single_saliency, in_dims=(0, None, 0, None, 0, None)
    )

    per_sample_grad = compute_batch_saliency(
        inputs, task, targets, model, delta_intermediates, target_layer_name
    )

    print(per_sample_grad)

    # activations = act_container[target_layer_name].squeeze(1)

    # # If backward not required, detach graph.
    # # Will not allow backpropagation
    # if not grad_enabled:
    #     per_sample_grad = per_sample_grad.detach()

    # # Find weights for activations
    # weights = torch.mean(per_sample_grad, dim=(2, 3), keepdim=True)

    # print(weights)

    # # Combine to get weighted activations
    # weighted_activation = weights * activations

    # # Sum over all channels
    # per_sample_cam = torch.sum(weighted_activation, dim=1, keepdim=True)

    # # ReLU on the heatmap to zero out negative values
    # per_sample_cam = F.relu(per_sample_cam)

    # # Resize to input
    # per_sample_cam = F.interpolate(
    #     per_sample_cam,
    #     size=(inputs.shape[2], inputs.shape[3]),
    #     mode="bilinear",
    #     align_corners=False,
    # )

    hook.remove()
    return per_sample_grad


def compute_score(
    x: torch.Tensor,
    task: int,
    y: torch.Tensor,
    model: torch.nn.Module,
    delta_intermediate: Optional[torch.Tensor] = None,
    layer_name: Optional[str] = None,
    act_container: Optional[dict] = None,
) -> torch.Tensor:
    """
    Since vmap will unbatch and vectorize the computation, we
    assume that all the inputs do not have a batch dimension.
    """

    # Batch
    x = x.unsqueeze(0)
    y = y.unsqueeze(0)

    if (
        delta_intermediate is not None
        and layer_name is not None
        and act_container is not None
    ):
        # Forward, grab hook output
        output = model(x, task)
        activation = act_container[layer_name].squeeze(0) + delta_intermediate
    else:
        # Simple forward
        output = model(x, task)

    if output.shape != y.shape:
        raise OneHotException(
            "The model outputs must be the shape as the target."
        )
    score = torch.sum(output * y)
    return score
