from typing import List, Union

import torch
from torch import Tensor


def rescale_batch(inputs: Union[List[Tensor], Tensor]) -> Tensor:
    """Rescale a NCHW batch from 0 to 1

    :param inputs: A list of 3D CHW tensors or a 4D tensor in NCHW
    :return: NCHW tensor scaled from 0 to 1
    """
    # Handle list of tensors
    if isinstance(inputs, list):
        # Stack the list of tensors to form a 4D tensor
        inputs = torch.stack(inputs, dim=0)

    # Get dims
    n, c, h, w = inputs.shape

    # Flatten CHW dims in view
    inputs_flattened = inputs.view(inputs.size(0), -1)

    min_vals = inputs_flattened.min(dim=1, keepdim=True).values
    max_vals = inputs_flattened.max(dim=1, keepdim=True).values

    # Reshape the min and max values to BCHW
    min_vals = min_vals.view(n, 1, 1, 1).expand(n, c, h, w)
    max_vals = max_vals.view(n, 1, 1, 1).expand(n, c, h, w)

    # Rescale between 0 and 1
    # Added epsilon to avoid division by zero
    inputs = (inputs - min_vals) / (max_vals - min_vals + 1e-15)

    inputs = inputs.to(torch.float32)  # type: ignore

    return inputs
