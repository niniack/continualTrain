import torch
import torch.nn.functional as F


def standardize_cut(heatmaps, axes=(2, 3), epsilon=1e-12):
    """Z-scores a NCHW tensor and returns the positive values

    :param heatmaps: NCHW tensor
    :param axes: Axes to z-score, defaults to (2, 3)
    :param epsilon: Small value to avoid divide by zero, defaults to 1e-5
    :return: Z-scored tensor
    """
    if heatmaps.dim() != 4:
        raise ValueError(
            f"""Ensure that the provided tensor is in NCHW format, 
            there are currently {heatmaps.dim()} dims"""
        )

    means = torch.mean(heatmaps, dim=axes, keepdim=True)
    stds = torch.std(heatmaps, dim=axes, keepdim=True)

    heatmaps = heatmaps - means
    heatmaps = heatmaps / (stds + epsilon)

    # Grab the positive parts of the explanation
    heatmaps = F.relu(heatmaps)

    return heatmaps
