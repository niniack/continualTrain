import torch
import torch.nn.functional as F
from avalanche.training.regularization import RegularizationMethod

from continualUtils.explain.tools import (
    compute_saliency_map,
    compute_score,
    standardize_cut,
)


class NeuralHarmonizerLoss(RegularizationMethod):
    """Neural Harmonizer Loss

    Built from https://serre-lab.github.io/Harmonization/
    """

    def __init__(self, weight: float, epsilon: float = 1e-8):
        self.weight = weight
        self.epsilon = epsilon

    def __call__(self, mb_x, mb_y, mb_heatmap, model, mb_tokens, mb_tasks):
        # The input must have gradients turned on
        if not mb_x.requires_grad:
            mb_x.requires_grad_(True)

        # Generate a saliency map
        # Make targets one hot for our pure function
        if mb_y.dim() == 1:
            mb_y = F.one_hot(mb_y, model.num_classes_per_head)
        output_maps = compute_saliency_map(
            pure_function=compute_score,
            model=model,
            inputs=mb_x,
            tasks=mb_tasks,
            targets=mb_y,
        )

        # Standardize cut procedure
        output_maps_standardized = standardize_cut(output_maps)
        ground_maps_standardized = standardize_cut(mb_heatmap)

        # Normalize the true heatmaps according to the saliency maps
        # No gradients needed, we are just updating the ground truth maps
        with torch.no_grad():
            _om_max = (
                torch.amax(output_maps_standardized, dim=(2, 3), keepdim=True)
                + self.epsilon
            )
            _gm_max = (
                torch.amax(ground_maps_standardized, dim=(2, 3), keepdim=True)
                + self.epsilon
            )

            ground_maps_standardized = (
                ground_maps_standardized / _gm_max * _om_max
            )

        # Pyramidal loss
        pyramidal_loss = compute_pyramidal_mse(
            output_maps_standardized, ground_maps_standardized, mb_tokens
        )

        return pyramidal_loss * self.weight

    def update(self, *args, **kwargs):
        pass


def compute_pyramidal_mse(predicted_maps, true_maps, mb_tokens, num_levels=5):
    """Compute the pyramidal versin of the mean squared error. Converts
    maps to pyramidal representation and then computes the mse

    :param predicted_maps: Output heatmaps from the model
    :param true_maps: Ground truth maps
    :param mb_tokens: Tokens from the dataset
    :param num_levels: The number of downsampled pyramidal representations, defaults to 5
    :return: Mean loss of all the representations
    """
    pyramid_y = _pyramidal_representation(true_maps, num_levels)
    pyramid_y_pred = _pyramidal_representation(predicted_maps, num_levels)

    pyramid_loss = [
        _mse(pyramid_y[i], pyramid_y_pred[i], mb_tokens)
        for i in range(num_levels + 1)
    ]

    return torch.mean(torch.stack(pyramid_loss), dim=0)


def _mse(heatmaps_a, heatmaps_b, tokens):
    """Computes the mean squared error between two heatmaps,
    if the token is set to 1, ignored if token is 0

    :param heatmaps_a: heatmap NCHW
    :param heatmaps_b: heatmap NCHW
    :param tokens: Token from ClickMe
    :return: Mean squared error
    """

    # First compute error without reduction, to ensure tokens are accounted for
    return torch.mean(
        F.mse_loss(heatmaps_a, heatmaps_b, reduction="none")
        * tokens[:, None, None, None]
    )


def _pyramidal_representation(maps, num_levels):
    """Returns a list with num_levels downsampled heatmaps

    :param maps: Heatmap NCHW
    :param num_levels: Number of levels to downsample
    :return: List of heatmaps
    """
    levels = [maps]
    for _ in range(num_levels):
        new_size = maps.shape[-1] // 2
        maps = F.interpolate(maps, size=new_size, mode="bilinear")
        levels.append(maps)
    return levels
