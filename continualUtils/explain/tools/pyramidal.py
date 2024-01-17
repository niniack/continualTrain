import torch
import torch.nn.functional as F


def standardize_cut(heatmaps, axes=(2, 3), epsilon=1e-12):
    """_summary_

    :param heatmaps: _description_
    :param axes: _description_, defaults to (2, 3)
    :param epsilon: _description_, defaults to 1e-5
    :raises ValueError: _description_
    :return: _description_
    """
    if heatmaps.dim() != 4:
        raise ValueError(
            f"""Ensure that heatmaps are in NCHW cuDNN format, 
            there are currently {heatmaps.dim()} dims"""
        )

    means = torch.mean(heatmaps, dim=axes, keepdim=True)
    stds = torch.std(heatmaps, dim=axes, keepdim=True)

    heatmaps = heatmaps - means
    heatmaps = heatmaps / (stds + epsilon)

    # Grab the positive parts of the explanation
    heatmaps = F.relu(heatmaps)

    return heatmaps


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
    :return: mean squared error
    """

    # First compute error without reduction, to ensure tokens are accounted for
    return torch.mean(
        F.mse_loss(heatmaps_a, heatmaps_b, reduction="none")
        * tokens[:, None, None, None]
    )


def _pyramidal_representation(maps, num_levels):
    levels = [maps]
    for _ in range(num_levels):
        # # DEBUG changing downsampling
        # maps = _downsample(maps, kernel)
        new_size = maps.shape[-1] // 2
        maps = F.interpolate(maps, size=new_size, mode="bilinear")
        levels.append(maps)
    return levels


# # DEBUG changing downsampling
# def _binomial_kernel(num_channels):
#     kernel = torch.FloatTensor([1., 4., 6., 4., 1.])
#     kernel = torch.outer(kernel, kernel)
#     kernel /= torch.sum(kernel)
#     return kernel.repeat((num_channels, num_channels, 1, 1))

# # DEBUG changing downsampling
# def _downsample(maps, kernel):
#     return F.conv2d(input=maps, weight=kernel, stride=2, padding='valid')

# How to use
# Returns a list
# sa_maps = _alt_saliency(mb_x, mb_y, mb_pred)[0]

# def _alt_saliency_map(mb_x, mb_y, mb_pred, cam):
#     '''
#     Returns a list
#     '''
#     mb_pred_argmax = mb_pred.argmax(dim=1).tolist()
#     grads = cam(mb_pred_argmax, mb_pred, retain_graph=True)
#     return grads
