class TotalVariationLoss:
    """
    Computes total variation loss. Adapted from:
    https://kornia.readthedocs.io/en/latest/_modules/kornia/losses/total_variation.html


    The loss is calculated as the sum of absolute differences between neighboring
    pixel values in both horizontal and vertical directions.

    The class can be used as a callable with an image tensor and an optional
    reduction method ('mean' or 'sum').
    """

    def __call__(self, img, reduction="mean"):
        # Calculate the difference between adjacent pixels along the height (axis -2)
        pixel_dif1 = img[..., 1:, :] - img[..., :-1, :]
        # Calculate the difference between adjacent pixels along the width (axis -1)
        pixel_dif2 = img[..., :, 1:] - img[..., :, :-1]

        # Compute absolute differences
        res1 = pixel_dif1.abs()
        res2 = pixel_dif2.abs()

        # Define axes for reduction
        reduce_axes = (-2, -1)

        # Apply mean reduction
        if reduction == "mean":
            # Convert to the same type as input image and compute mean if floating point
            if img.is_floating_point():
                res1 = res1.to(img).mean(dim=reduce_axes)
                res2 = res2.to(img).mean(dim=reduce_axes)
            # Convert to float and compute mean if not floating point
            else:
                res1 = res1.float().mean(dim=reduce_axes)
                res2 = res2.float().mean(dim=reduce_axes)

        # Apply sum reduction
        elif reduction == "sum":
            res1 = res1.sum(dim=reduce_axes)
            res2 = res2.sum(dim=reduce_axes)

        # Raise error if invalid reduction option is provided
        else:
            raise NotImplementedError("Invalid reduction option.")

        # Return the total variation loss
        return res1 + res2
