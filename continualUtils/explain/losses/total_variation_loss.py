import torch


class TotalVariationLoss:
    """Computes total variation loss. Adapted from:
    https://kornia.readthedocs.io/en/latest/_modules/kornia/losses/total_variation.html
    """

    def __call__(self, img, reduction="mean"):
        pixel_dif1 = img[..., 1:, :] - img[..., :-1, :]
        pixel_dif2 = img[..., :, 1:] - img[..., :, :-1]

        res1 = pixel_dif1.abs()
        res2 = pixel_dif2.abs()

        reduce_axes = (-2, -1)
        if reduction == "mean":
            if img.is_floating_point():
                res1 = res1.to(img).mean(dim=reduce_axes)
                res2 = res2.to(img).mean(dim=reduce_axes)
            else:
                res1 = res1.float().mean(dim=reduce_axes)
                res2 = res2.float().mean(dim=reduce_axes)
        elif reduction == "sum":
            res1 = res1.sum(dim=reduce_axes)
            res2 = res2.sum(dim=reduce_axes)
        else:
            raise NotImplementedError("Invalid reduction option.")

        return res1 + res2
