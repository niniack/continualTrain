import numpy as np
import torch
import torch.nn.functional as F
from avalanche.training.regularization import RegularizationMethod
from captum.attr import Saliency
from captum.attr import visualization as viz


class SaliencyGuidedLoss(RegularizationMethod):
    """
    Computes saliency guided loss. Adapted from:
    https://github.com/ayaabdelsalam91/saliency_guided_training


    """

    def __init__(
        self,
        random_masking: bool = True,
        features_dropped: float = 0.5,
        add_noise: bool = False,
        noise_mag: float = 1e-1,
        noisy_features: float = 0.1,
        abs_grads: bool = False,
        mean_saliency: bool = True,
    ):
        self.abs_grads = abs_grads
        self.random_masking = random_masking
        self.noisy_features = noisy_features
        self.features_dropped = features_dropped
        self.add_noise = add_noise
        self.noise_mag = noise_mag
        self.mean_saliency = mean_saliency

    def update(self, *args, **kwargs):
        pass

    def get_masks(self, num_masked_features, grads, noise=False):
        # Get batch size and number of channels
        batch, channels, *_ = grads.shape

        # Reshape grads for multi-channel processing
        grads_reshaped = grads.view(batch, channels, -1)

        # Get topk indices for each channel
        _, top_indices = torch.topk(
            grads_reshaped, num_masked_features, dim=2, largest=noise
        )

        # Initialize flat mask for each channel
        mask = torch.zeros_like(grads_reshaped, dtype=torch.bool)

        # Fill mask for each channel
        image_idx = torch.arange(batch).unsqueeze(1).unsqueeze(2)
        channel_idx = torch.arange(channels).unsqueeze(0).unsqueeze(2)
        mask[image_idx, channel_idx, top_indices] = True

        # Reshape mask to original grad shape
        mask = mask.reshape(grads.shape)

        return mask

    def fill_masks(self, mb_x, mb_masks, noise=False):
        # Get input shape
        batch, channels, height, width = mb_x.shape

        # Expand mask in channel dim, if needed
        if mb_masks.shape[1] != channels:
            mb_masks = mb_masks.expand(batch, channels, height, width)

        if noise:
            noise_values = torch.rand_like(mb_x) * (2e-1) - 1e-1
            mb_x = torch.where(mb_masks, mb_x + noise_values, mb_x)
        else:
            # Get min and max for each channel, for each image
            min_vals = (
                mb_x.view(batch, channels, -1)
                .min(dim=2)[0]
                .unsqueeze(-1)
                .unsqueeze(-1)
            )
            max_vals = (
                mb_x.view(batch, channels, -1)
                .max(dim=2)[0]
                .unsqueeze(-1)
                .unsqueeze(-1)
            )

            # Build random values within range
            random_values = (
                torch.rand_like(mb_x) * (max_vals - min_vals)
            ) + min_vals

            # Replace with random values where indices are True
            mb_x = torch.where(
                condition=mb_masks, input=random_values, other=mb_x
            )

        return mb_x

    def __call__(self, mb_x, mb_y, model, mb_output=None, *args, **kwargs):
        batch, channels, height, width = mb_x.shape
        num_masked_features = int(self.features_dropped * height * width)

        # Build saliency map
        # Take mean
        saliency_engine = Saliency(model)
        grads = (
            saliency_engine.attribute(mb_x, mb_y, abs=False)
            .detach()
            .to(dtype=torch.float)
        )

        if self.mean_saliency:
            grads = grads.mean(dim=1, keepdim=True)

        # if self.abs_grads:
        #     grads = grads.abs()

        # Clone images
        temp_mb_x = mb_x.detach().clone()

        # Build boolean mask with top k salient features
        single_masks = self.get_masks(num_masked_features, grads)

        # Fill top k indices with random values
        temp_mb_x = self.fill_masks(mb_x, single_masks)

        ################# AMIRA #################
        # indices = indices.view(batch, channels, num_masked_features)
        # for idx in range(batch):
        #     if self.random_masking:
        #         # Iterate over each channel
        #         for channel in range(channels):
        #             # Get the minimum and maximum values in the current channel
        #             min_ = torch.min(temp_mb_x[idx, channel, :]).item()
        #             max_ = torch.max(temp_mb_x[idx, channel, :]).item()

        #             randomMask = np.random.uniform(
        #                 low=min_,
        #                 high=max_,
        #                 size=(len(indices[idx][channel]),),
        #             )

        #             temp_mb_x[idx][channel][
        #                 indices[idx][channel]
        #             ] = torch.Tensor(randomMask).to(temp_mb_x.device)

        #     else:
        #         for channel in range(temp_mb_x.shape[1]):
        #             temp_mb_x[idx][channel][indices[idx][channel]] = mb_x[
        #                 0, channel, 0, 0
        #             ]
        ################# AMIRA #################

        # Add noise
        # TODO: This is broken
        if self.add_noise:
            num_noisy_features = int(self.noisy_features * height * width)
            # Build and fill random masks
            single_masks = self.get_masks(
                num_noisy_features, grads.unsqueeze(1), noise=True
            )
            temp_mb_x = self.fill_masks(mb_x, single_masks, noise=True)

        # Reshape to original tensor
        masked_input = temp_mb_x.view(mb_x.shape).detach()

        # Feed into model
        masked_output = F.log_softmax(model(masked_input), dim=1)
        standard_output = mb_output if mb_output else model(mb_x)
        standard_output = F.log_softmax(standard_output, dim=1)

        # KL Loss will be added to main loss
        kl_loss = F.kl_div(
            masked_output,
            standard_output,
            reduction="batchmean",
            log_target=True,
        )

        return kl_loss
