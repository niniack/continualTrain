from typing import Any

import torch
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin

from continualUtils.benchmarks.datasets.clickme import (
    HEATMAP_INDEX,
    TOKEN_INDEX,
)
from continualUtils.explain.losses.harmonizer_loss import NeuralHarmonizerLoss


class NeuralHarmonizerPlugin(SupervisedPlugin):
    """Neural Harmonizer plugin"""

    def __init__(self, weight=1):
        super().__init__()
        self.weight = weight
        self.harmonizer = NeuralHarmonizerLoss(weight)

    def before_forward(self, strategy: Any, *args, **kwargs):
        # Clean up harmonizer loss
        strategy.harmonizer_loss = torch.zeros(1, device=strategy.device)

    def before_backward(self, strategy: Any, *args, **kwargs):
        # Clone the input
        cloned_mb_x = strategy.mb_x.detach()

        # Turn on gradients for the cloned input
        if cloned_mb_x.requires_grad is False:
            cloned_mb_x.requires_grad_(True)

        # Get the heatmaps and tokens
        strategy.mb_heatmaps = strategy.mbatch[HEATMAP_INDEX]
        strategy.mb_tokens = strategy.mbatch[TOKEN_INDEX]
        strategy.mb_tasks = strategy.mbatch[4]

        # NH is a heavy operation, should only be done for samples with heatmaps
        # Compute all indices with token set to 1
        indices = torch.nonzero(strategy.mb_tokens, as_tuple=True)[0]

        # Uses the `__call__` method
        # Send in the cloned input with gradients turned on
        # Select samples with heatmap
        harmonizer_loss = self.harmonizer(
            cloned_mb_x[indices],
            strategy.mb_y[indices],
            strategy.mb_heatmaps[indices],
            strategy.model,
            strategy.mb_tokens[indices],
            strategy.mb_tasks[indices],
        )

        # Add the harmonizer loss
        strategy.loss += harmonizer_loss
        strategy.harmonizer_loss = harmonizer_loss

        # Turn off gradients for the cloned input
        cloned_mb_x.requires_grad_(False)
        cloned_mb_x.detach()
