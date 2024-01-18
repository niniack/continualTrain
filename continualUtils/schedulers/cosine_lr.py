import math
import warnings
from bisect import bisect_right
from collections import Counter

from torch.optim.lr_scheduler import _LRScheduler


class CosineLR(_LRScheduler):
    """
    A learning rate scheduler that implements a cosine annealing strategy.
    This scheduler adjusts the learning rate for each parameter group
    according to a cosine curve, which can be useful for fine-tuning models
    in deep learning.

    :param optimizer: Wrapped optimizer.
    :param milestones: List of epoch indices. Must be increasing.
    :param t_max: Maximum number of iterations.
    :param eta_min: Minimum learning rate, defaults to 0.
    :param last_epoch: The index of the last epoch, defaults to -1.
    :param verbose: If True, prints a message to stdout for each update, defaults to False.
    """

    def __init__(
        self,
        optimizer,
        milestones,
        t_max,
        eta_min=0,
        last_epoch=-1,
        verbose=False,
    ):
        milestones.insert(0, 0)  # Insert the initial milestone at the beginning
        self.milestones = Counter(
            milestones
        )  # Count occurrences of each milestone
        self.t_max = t_max  # Store the maximum number of iterations
        self.eta_min = eta_min  # Store the minimum learning rate
        super().__init__(
            optimizer, last_epoch, verbose
        )  # Initialize the superclass

    def get_lr(self):
        """
        Compute and return the current learning rate. If the current epoch is not
        a milestone, the method returns the last computed learning rate. Otherwise,
        it computes a new learning rate based on the cosine schedule.
        """
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )

        # If it's the first epoch or a non-milestone epoch, return the last learning rate
        if self.last_epoch == 0 or self.last_epoch not in self.milestones:
            return [group["lr"] for group in self.optimizer.param_groups]

        # Compute and return the new learning rates based on the cosine schedule
        return [
            self.eta_min
            + (base_lr - self.eta_min)
            * (1 + math.cos((self.last_epoch) * math.pi / self.t_max))
            / 2
            for base_lr, group in zip(
                self.base_lrs, self.optimizer.param_groups
            )
        ]

    def _get_closed_form_lr(self):
        """
        Compute the learning rate in a closed form. This method is used for
        internal calculations and may not be typically called by the end user.
        """
        # Sort milestones and find the current T value
        milestones = sorted(self.milestones.elements())
        t_curr = milestones[bisect_right(milestones, self.last_epoch) - 1]

        # Compute and return the learning rates in closed form
        return [
            self.eta_min
            + (base_lr - self.eta_min)
            * (1 + math.cos(t_curr * math.pi / self.T_max))
            / 2
            for base_lr, group in zip(
                self.base_lrs, self.optimizer.param_groups
            )
        ]
