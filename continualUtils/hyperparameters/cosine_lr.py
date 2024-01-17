import math
import warnings
from bisect import bisect_right
from collections import Counter

from torch.optim.lr_scheduler import _LRScheduler


class CosineLR(_LRScheduler):
    def __init__(
        self,
        optimizer,
        milestones,
        T_max,
        eta_min=0,
        last_epoch=-1,
        verbose=False,
    ):
        milestones.insert(0, 0)
        self.milestones = Counter(milestones)
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:  # type: ignore
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )

        # No changes to LR, return last one
        if self.last_epoch == 0:
            return [group["lr"] for group in self.optimizer.param_groups]

        # No changes to LR, return last one
        elif self.last_epoch not in self.milestones:
            return [group["lr"] for group in self.optimizer.param_groups]

        # Otherwise grab from cosine schedule
        return [
            self.eta_min
            + (base_lr - self.eta_min)
            * (1 + math.cos((self.last_epoch) * math.pi / self.T_max))
            / 2
            for base_lr, group in zip(
                self.base_lrs, self.optimizer.param_groups
            )
        ]

    def _get_closed_form_lr(self):
        milestones = sorted(self.milestones.elements())

        T_curr = milestones[bisect_right(milestones, self.last_epoch) - 1]

        return [
            self.eta_min
            + (base_lr - self.eta_min)
            * (1 + math.cos(T_curr * math.pi / self.T_max))
            / 2
            for base_lr, group in zip(
                self.base_lrs, self.optimizer.param_groups
            )
        ]
