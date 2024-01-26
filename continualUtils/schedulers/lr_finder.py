from typing import Union

from avalanche.training.plugins import LRSchedulerPlugin
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from typing_extensions import Literal


class LRFinderScheduler(_LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        epochs: int,
        min_lr: float,
        max_lr: float,
    ):
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.epochs = epochs
        self.total_iterations = None
        self.iteration = 0
        self.last_epoch = 0
        super().__init__(optimizer)

    def update_total_iterations(
        self, num_samples_experience: int, batch_size: int
    ):
        iterations_per_epoch = num_samples_experience // batch_size
        self.total_iterations = self.epochs * iterations_per_epoch

    def get_lr(self) -> float:
        return self.clr()

    def clr(self) -> float:
        """Calculate the learning rate."""
        if self.total_iterations is None or self.total_iterations == 0:
            return self.min_lr

        x = self.iteration / self.total_iterations
        return self.min_lr + (self.max_lr - self.min_lr) * x

    def step(self, epoch: Union[float, Tensor] = None):
        """Override step."""

        # Compute new learning rate
        lr = self.clr()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        self.iteration += 1

    def reset(self, *args, **kwargs):
        self.total_iterations = 0


class LRFinderSchedulerPlugin(LRSchedulerPlugin):
    def before_training_exp(self, strategy, *args, **kwargs):
        # Compute dataset length
        num_samples_experience = len(strategy.experience.dataset)
        batch_size = strategy.train_mb_size

        # Update scheduler
        self.scheduler.update_total_iterations(
            num_samples_experience, batch_size
        )

        # Call parent implementation
        super().before_training_exp(strategy, *args, **kwargs)
