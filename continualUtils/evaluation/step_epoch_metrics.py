from typing import List

from avalanche.evaluation.metric_definitions import GenericPluginMetric
from avalanche.training.templates import SupervisedTemplate


class StepEpochMetric:
    """Step epoch metric class"""

    def __init__(self):
        self.epoch: float = 0

    def update(self, epoch):
        self.epoch = epoch

    def reset(self):
        self.epoch = 0

    def result(self):
        return self.epoch


class StepEpochPluginMetric(GenericPluginMetric[float, StepEpochMetric]):
    """Plugin wrapper for step epoch"""

    def __init__(self, reset_at, emit_at, mode):
        super().__init__(
            metric=StepEpochMetric(),
            reset_at=reset_at,
            emit_at=emit_at,
            mode=mode,
        )

    def reset(self) -> None:
        self._metric.reset()

    def result(self) -> float:
        return self._metric.result()

    # Called after training iteration
    def update(self, strategy: "SupervisedTemplate"):
        self._metric.update(strategy.clock.train_exp_epochs)


class StepEpoch(StepEpochPluginMetric):
    """Wrapper for step epoch plugin"""

    def __init__(self):
        super().__init__(reset_at="experience", emit_at="epoch", mode="train")

    def __str__(self):
        return "Epoch"


def step_epoch_metrics() -> List[StepEpochPluginMetric]:
    """
    Helper method that can be used to obtain the desired set of
    plugin metrics.
    """
    return [StepEpoch()]
