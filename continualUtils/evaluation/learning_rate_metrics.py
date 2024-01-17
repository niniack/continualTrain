from typing import List

from avalanche.evaluation.metric_definitions import GenericPluginMetric
from avalanche.training.templates import SupervisedTemplate


class LearningRateMetric:
    def __init__(self):
        self.learning_rate: float = 0

    def update(self, learning_rate):
        self.learning_rate = learning_rate

    def reset(self):
        self.learning_rate = -1

    def result(self):
        return self.learning_rate


class LearningRatePluginMetric(GenericPluginMetric[float, LearningRateMetric]):
    def __init__(self, reset_at, emit_at, mode):
        """
        Initialize the metric
        """
        super().__init__(
            metric=LearningRateMetric(),
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
        self._metric.update(strategy.optimizer.param_groups[0]["lr"])


class MiniBatchLearningRate(LearningRatePluginMetric):
    def __init__(self):
        super().__init__(reset_at="never", emit_at="iteration", mode="train")

    def __str__(self):
        return "LearningRate_Iteration"


class EpochLearningRate(LearningRatePluginMetric):
    def __init__(self):
        super().__init__(reset_at="never", emit_at="epoch", mode="train")

    def __str__(self):
        return "LearningRate_Epoch"


def learning_rate_metrics(
    minibatch=False, epoch=False
) -> List[LearningRatePluginMetric]:
    plugins: List[LearningRatePluginMetric] = []

    if minibatch:
        plugins.append(MiniBatchLearningRate())
    if epoch:
        plugins.append(EpochLearningRate())
    return plugins
