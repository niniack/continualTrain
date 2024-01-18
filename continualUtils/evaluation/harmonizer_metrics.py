from typing import List

import torch
from avalanche.evaluation import GenericPluginMetric, Metric
from avalanche.evaluation.metrics.mean import Mean
from torch import Tensor

# if torch.is_tensor(harmonizer_loss):
#     harmonizer_loss = harmonizer_loss.cpu().item()

# step = strategy.clock.total_iterations
# exp_counter = strategy.clock.train_exp_counter
# mname = "harmonizer_loss/" + "exp" + str(exp_counter)
# mval = MetricValue(self, mname, harmonizer_loss, step)
# strategy.evaluator.publish_metric_value(mval)


class HarmonizerLossMetric(Metric[float]):
    """Harmonizer Loss Metric."""

    def __init__(self):
        self._mean_loss = Mean()

    @torch.no_grad()
    def update(self, harmonizer_loss: Tensor, patterns: int) -> None:
        self._mean_loss.update(torch.mean(harmonizer_loss), weight=patterns)

    def result(self) -> float:
        return self._mean_loss.result()

    def reset(self) -> None:
        self._mean_loss.reset()


class HarmonizerPluginMetric(GenericPluginMetric[float, HarmonizerLossMetric]):
    def __init__(self, reset_at, emit_at, mode):
        super().__init__(
            metric=HarmonizerLossMetric(),
            reset_at=reset_at,
            emit_at=emit_at,
            mode=mode,
        )

    def reset(self) -> None:
        self._metric.reset()

    def result(self) -> float:
        try:
            return self._metric.result()
        except Exception:
            return None

    def update(self, strategy):
        try:
            self._metric.update(
                harmonizer_loss=strategy.harmonizer_loss,  # type: ignore
                patterns=len(strategy.mb_y),
            )
        except Exception:
            pass


class MinibatchHarmonizerLoss(HarmonizerPluginMetric):
    def __init__(self):
        super().__init__(
            reset_at="iteration", emit_at="iteration", mode="train"
        )

    def __str__(self):
        return "HarmonizerLoss_Iteration"


class EpochHarmonizerLoss(HarmonizerPluginMetric):
    def __init__(self):
        super().__init__(reset_at="epoch", emit_at="epoch", mode="train")

    def __str__(self):
        return "HarmonizerLoss_Epoch"


class ExperienceHarmonizerLoss(HarmonizerPluginMetric):
    def __init__(self):
        super().__init__(
            reset_at="experience", emit_at="experience", mode="eval"
        )

    def __str__(self):
        return "HarmonizerLoss_Exp"


def harmonizer_metrics(
    *,
    minibatch=False,
    epoch=False,
    experience=False,
) -> List[HarmonizerPluginMetric]:
    """
    Helper method that can be used to obtain the desired set of
    plugin metrics.

    :param minibatch: If True, will return a metric able to log
        the minibatch harmonizer loss at training time.
    :param epoch: If True, will return a metric able to log
        the epoch harmonizer loss at training time.
    :param experience: If True, will return a metric able to log
        the accuracy on each evaluation experience.
    :return: A list of plugin metrics.
    """

    metrics: List[HarmonizerPluginMetric] = []
    if minibatch:
        metrics.append(MinibatchHarmonizerLoss())

    if epoch:
        metrics.append(EpochHarmonizerLoss())

    if experience:
        metrics.append(ExperienceHarmonizerLoss())

    return metrics
