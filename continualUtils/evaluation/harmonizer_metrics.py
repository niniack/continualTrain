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
    """Harmonizer Loss Metric.

    Instances of this metric keep the running average harmonizer loss
    over multiple <prediction, target> pairs of Tensors,
    provided incrementally.

    Each time `result` is called, this metric emits the average harmonizer
    loss across all predictions made since the last `reset`.

    The reset method will bring the metric to its initial state. By default
    this metric in its initial state will return a loss value of 0.
    """

    def __init__(self):
        """
        Creates an instance of the harmonizer loss metric.

        By default this metric in its initial state will return a loss
        value of 0. The metric can be updated by using the `update` method
        while the running loss can be retrieved using the `result` method.
        """
        self._mean_loss = Mean()
        """
        The mean utility that will be used to store the running accuracy
        for each task label.
        """

    @torch.no_grad()
    def update(self, harmonizer_loss: Tensor, patterns: int) -> None:
        """Update the running loss.

        :param loss: The harmonizer Tensor. Different reduction types
            don't affect the result.
        :param patterns: The number of patterns in the minibatch.
        :return: None.
        """
        self._mean_loss.update(torch.mean(harmonizer_loss), weight=patterns)

    def result(self) -> float:
        """Retuns the running average loss per pattern.

        Calling this method will not change the internal state of the metric.

        :return: The running loss, as a float.
        """
        return self._mean_loss.result()

    def reset(self) -> None:
        """Resets the metric.

        :return: None.
        """
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
    """
    The minibatch plugin harmonizer loss metric.
    This metric only works at training time.

    This metric computes the average harmonizer loss over patterns
    from a single minibatch.
    It reports the result after each iteration.
    """

    def __init__(self):
        """
        Creates an instance of the MinibatchAccuracy metric.
        """
        super().__init__(
            reset_at="iteration", emit_at="iteration", mode="train"
        )

    def __str__(self):
        return "HarmonizerLoss_Iteration"


class EpochHarmonizerLoss(HarmonizerPluginMetric):
    """
    The average harmonizer loss over a single training epoch.
    This plugin metric only works at training time.

    The harmonizer loss will be logged after each training epoch by computing
    the number of correctly predicted patterns during the epoch divided by
    the overall number of patterns encountered in that epoch.
    """

    def __init__(self):
        """
        Creates an instance of the EpochAccuracy metric.
        """

        super().__init__(reset_at="epoch", emit_at="epoch", mode="train")

    def __str__(self):
        return "HarmonizerLoss_Epoch"


class ExperienceHarmonizerLoss(HarmonizerPluginMetric):
    """
    At the end of each experience, this plugin metric reports
    the average harmonizer loss over all patterns seen in that experience.
    This metric only works at eval time.
    """

    def __init__(self):
        """
        Creates an instance of ExperienceHarmonizerLoss metric
        """
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
