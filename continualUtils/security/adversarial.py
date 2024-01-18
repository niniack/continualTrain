from typing import TYPE_CHECKING, Callable, Optional

import torch
from avalanche.benchmarks.utils import _make_taskaware_classification_dataset
from avalanche.benchmarks.utils.data_loader import ReplayDataLoader
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin
from avalanche.training.storage_policy import (
    ExemplarsBuffer,
    ExperienceBalancedBuffer,
)
from packaging.version import parse

from continualUtils.security.tools.universal_adversarial_perturbation import (
    UniversalAdversarialPerturbation,
)

if TYPE_CHECKING:
    from avalanche.training.templates import SupervisedTemplate


class AdversarialReplayPlugin(SupervisedPlugin, supports_distributed=True):
    """
    Adversarial experience replay plugin.

    Handles an external memory filled with randomly selected
    patterns and implementing `before_training_exp` and `after_training_exp`
    callbacks. The external memory will contain adversarial examples.

    The `before_training_exp` callback is implemented in order to use the
    dataloader that creates mini-batches with examples from both training
    data and external memory. The examples in the mini-batch is balanced
    such that there are the same number of examples for each experience.

    The `after_training_exp` callback is implemented in order to add new
    patterns to the external memory.

    The :mem_size: attribute controls the total number of patterns to be stored
    in the external memory.

    :param batch_size: the size of the data batch. If set to `None`, it
        will be set equal to the strategy's batch size.
    :param batch_size_mem: the size of the memory batch. If
        `task_balanced_dataloader` is set to True, it must be greater than or
        equal to the number of tasks. If its value is set to `None`
        (the default value), it will be automatically set equal to the
        data batch size.
    :param task_balanced_dataloader: if True, buffer data loaders will be
            task-balanced, otherwise it will create a single dataloader for the
            buffer samples.
    :param storage_policy: The policy that controls how to add new exemplars
                           in memory
    :param attack_method: The attack function to generate adversarial noise, defaults to None
    """

    def __init__(
        self,
        mem_size: int = 200,
        batch_size: Optional[int] = None,
        batch_size_mem: Optional[int] = None,
        task_balanced_dataloader: bool = False,
        storage_policy: Optional["ExemplarsBuffer"] = None,
        attack_method: Optional[Callable] = None,
        learning_rate: float = 0.01,
        iterations: int = 2000,
        epsilon: float = 0.5,
    ):
        super().__init__()
        self.mem_size = mem_size
        self.batch_size = batch_size
        self.batch_size_mem = batch_size_mem
        self.task_balanced_dataloader = task_balanced_dataloader

        if storage_policy is not None:  # Use other storage policy
            self.storage_policy = storage_policy
            assert storage_policy.max_size == self.mem_size
        else:  # Default
            self.storage_policy = ExperienceBalancedBuffer(
                max_size=self.mem_size, adaptive_size=True
            )

        if attack_method is not None:  # Use other attack method
            self.attack_method = attack_method
        else:  # Default
            self.attack_method = UniversalAdversarialPerturbation()

        self.learning_rate = learning_rate
        self.iterations = iterations
        self.epsilon = epsilon

    def before_training_exp(
        self,
        strategy: "SupervisedTemplate",
        num_workers: int = 0,
        shuffle: bool = True,
        drop_last: bool = False,
        **kwargs
    ):
        """
        Dataloader to build batches containing examples from both memories and
        the training dataset
        """
        if len(self.storage_policy.buffer) == 0:
            # first experience. We don't use the buffer, no need to change
            # the dataloader.
            return

        batch_size = self.batch_size
        if batch_size is None:
            batch_size = strategy.train_mb_size

        batch_size_mem = self.batch_size_mem
        if batch_size_mem is None:
            batch_size_mem = strategy.train_mb_size

        assert strategy.adapted_dataset is not None

        other_dataloader_args = dict()

        if "ffcv_args" in kwargs:
            other_dataloader_args["ffcv_args"] = kwargs["ffcv_args"]

        if "persistent_workers" in kwargs:
            if parse(torch.__version__) >= parse("1.7.0"):
                other_dataloader_args["persistent_workers"] = kwargs[
                    "persistent_workers"
                ]

        # Attack replay buffer
        adversarial_transform = self.attack_method(
            dataset=self.storage_policy.buffer,
            model=strategy.model,
            learning_rate=self.learning_rate,
            iterations=self.iterations,
            eps=self.epsilon,
            device=strategy.device,
        )
        self.storage_policy.buffer = _make_taskaware_classification_dataset(
            self.storage_policy.buffer, transform=adversarial_transform
        )

        strategy.dataloader = ReplayDataLoader(
            strategy.adapted_dataset,
            self.storage_policy.buffer,
            oversample_small_tasks=True,
            batch_size=batch_size,
            batch_size_mem=batch_size_mem,
            task_balanced_dataloader=self.task_balanced_dataloader,
            num_workers=num_workers,
            shuffle=shuffle,
            drop_last=drop_last,
            **other_dataloader_args
        )

    def after_training_exp(self, strategy: "SupervisedTemplate", **kwargs):
        self.storage_policy.update(strategy, **kwargs)