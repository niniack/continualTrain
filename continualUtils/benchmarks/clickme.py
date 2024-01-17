from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, Union

from avalanche.benchmarks import (
    NCExperience,
    NCScenario,
    NCStream,
    nc_benchmark,
)
from avalanche.benchmarks.scenarios import (
    ClassificationScenario,
    DatasetScenario,
    StreamDef,
    StreamUserDef,
)
from avalanche.benchmarks.utils import AvalancheDataset

from continualUtils.benchmarks.datasets.clickme import (
    make_combined_clickme_dataset,
)


def SplitClickMe(  # pylint: disable=C0103
    n_experiences: int,
    root: str,
    return_task_id=False,
    seed: Optional[int] = None,
    fixed_class_order: Optional[Sequence[int]] = None,
    shuffle: bool = False,
    class_ids_from_zero_in_each_exp: bool = False,
    train_transform: Optional[Any] = None,
    eval_transform: Optional[Any] = None,
    dummy: bool = False,
    include_imagenet: bool = True,
) -> Union[NCScenario, DatasetScenario]:
    """Returns a split version of the ClickMe dataset

    :param n_experiences: The number of incremental experience. This is not used
        when using multiple train/test datasets with the ``one_dataset_per_exp``
        parameter set to True.
    :param root: Root of the dataset, where train and test are accessible
    :param return_task_id: If True, each experience will have an ascending task
            label. If False, the task label will be 0 for all the experiences., defaults to False
    :param seed: If ``shuffle`` is True and seed is not None, the class (or
        experience) order will be shuffled according to the seed. When None, the
        current PyTorch random number generator state will be used. Defaults to
        None.
    :param fixed_class_order: If not None, the class order to use (overrides
        the shuffle argument). Very useful for enhancing reproducibility.
        Defaults to None.
    :param shuffle: If True, the class (or experience) order will be shuffled.
        Defaults to True.
    :param class_ids_from_zero_in_each_exp:If True, original class IDs
        will be remapped so that they will appear as having an ascending
        order. For instance, if the resulting class order after shuffling
        (or defined by fixed_class_order) is [23, 34, 11, 7, 6, ...] and
        class_ids_from_zero_from_first_exp is True, then all the patterns
        belonging to class 23 will appear as belonging to class "0",
        class "34" will be mapped to "1", class "11" to "2" and so on.
        This is very useful when drawing confusion matrices and when dealing
        with algorithms with dynamic head expansion. Defaults to False.
        Mutually exclusive with the ``class_ids_from_zero_in_each_exp``
        parameter., defaults to False
    :param train_transform: The transformation to apply to the training data,
        e.g. a random crop, a normalization or a concatenation of different
        transformations (see torchvision.transform documentation for a
        comprehensive list of possible transformations). Defaults to None.
    :param eval_transform: The transformation to apply to the test data,
        e.g. a random crop, a normalization or a concatenation of different
        transformations (see torchvision.transform documentation for a
        comprehensive list of possible transformations). Defaults to None.
    :param dummy: If True, the scenario will be a reduced version for testing
        and debugging purposes, defaults to False
    :param include_imagenet: If True, the scenario will include the Imagenet
        directory
    :return: A properly initialized :class:`NCScenario` instance.
    """

    # DEBUG for faster loading of the dataset
    if dummy:
        clickme_train = make_combined_clickme_dataset(
            imagenet_root="/mnt/datasets/fake_imagenet"
            if include_imagenet
            else None,
            imagenet_split="train" if include_imagenet else None,
            clickme_root=root,
            clickme_split="dtrain",
        )
        # Imagenet split is val because we don't have the test split
        clickme_test = make_combined_clickme_dataset(
            imagenet_root="/mnt/datasets/fake_imagenet"
            if include_imagenet
            else None,
            imagenet_split="val" if include_imagenet else None,
            clickme_root=root,
            clickme_split="dtest",
        )

        return nc_benchmark(
            train_dataset=clickme_train,  # type: ignore
            test_dataset=clickme_test,  # type: ignore
            n_experiences=n_experiences,
            task_labels=return_task_id,
            seed=seed,
            fixed_class_order=fixed_class_order,
            shuffle=shuffle,
            per_exp_classes=None,
            class_ids_from_zero_in_each_exp=class_ids_from_zero_in_each_exp,
            train_transform=train_transform,
            eval_transform=eval_transform,
        )

    # Actual benchmark
    else:
        clickme_train = make_combined_clickme_dataset(
            imagenet_root="/imagenet" if include_imagenet else None,
            imagenet_split="train" if include_imagenet else None,
            clickme_root=root,
            clickme_split="train",
        )
        #
        clickme_val = make_combined_clickme_dataset(
            imagenet_root=None,
            imagenet_split=None,
            clickme_root=root,
            clickme_split="val",
        )
        # Imagenet split is val because we don't have the test split
        clickme_test = make_combined_clickme_dataset(
            imagenet_root="/imagenet" if include_imagenet else None,
            imagenet_split="val" if include_imagenet else None,
            clickme_root=root,
            clickme_split="test",
        )

        # Make a benchmark with train stream
        benchmark_with_train = nc_benchmark(
            train_dataset=clickme_train,  # type: ignore
            test_dataset=clickme_test,  # type: ignore
            n_experiences=n_experiences,
            task_labels=return_task_id,
            seed=seed,
            fixed_class_order=fixed_class_order,
            shuffle=shuffle,
            per_exp_classes=None,
            class_ids_from_zero_in_each_exp=class_ids_from_zero_in_each_exp,
            train_transform=train_transform,
            eval_transform=eval_transform,
        )

        # Make a benchmark with val stream
        benchmark_with_val = nc_benchmark(
            train_dataset=clickme_val,  # type: ignore
            test_dataset=clickme_test,  # type: ignore
            n_experiences=n_experiences,
            task_labels=return_task_id,
            seed=seed,
            fixed_class_order=fixed_class_order,
            shuffle=shuffle,
            per_exp_classes=None,
            class_ids_from_zero_in_each_exp=class_ids_from_zero_in_each_exp,
            train_transform=train_transform,
            eval_transform=eval_transform,
        )

        # Frankenstein the two benchmarks
        # We grab the two train streams and one of the two test streams
        # The naming below might be confusing, so read carefully
        # I will write "note" near places where it might be confusing
        val_stream = benchmark_with_val.train_stream

        # We need to do a bit of surgery here
        # cannot just val_stream.name = "val" and call it a day :/
        # Most of this is adapted from the avalanche 0.4.0 implementation of
        # `benchmark_with_validation_stream`, later versions use new implementation

        # Get stream definitions from benchmark with train
        # and create new one based on that
        benchmark_with_train_stream_definitions: Dict[
            str, StreamDef[AvalancheDataset]  # type: ignore
        ] = benchmark_with_train.stream_definitions  # type: ignore

        new_stream_definitions: Dict[
            str, Union[StreamUserDef[AvalancheDataset], StreamDef[AvalancheDataset]]  # type: ignore
        ] = dict(benchmark_with_train_stream_definitions)

        # Get stream definitions from benchmark with val
        # and use it to grab task_labels
        benchmark_with_val_stream_definitions: Dict[
            str, StreamDef[AvalancheDataset]  # type: ignore
        ] = benchmark_with_val.stream_definitions  # type: ignore

        val_exps_tasks_labels = list(
            benchmark_with_val_stream_definitions[
                "train"  # note
            ].exps_task_labels
        )

        # Get val experiences
        valid_exps_source: Union[
            Iterable[AvalancheDataset], Tuple[Iterable[AvalancheDataset], int]  # type: ignore
        ] = []
        for val_exp in val_stream:
            valid_exps_source.append(val_exp.dataset)

        # Make new stream definition for val stream
        val_stream_def: StreamUserDef[AvalancheDataset] = StreamUserDef(  # type: ignore
            valid_exps_source,
            val_exps_tasks_labels,
            benchmark_with_val_stream_definitions[
                "train"  # note
            ].origin_dataset,
            False,
        )

        # Add the new val stream def
        new_stream_definitions["val"] = val_stream_def  # type: ignore

        # Grab setting
        complete_test_set_only = benchmark_with_train.complete_test_set_only

        return ClassificationScenario(
            stream_definitions=new_stream_definitions,
            complete_test_set_only=complete_test_set_only,
            stream_factory=NCStream,  # type: ignore
            experience_factory=NCExperience,
        )
