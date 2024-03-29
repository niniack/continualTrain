from typing import Callable, List, Optional, Tuple, Union

import pluggy
from avalanche.training.plugins import EvaluationPlugin, LRSchedulerPlugin
from torch import device
from torch.nn import Module
from torch.optim.optimizer import Optimizer

hookspec = pluggy.HookspecMarker("continualTrain")


@hookspec(firstresult=True)
def get_collate() -> Callable:
    """
    Retrieves the collate fn for each experience.
    """


@hookspec(firstresult=True)
def get_dataset(root_path: str, seed: int) -> object:
    """
    Retrieves the dataset for the training process.

    :param root_path: The root path where the dataset is located or should be downloaded to.
    :param seed: Random seed for any dataset-related operations that need it.
    :return: A dataset object.
    """


@hookspec(firstresult=True)
def get_model(device: str, seed: int) -> Module:
    """
    Constructs and retrieves the model for the training process.

    :param device: The device (e.g., "cuda", "cpu") where the model will be placed.
    :param seed: Random seed for model initialization.
    :return: A model object.
    """


@hookspec(firstresult=True)
def get_device(available_id: int) -> Union[str, device]:
    """
    The device (e.g., "cuda", "cpu") where training will take place.

    :return: Device
    """


@hookspec(firstresult=True)
def get_strategy(
    model: Module,
    optimizer: Optimizer,
    criterion: Module,
    evaluator: any,
    plugins: List[any],
    device: str,
) -> any:
    """
    Constructs and retrieves the training strategy.

    :param model: The model object.
    :param optimizer: The optimizer object.
    :param criterion: The loss criterion object.
    :param evaluator: An evaluator object for strategy-related evaluations.
    :param plugins: List of additional plugins the strategy takes in
    :param device: The device (e.g., "cuda", "cpu") where operations will be conducted.
    :return: A strategy object.
    """


@hookspec(firstresult=True)
def get_optimizer(parameters: Union[List, Tuple]) -> Optimizer:
    """
    Constructs and retrieves the optimizer for the training process.

    :param parameters: Model parameters to be optimized.
    :return: An optimizer object.
    """


@hookspec(firstresult=True)
def get_scheduler(
    optimizer: Union[Optimizer, None]
) -> Optional[LRSchedulerPlugin]:
    """
    Constructs and retrieves a scheduler

    :param parameters: Torch optimizer
    :return: Avalanche LR Scheduler
    """


@hookspec(firstresult=True)
def get_criterion() -> Module:
    """
    Constructs and retrieves the criterion (loss function) for the training process.

    :return: A loss criterion object.
    """


@hookspec(firstresult=True)
def get_ffcv_decoder_pipeline() -> dict:
    """
    Constructs a custom decoder pipeline for FFCv

    :return: A loss criterion object.
    """


@hookspec(firstresult=True)
def get_evaluator(loggers) -> EvaluationPlugin:
    """Constructs and retreives the evaluator plugin

    :param loggers: The loggers to use with the evaluator
    """


required_metadata_keys = {
    "strategy_name",
    "dataset_name",
    "wandb_entity",
    "wandb_project_name",
}


@hookspec(firstresult=True)
def get_metadata() -> dict:
    """
    Retrieves metadata information.

    :return: A dictionary containing:
             - 'strategy_name': Name of the strategy.
             - 'dataset_name': Name of the dataset.
             - 'wandb_entity': WandB user.
             - 'wandb_project_name': WandB project.
    """
