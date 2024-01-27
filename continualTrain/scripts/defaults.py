import pluggy
import torch
from avalanche.evaluation.metrics import accuracy_metrics
from avalanche.training.plugins import EvaluationPlugin
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import default_collate

hookimpl = pluggy.HookimplMarker("continualTrain")


@hookimpl
def get_collate():
    """Returns default collate"""
    return default_collate


@hookimpl
def get_optimizer(parameters):
    """Returns SGD Optimizer

    :param parameters: parameters to be optimized
    """
    optimizer = SGD(parameters, lr=0.01, momentum=0.9, weight_decay=1e-4)
    return optimizer


@hookimpl
def get_scheduler(optimizer):
    """Dummy function to return None, in place of a scheduler"""
    return None


@hookimpl
def get_device(available_id: int):
    """
    The device (e.g., "cuda", "cpu") where training will take place.

    :return: Device
    """
    return torch.device(
        f"cuda:{available_id[0]}" if available_id is not None else "cpu"
    )


@hookimpl
def get_criterion():
    """Returns cross-entropy loss"""
    loss = CrossEntropyLoss()
    return loss


@hookimpl
def get_ffcv_decoder_pipeline() -> dict:
    """Dummy function to return None, in place of a decoder pipeline"""

    return None


@hookimpl
def get_evaluator(loggers):
    """Returns standard Avalanche evaluator.
    Tracks statistics over epoch and experience.

    :param loggers: logging plugins
    """
    eval_plugin = EvaluationPlugin(
        accuracy_metrics(
            minibatch=False,
            epoch=True,
            epoch_running=False,
            experience=True,
            stream=False,
        ),
        loggers=loggers,
    )
    return eval_plugin
