import pluggy
from avalanche.evaluation.metrics import accuracy_metrics
from avalanche.training.plugins import EvaluationPlugin
from torch.nn import CrossEntropyLoss
from torch.optim import SGD

hookimpl = pluggy.HookimplMarker("continualTrain")


@hookimpl
def get_optimizer(parameters):
    optimizer = SGD(parameters, lr=0.01, momentum=0.9, weight_decay=1e-4)
    return optimizer


@hookimpl
def get_scheduler(optimizer):
    return (None, None)


@hookimpl
def get_criterion():
    loss = CrossEntropyLoss()
    return loss


@hookimpl
def get_evaluator(loggers):
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
