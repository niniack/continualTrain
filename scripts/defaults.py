import pluggy
from torch.nn import CrossEntropyLoss
from torch.optim import SGD

hookimpl = pluggy.HookimplMarker("continualTrain")


@hookimpl
def get_optimizer(parameters):
    optimizer = SGD(parameters, lr=0.01, momentum=0.9, weight_decay=1e-4)
    return optimizer


@hookimpl
def get_criterion():
    loss = CrossEntropyLoss()
    return loss
