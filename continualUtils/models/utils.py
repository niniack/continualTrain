import torch
import torch.nn as nn
from avalanche.models.dynamic_modules import (
    MultiHeadClassifier,
    MultiTaskModule,
)


# These methods are adapted from Avalanche to extend compatibility
class MultiTaskDecorator(MultiTaskModule):
    """
    Encapsulates an existing nn.Module to make it subclass MultiTaskModule,
    the user should still be able to interact with the encapsulated module
    as if it was the module itself.

    The only things that change are the following, the classifier from the
    given model will be replaced by a MultiHeadClassifier, and the forward()
    implementation will be overwritten by one that accepts task labels.
    The encapsulated module will then be automatically extended to
    fit new classes during calls to model.adaptation()
    """

    def __init__(self, model: nn.Module, classifier_name: str):
        """
        :param model: pytorch nn.Module that does not support multitask
        :param classifier_name: attribute name of the existing classification
                                layer inside the module
        """
        for m in model.modules():
            assert not isinstance(m, MultiTaskModule)

        self.__dict__["_initialized"] = False
        super().__init__()
        self.model = model
        self.classifier_name = classifier_name

        old_classifier = getattr(model, classifier_name)

        if isinstance(old_classifier, nn.Linear):
            in_size = old_classifier.in_features
            out_size = old_classifier.out_features
            old_params = [
                torch.clone(p.data) for p in old_classifier.parameters()
            ]
            # Replace old classifier by empty block
            setattr(self.model, classifier_name, nn.Sequential())
        elif isinstance(old_classifier, nn.Sequential):
            in_size = old_classifier[-1].in_features
            out_size = old_classifier[-1].out_features
            old_params = [
                torch.clone(p.data) for p in old_classifier[-1].parameters()
            ]
            del old_classifier[-1]
        else:
            raise NotImplementedError(
                f"Cannot handle the following type \
            of classification layer {type(old_classifier)}"
            )

        # Set new classifier and initialize to previous param values
        setattr(self, classifier_name, MultiHeadClassifier(in_size, out_size))

        for param, param_old in zip(
            getattr(self, classifier_name).parameters(), old_params
        ):
            param.data = param_old

        self.max_class_label = max(self.max_class_label, out_size)
        self._initialized = True

    def forward(
        self, x: torch.Tensor, task_labels: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor:
        """Compute the output given the input `x` and task labels.

        :param x:
        :param task_labels: task labels for each sample. if None, the
            computation will return all the possible outputs as a dictionary
            with task IDs as keys and the output of the corresponding task as
            output.
        :return:
        """
        if task_labels is None:
            return self.forward_all_tasks(x, *args, **kwargs)

        if isinstance(task_labels, int):
            # fast path. mini-batch is single task.
            return self.forward_single_task(x, task_labels, *args, **kwargs)
        else:
            unique_tasks = torch.unique(task_labels)

        out = torch.zeros(x.shape[0], self.max_class_label, device=x.device)
        for task in unique_tasks:
            task_mask = task_labels == task
            x_task = x[task_mask]
            out_task = self.forward_single_task(
                x_task, task.item(), *args, **kwargs
            )
            assert len(out_task.shape) == 2, (
                "multi-head assumes mini-batches of 2 dimensions "
                "<batch, classes>"
            )
            n_labels_head = out_task.shape[1]
            out[task_mask, :n_labels_head] = out_task
        return out

    def forward_single_task(
        self, x: torch.Tensor, task_label: int, *args, **kwargs
    ):
        out = self.model(x, *args, **kwargs)
        hidden_states = None
        if isinstance(out, tuple):
            out = out[0]
            hidden_states = out[1] if len(out) > 1 else None
        elif isinstance(out, dict) and "logits" in out:
            out = out["logits"]
            out = out.get("hidden_states", None)
        return getattr(self, self.classifier_name)(
            out.view(out.size(0), -1), task_labels=task_label
        )

    def forward_all_tasks(self, x: torch.Tensor, *args, **kwargs):
        """compute the output given the input `x` and task label.
        By default, it considers only tasks seen at training time.

        :param x:
        :return: all the possible outputs are returned as a dictionary
            with task IDs as keys and the output of the corresponding
            task as output.
        """
        out = self.model(x, *args, **kwargs)
        hidden_states = None
        if isinstance(out, tuple):
            out = out[0]
            hidden_states = out[1] if len(out) > 1 else None
        elif isinstance(out, dict) and "logits" in out:
            out = out["logits"]
            out = out.get("hidden_states", None)
        return getattr(self, self.classifier_name)(
            out.view(out.size(0), -1), task_labels=None
        )

    def __getattr__(self, name):
        # Override pytorch impl from nn.Module

        # Its a bit particular since pytorch nn.Module does not
        # keep some attributes in a classical manner in self.__dict__
        # rather it puts them into _parameters, _buffers and
        # _modules attributes. We have to add these lines to avoid recursion
        if name == "model":
            return self.__dict__["_modules"]["model"]
        if name == self.classifier_name:
            return self.__dict__["_modules"][self.classifier_name]

        # If its a different attribute, return the one from the model
        return getattr(self.model, name)

    def __setattr__(self, name, value):
        # During initialization, use pytorch routine
        if not self.__dict__["_initialized"] or name in self.__dict__:
            super().__setattr__(name, value)
        else:
            return setattr(self.model, name, value)


def as_multitask(model: nn.Module, classifier_name: str) -> MultiTaskModule:
    """Wraps around a model to make it a multitask model.

    :param model: model to be converted into MultiTaskModule
    :param classifier_name: the name of the attribute containing
                            the classification layer (nn.Linear). It can also
                            be an instance of nn.Sequential containing multiple
                            layers as long as the classification layer is the
                            last layer.
    :return: the decorated model, now subclassing MultiTaskModule, and
        accepting task_labels as forward() method argument
    """
    return MultiTaskDecorator(model, classifier_name)


__all__ = ["as_multitask"]
