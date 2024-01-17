import torch
from torch import nn, optim
from torch.nn import Module
from torch.utils.data import Dataset


class AdversarialPerturbationTransform:
    """Returns"""

    def __init__(self, adv_noise):
        self.adv_noise = adv_noise

    def __call__(self, x):
        return x + self.adv_noise


class UniversalAdversarialPerturbation:
    """Class agnostic adversarial attack"""

    def __call__(
        self,
        dataset: Dataset,
        model: Module,
        learning_rate: float,
        iterations: int,
        eps: float,
        device: torch.device,
        *args,
        **kwargs
    ):
        """_summary_

        :param dataset: Dataset to adversarially perturb
        :param model: Latest model of the previous task
        :param learning_rate: Learning rate of the optimizer
        :param iterations: Number of iterations to finetune the attack
        :param eps: Epsilon range of adversarial noise
        :param device: Pytorch device
        :return: Adversarially attacked dataset
        """
        # Set model to evaluation mode
        model.eval()

        # Extract all images and labels from the dataset
        inputs, targets, *_ = zip(*dataset)
        inputs = torch.stack(inputs).to(device)
        targets = torch.tensor(targets).to(device)

        # Get the dimensions of the first image
        channels, height, width = inputs[0].shape

        # Initialize adversarial noise
        adv_noise = torch.rand((channels, height, width), device=device)
        adv_noise.requires_grad_(True)

        # Initialize loss and optimizer
        ce_loss = nn.CrossEntropyLoss()
        optimizer = optim.Adam([adv_noise], lr=learning_rate, amsgrad=True)

        for _ in range(iterations):
            # Add adversarial noise to the input
            perturbed_inputs = inputs + adv_noise.expand_as(inputs)
            output = model(perturbed_inputs)

            # Calculate the total loss and maximize
            loss = -ce_loss(output, targets).mean()
            loss.backward()

            # Update the adversarial noise and reset gradients
            optimizer.step()
            optimizer.zero_grad()

            # Clamp the adversarial noise to ensure it's within the allowed range
            adv_noise.data.clamp_(-eps, eps)

            # Clamp perturbed inputs between min and max of original inputs
            perturbed_inputs = inputs + adv_noise.expand_as(inputs)
            perturbed_inputs.data.clamp_(inputs.min(), inputs.max())

            # Extract adversarial noise again
            adv_noise = perturbed_inputs - inputs

        return AdversarialPerturbationTransform(adv_noise)
