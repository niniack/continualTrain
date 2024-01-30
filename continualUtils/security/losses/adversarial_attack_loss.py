import torch
import torch.nn.functional as F
import torchattacks
from avalanche.training.regularization import RegularizationMethod

from continualUtils.benchmarks.datasets.normalizations import *


class AdversarialAttackLoss(RegularizationMethod):
    def __init__(self, attack, attack_params):
        self.attack_method = attack
        self.attack_params = attack_params
        self.attack = None

    def update(self, *args, **kwargs):
        pass

    def __call__(self, mb_x, mb_y, mb_tasks, model, *args, **kwargs):
        """Call the adversarial attack loss

        :param mb_x: NCHW input batch
        :param mb_y: N labels
        :param mb_tasks: N tasks (per sample)
        :param model: Model
        :return: Returns adversarial loss and adversarially attacked images
        """
        # Set up attack
        self.attack = self.attack_method(model, **self.attack_params)

        # Set normalization
        # mean and std must be lists (not numpy arrays)
        self.attack.set_normalization_used(mean=CIFAR10_MEAN, std=CIFAR10_STD)

        # Generate adversarial images
        adv_images = self.attack(mb_x, mb_y)

        # Forward with adversarial images
        adv_out = model(adv_images, mb_tasks)

        # Compute loss
        adv_loss = F.cross_entropy(adv_out, mb_y)

        return adv_loss, adv_images, adv_out
