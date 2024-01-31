import torch.nn.functional as F
import torchattacks
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin

from continualUtils.explain.losses import SaliencyGuidedLoss
from continualUtils.security.losses import AdversarialAttackLoss


class MedicalAdversarialSaliency(SupervisedPlugin):
    """
    Adversarial Saliency Guided Training from the medical paper
    https://arxiv.org/pdf/2209.04326.pdf
    """

    def __init__(
        self,
        features_dropped,
        attack,
        attack_params,
        weight=1,
    ):
        super().__init__()
        self.weight = weight
        self.sg_loss = SaliencyGuidedLoss(
            random_masking=True,
            features_dropped=features_dropped,
            add_noise=False,
        )
        self.attack_loss = AdversarialAttackLoss(
            attack=attack,
            attack_params=attack_params,
        )

    def before_backward(self, strategy, **kwargs):
        # Clone the input for adversarial attack
        cloned_mb_x = strategy.mb_x.clone().detach()

        # Uses the `__call__` method
        # NOTE: Calling saliency guided loss with adversarial examples
        # and output of model from adversarial examples
        _, masked_images = self.sg_loss(
            mb_x=cloned_mb_x,
            mb_y=strategy.mb_y,
            mb_tasks=strategy.mbatch[-1],
            model=strategy.model,
        )

        _, adversarial_masked_images, _ = self.attack_loss(
            mb_x=masked_images,
            mb_y=strategy.mb_y,
            mb_tasks=strategy.mbatch[-1],
            model=strategy.model,
        )

        # Feed into model
        mb_tasks = strategy.mbatch[-1]
        masked_output = F.log_softmax(
            strategy.model(adversarial_masked_images, mb_tasks), dim=1
        )
        standard_output = F.log_softmax(strategy.mb_output, dim=1)

        # KL Loss will be added to main loss
        adversarial_loss = F.kl_div(
            masked_output,
            standard_output,
            reduction="batchmean",
            log_target=True,
        )

        # Add the harmonizer loss
        strategy.loss += adversarial_loss
        strategy.adversarial_loss = adversarial_loss

    def after_training_exp(self, strategy, **kwargs):
        """Some update that needs to happen after each training experience"""
        self.sg_loss.update(strategy.experience, strategy.model)
