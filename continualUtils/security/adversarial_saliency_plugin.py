import torchattacks
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin

from continualUtils.explain.losses import SaliencyGuidedLoss
from continualUtils.security.losses import AdversarialAttackLoss


class AdversarialSaliency(SupervisedPlugin):
    """Adversarial Saliency Guided Training plugin"""

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

        # Generate advesarial examples
        (
            adversarial_loss,
            adversarial_images,
            adversarial_preds,
        ) = self.attack_loss(
            mb_x=cloned_mb_x,
            mb_y=strategy.mb_y,
            mb_tasks=strategy.mbatch[-1],
            model=strategy.model,
        )

        # Turn on gradients for the adversarially attacked images
        if adversarial_images.requires_grad is False:
            adversarial_images.requires_grad_(True)

        # Uses the `__call__` method
        # NOTE: Calling saliency guided loss with adversarial examples
        # and output of model from adversarial examples
        sg_loss = self.sg_loss(
            mb_x=adversarial_images,
            mb_y=adversarial_preds.argmax(dim=1),
            mb_tasks=strategy.mbatch[-1],
            model=strategy.model,
            mb_output=adversarial_preds,
        )

        # Turn off gradients
        adversarial_images.requires_grad_(False)

        # Add the harmonizer loss
        strategy.loss += adversarial_loss + sg_loss
        strategy.saliency_guided_loss = sg_loss
        strategy.adversarial_loss = adversarial_loss

    def after_training_exp(self, strategy, **kwargs):
        """Some update that needs to happen after each training experience"""
        self.sg_loss.update(strategy.experience, strategy.model)
