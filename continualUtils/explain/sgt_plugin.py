from avalanche.training.plugins.strategy_plugin import SupervisedPlugin

from continualUtils.explain.losses import SaliencyGuidedLoss


class SaliencyGuidedPlugin(SupervisedPlugin):
    """Saliency Guided Training plugin"""

    def __init__(
        self,
        noisy_features,
        features_dropped,
        weight=1,
    ):
        super().__init__()
        self.weight = weight
        self.sg_loss = SaliencyGuidedLoss(
            random_masking=True,
            features_dropped=features_dropped,
            add_noise=True if noisy_features > 0 else False,
            noise_mag=1e-1,
            noisy_features=noisy_features,
        )

    def before_backward(self, strategy, **kwargs):
        # Clone the input
        cloned_mb_x = strategy.mb_x.clone().detach()

        # Turn on gradients for the cloned input
        if cloned_mb_x.requires_grad is False:
            cloned_mb_x.requires_grad_(True)

        # Uses the `__call__` method
        sg_loss = self.sg_loss(
            mb_x=cloned_mb_x,
            mb_y=strategy.mb_y,
            mb_tasks=strategy.mbatch[-1],
            model=strategy.model,
        )

        cloned_mb_x.requires_grad_(False)

        # Add the harmonizer loss
        strategy.loss += sg_loss
        strategy.saliency_guided_loss = sg_loss

    def after_training_exp(self, strategy, **kwargs):
        """Some update that needs to happen after each training experience"""
        self.sg_loss.update(strategy.experience, strategy.model)

    def after_eval_forward(self, strategy, **kwargs):
        # Clone the input
        cloned_mb_x = strategy.mb_x.clone().detach()

        # Turn on gradients for the cloned input
        if cloned_mb_x.requires_grad is False:
            cloned_mb_x.requires_grad_(True)

        # Uses the `__call__` method
        sg_loss, _ = self.sg_loss(
            mb_x=cloned_mb_x,
            mb_y=strategy.mb_y,
            mb_tasks=strategy.mbatch[-1],
            model=strategy.model,
        )

        cloned_mb_x.requires_grad_(False)

        # Add the harmonizer loss
        strategy.loss += sg_loss
        strategy.saliency_guided_loss = sg_loss
