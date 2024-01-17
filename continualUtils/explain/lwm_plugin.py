from avalanche.training.plugins.strategy_plugin import SupervisedPlugin

from continualUtils.explain.losses.lwm_loss import LwMLoss


class LwMPlugin(SupervisedPlugin):
    """Learning without Memorizing plugin"""

    def __init__(self, weight=1):
        super().__init__()
        self.weight = weight
        self.lwm = LwMLoss(weight)

    def before_backward(self, strategy, **kwargs):
        cloned_mb_x = strategy.mb_x.detach()

        if not cloned_mb_x.requires_grad:
            cloned_mb_x.requires_grad_(True)

        # Uses the `__call__` method
        lwm_loss = self.lwm(mb_x=cloned_mb_x, curr_model=strategy.model)

        cloned_mb_x.requires_grad_(False)
        cloned_mb_x.detach()

        # Add the harmonizer loss
        strategy.loss += lwm_loss
        strategy.harmonizer_loss = lwm_loss

        # Turn off gradients for the cloned input
        cloned_mb_x.requires_grad_(False)
        cloned_mb_x.detach()

    # def after_backward(self, strategy, *args, **kwargs):
    #     torch.cuda.empty_cache()

    def after_training_exp(self, strategy, **kwargs):
        """Some update that needs to happen after each training experience
        Args:
            strategy (_type_): _description_
        """
        self.lwm.update(strategy.experience, strategy.model)
