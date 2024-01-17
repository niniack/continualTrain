import os
import random

import numpy as np
import torch
from avalanche.core import SupervisedPlugin

DS_SEED = 79
MODEL_SEEDS = [0]


def seed_everything(seed):
    """Seeds everything with the same seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


class ModelSaverPlugin(SupervisedPlugin):
    """
    Plugin to save a snapshot of a model with a unique name
    """

    def __init__(self, save_frequency, strategy_name, rand_uuid, save_path):
        super().__init__()
        self.save_frequency = save_frequency
        self.strategy_name = strategy_name
        self.rand_uuid = rand_uuid
        self.save_path = save_path

    def after_training_epoch(self, strategy, *args, **kwargs):
        # Save if hits frequency
        if strategy.clock.train_exp_epochs % self.save_frequency == 0:
            save_name = generate_model_save_name(
                save_path=self.save_path,
                strategy=self.strategy_name,
                rand_uuid=self.rand_uuid,
                experience=strategy.clock.train_exp_counter,
                epoch=strategy.clock.train_exp_epochs,
            )
            strategy.model.save_weights(save_name)


def generate_model_save_name(save_path, strategy, rand_uuid, experience, epoch):
    return f"{save_path}/{strategy}/{rand_uuid}/experience_{experience}_epoch_{epoch}"


def verify_model_save_weights(model):
    return hasattr(model, "save_weights") and callable(
        getattr(model, "save_weights")
    )
