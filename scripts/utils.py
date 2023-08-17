from enum import Enum
import argparse
import inspect
import importlib.util


from avalanche.benchmarks.classic import (
    SplitCIFAR100,
    SplitCIFAR10,
    SplitMNIST,
    SplitFMNIST,
)
from avalanche.training.supervised import Naive, JointTraining, Cumulative
from avalanche.training.plugins import (
    EWCPlugin,
    MASPlugin,
    ReplayPlugin,
    RWalkPlugin,
    SupervisedPlugin,
)
from avalanche.training.storage_policy import ClassBalancedBuffer

DS_SEED = 42


class ModelSaverPlugin(SupervisedPlugin):
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


def import_module_from_file(file_path):
    spec = importlib.util.spec_from_file_location("model", file_path)
    if spec is None:
        raise ImportError(f"Couldn't load module from file: {file_path}")

    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)

    # Look for the first class in the module
    for name, obj in inspect.getmembers(model_module):
        if inspect.isclass(obj):
            return obj  # Return the class object

    # If no class is found, raise an error
    raise ImportError(f"No class found in module: {file_path}")


def generate_model_save_name(save_path, strategy, rand_uuid, experience, epoch):
    return f"{save_path}/{strategy}/{rand_uuid}/experience_{experience}_epoch_{epoch}"


class Strategy(Enum):
    JOINT = "joint"
    NAIVE = "naive"
    CUMULATIVE = "cumulative"
    EWC = "ewc"
    REPLAY = "replay"
    MAS = "mas"
    RWALK = "rwalk"
    MAS_REPLAY = "mas_replay"


class Dataset(Enum):
    SPLITCIFAR100 = "splitcifar100"
    SPLITCIFAR10 = "splitcifar10"
    SPLITMNIST = "splitmnist"
    SPLITFMNIST = "splitfmnist"


class Scenario(Enum):
    SINGLE_HEAD = "single_head"
    MULTI_HEAD = "multi_head"


def build_dataset(config, kwarg_dict):
    if config.dataset_name == Dataset.SPLITCIFAR100:
        dataset = SplitCIFAR100(
            n_experiences=config.num_experiences,
            dataset_root=config.dataset_path,
            return_task_id=True,
            shuffle=True,
            seed=config.dataset_seed,
        )

    elif config.dataset_name == Dataset.SPLITCIFAR10:
        dataset = SplitCIFAR10(
            n_experiences=config.num_experiences,
            dataset_root=config.dataset_path,
            return_task_id=True,
            shuffle=True,
            seed=config.dataset_seed,
        )

    elif config.dataset_name == Dataset.SPLITFMNIST:
        dataset = SplitFMNIST(
            n_experiences=config.num_experiences,
            dataset_root=config.dataset_path,
            return_task_id=True,
            shuffle=True,
            seed=config.dataset_seed,
        )

    elif config.dataset_name == Dataset.SPLITMNIST:
        dataset = SplitMNIST(
            n_experiences=config.num_experiences,
            dataset_root=config.dataset_path,
            return_task_id=True,
            shuffle=True,
            seed=config.dataset_seed,
        )

    return dataset


def build_strategy(
    config, kwarg_dict, model, optimizer, criterion, eval_plugin, device
):
    strategy = config.strategy
    batch_size = config.batch_size
    epochs = config.epochs
    plugins = []
    plugins.append(
        ModelSaverPlugin(
            save_frequency=config.save_frequency,
            strategy_name=config.strategy,
            rand_uuid=kwarg_dict["rand_uuid"],
            save_path=kwarg_dict["save_path"],
        )
    )

    # Training strategy
    if strategy == Strategy.JOINT:
        cl_strategy = JointTraining(
            model,
            optimizer,
            criterion,
            train_mb_size=batch_size,
            train_epochs=epochs,
            eval_mb_size=batch_size,
            evaluator=eval_plugin,
            device=device,
        )
    if strategy == Strategy.CUMULATIVE:
        cl_strategy = Cumulative(
            model,
            optimizer,
            criterion,
            train_mb_size=batch_size,
            train_epochs=epochs,
            eval_mb_size=batch_size,
            evaluator=eval_plugin,
            device=device,
            plugins=plugins,
        )
    else:
        if strategy == Strategy.EWC:
            apply_ewc_plugin(kwarg_dict, plugins)
        elif strategy == Strategy.RWALK:
            apply_rwalk_plugin(kwarg_dict, plugins)
        elif strategy == Strategy.MAS:
            apply_mas_plugin(kwarg_dict, plugins)
        elif strategy == Strategy.REPLAY:
            apply_replay_plugin(kwarg_dict, plugins)
        elif strategy == Strategy.MAS_REPLAY:
            apply_mas_plugin(kwarg_dict, plugins)
            apply_replay_plugin(kwarg_dict, plugins)

        cl_strategy = Naive(
            model,
            optimizer,
            criterion,
            train_mb_size=batch_size,
            train_epochs=epochs,
            eval_mb_size=batch_size,
            evaluator=eval_plugin,
            device=device,
            plugins=plugins,
        )

    return cl_strategy


def apply_ewc_plugin(kwarg_dict, plugins):
    plugins.append(EWCPlugin(ewc_lambda=kwarg_dict["ewc_lambda"]))


def apply_rwalk_plugin(kwarg_dict, plugins):
    plugins.append(
        RWalkPlugin(
            ewc_lambda=kwarg_dict["rwalk_lambda"],
            ewc_alpha=kwarg_dict["rwalk_alpha"],
            delta_t=kwarg_dict["rwalk_delta_t"],
        )
    )


def apply_mas_plugin(kwarg_dict, plugins):
    plugins.append(
        MASPlugin(lambda_reg=kwarg_dict["mas_lambda"], alpha=kwarg_dict["mas_alpha"])
    )


def apply_replay_plugin(kwarg_dict, plugins):
    storage_policy = ClassBalancedBuffer(
        max_size=kwarg_dict["replay_mem_size"],
        adaptive_size=False,
        total_num_classes=100,
    )
    plugins.append(
        ReplayPlugin(
            mem_size=kwarg_dict["replay_mem_size"],
            storage_policy=storage_policy,
        )
    )
