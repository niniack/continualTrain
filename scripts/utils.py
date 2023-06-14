from enum import Enum
import argparse
import inspect
import importlib.util

from avalanche.benchmarks.classic import SplitCIFAR100
from avalanche.training.supervised import Naive, JointTraining, Cumulative
from avalanche.training.plugins import EWCPlugin, MASPlugin, ReplayPlugin, RWalkPlugin
from avalanche.training.storage_policy import ClassBalancedBuffer

DS_SEED = 42


def parse_args():
    parser = argparse.ArgumentParser(description="Train a model.")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to JSON model config file"
    )
    parser.add_argument(
        "--model_file",
        type=str,
        required=True,
        help="Path to Python file defining model",
    )
    args = parser.parse_args()
    return args


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


class Strategy(Enum):
    JOINT = "joint"
    NAIVE = "naive"
    CUMULATIVE = "cumulative"
    EWC = "ewc"
    REPLAY = "replay"
    MAS = "mas"
    RWALK = "rwalk"


class Dataset(Enum):
    SPLITCIFAR100 = "splitcifar100"


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

    return dataset


def build_strategy(
    config, kwarg_dict, model, optimizer, criterion, eval_plugin, device
):
    strategy = config.strategy
    batch_size = config.batch_size
    epochs = config.epochs

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
        )
    else:
        plugins = []

        if strategy == Strategy.EWC:
            plugins.append(EWCPlugin(ewc_lambda=kwarg_dict["ewc_lambda"]))

        elif strategy == Strategy.RWALK:
            plugins.append(
                RWalkPlugin(
                    ewc_lambda=kwarg_dict["rwalk_lambda"],
                    ewc_alpha=kwarg_dict["rwalk_alpha"],
                    delta_t=kwarg_dict["rwalk_delta_t"],
                )
            )

        elif strategy == Strategy.MAS:
            plugins.append(
                MASPlugin(
                    lambda_reg=kwarg_dict["mas_lambda"], alpha=kwarg_dict["mas_alpha"]
                )
            )
        elif strategy == Strategy.REPLAY:
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
