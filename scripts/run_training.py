from pathlib import Path
import argparse
import json
import shortuuid

import GPUtil
import torch
from torch.optim import SGD
from torch.nn import CrossEntropyLoss

import avalanche
from avalanche.logging import InteractiveLogger, WandBLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.evaluation.metrics import accuracy_metrics
from avalanche.training.supervised import Naive, Cumulative, EWC, MAS
from avalanche.training.plugins import ReplayPlugin

from scripts.parse_config import parse_config
from scripts.utils import (
    Strategy,
    DS_SEED,
    build_strategy,
    build_dataset,
    import_module_from_file,
)

"""This script imports the given model file and executes training 
in accordance with the configuration file 
"""


def parse_args():
    parser = argparse.ArgumentParser(description="Train a model.")
    parser.add_argument(
        "--config_file", type=str, required=True, help="Path to JSON model config file"
    )
    parser.add_argument(
        "--model_file",
        type=str,
        required=True,
        help="Path to Python file defining model",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="Path to save models",
    )
    args = parser.parse_args()
    return args


def main():
    rand_uuid = str(shortuuid.uuid())

    # ARGS
    args = parse_args()

    # CONFIGURATION
    config, kwarg_dict = parse_config(Path(f"{args.config_file}"))
    try:
        deviceID = GPUtil.getFirstAvailable(
            order="load",
            maxLoad=0,
            maxMemory=0,
            attempts=1,
            interval=15,
            verbose=False,
        )
    except:
        deviceID = None
    device = torch.device(f"cuda:{deviceID[0]}" if deviceID else "cpu")

    interactive_logger = InteractiveLogger()
    loggers = [interactive_logger]
    if config.use_wandb:
        # Get config dict to save on wandb
        with open(args.config_file, "r") as f:
            config_dict = json.load(f)

        # Run name
        run_name = f"{rand_uuid}_{config.strategy.name}_{config.dataset_name.name}"
        wandb_params = {"entity": config.wandb_entity, "name": run_name}
        wandb_logger = WandBLogger(
            project_name=config.wandb_project, params=wandb_params, config=config_dict
        )
        loggers.append(wandb_logger)

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(
            minibatch=False,
            epoch=True,
            epoch_running=False,
            experience=True,
            stream=False,
        ),
        loggers=loggers,
    )

    # Benchmark with 10 experiences
    dataset = build_dataset(config, kwarg_dict)
    train_stream = dataset.train_stream
    test_stream = dataset.test_stream

    # TRAINING
    for model_seed in config.model_seeds:
        # # Model
        ModelClass = import_module_from_file(Path(f"{args.model_file}"))
        model = ModelClass(
            num_classes=100,
            device=device,
            seed=model_seed,
            multihead=config.use_multihead,
        )

        # Training configuration
        momentum = getattr(config, "momentum", 0.9)
        weight_decay = getattr(config, "weight_decay", 1e-4)
        optimizer = SGD(
            model.parameters(),
            lr=config.learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
        )
        criterion = CrossEntropyLoss()

        # Training strategy
        cl_strategy = build_strategy(
            config, kwarg_dict, model, optimizer, criterion, eval_plugin, device
        )

        # Train and test loop
        results = []

        if config.strategy == Strategy.JOINT:
            cl_strategy.train(train_stream)
            results.append(cl_strategy.eval(test_stream))
            model.save_weights(
                f"{args.save_path}/{config.strategy}/{rand_uuid}/experience_0"
            )
        else:
            for i, experience in enumerate(train_stream):
                print("Start of experience: ", experience.current_experience)
                print("Current Classes: ", experience.classes_in_this_experience)

                cl_strategy.train(experience)
                print("Training completed")

                print("Computing accuracy on the whole test set")
                results.append(cl_strategy.eval(test_stream))
                model.save_weights(
                    f"{args.save_path}/{config.strategy}/{rand_uuid}/experience_0"
                )


if __name__ == "__main__":
    main()
