import argparse
import json
import shortuuid
import importlib.util

import GPUtil
import torch

import pluggy

from avalanche.logging import InteractiveLogger, WandBLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.evaluation.metrics import accuracy_metrics
from avalanche.training.supervised.joint_training import JointTraining

import spec, defaults
from utils import (
    DS_SEED,
    MODEL_SEEDS,
    generate_model_save_name,
    ModelSaverPlugin,
)

"""This script imports the given model file and executes training 
in accordance with the configuration file 
"""


def parse_args():
    parser = argparse.ArgumentParser(description="Train a model.")

    # Grabs arguments from API
    parser.add_argument(
        "hook_implementation",
        type=str,
        help="Path with your hook implementations to drive the training script.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="Path to save models",
    )
    parser.add_argument(
        "--use_wandb", action="store_true", default=True, help="Uses WandB logging"
    )
    args = parser.parse_args()
    return args


def main():
    # Args from the CLI interface
    args = parse_args()

    # Initialize Plugin manager
    pm = pluggy.PluginManager("continualTrain")
    pm.add_hookspecs(spec)

    # Register default implementations first
    pm.register(defaults)

    # Dynamic loading of the provided plugin
    plugin_spec = importlib.util.spec_from_file_location(
        "user_plugin", args.hook_implementation
    )
    plugin_module = importlib.util.module_from_spec(plugin_spec)
    plugin_spec.loader.exec_module(plugin_module)

    # Register the loaded plugin with pluggy
    pm.register(plugin_module)

    # Generate a UUID for logging purposes
    rand_uuid = str(shortuuid.uuid())[:4]

    # Get metadata from pluggy
    metadata = pm.hook.get_metadata()

    # Validate that all required keys are present
    missing_keys = spec.required_metadata_keys - metadata.keys()

    if missing_keys:
        raise ValueError(
            f"Metadata does not have the required keys. Missing: {missing_keys}"
        )

    # Extract setup metadata
    strategy_name = metadata["strategy_name"]
    dataset_name = metadata["dataset_name"]
    model_name = metadata["model_name"]
    wandb_entity = metadata["wandb_entity"]
    wandb_project_name = metadata["wandb_project_name"]

    # Get a GPU
    try:
        deviceID = GPUtil.getFirstAvailable(
            order="memory",
            maxLoad=0.85,
            maxMemory=0.85,
            attempts=1,
            interval=15,
            verbose=False,
        )
    except RuntimeError as e:
        print(f"Error obtaining GPU: {e}")
        deviceID = None
    device = torch.device(f"cuda:{deviceID[0]}" if deviceID is not None else "cpu")

    # Set up printing locally
    interactive_logger = InteractiveLogger()

    # Benchmark obtained from the plugin manager
    dataset = pm.hook.get_dataset(root_path="/datasets", seed=DS_SEED)
    train_stream = dataset.train_stream
    test_stream = dataset.test_stream

    # TRAINING
    for model_seed in MODEL_SEEDS:
        # Set up logging
        loggers = [interactive_logger]
        if args.use_wandb:
            # Run name
            run_name = f"{rand_uuid}_seed{model_seed}_{strategy_name}_{dataset_name}"
            wandb_params = {"entity": wandb_entity, "name": run_name}
            wandb_logger = WandBLogger(
                project_name=wandb_project_name,
                params=wandb_params,
                # config=config_dict,
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

        # Model from plugin manager
        model = pm.hook.get_model(device=device, seed=model_seed)

        # Training configuration
        optimizer = pm.hook.get_optimizer(parameters=model.parameters())
        criterion = pm.hook.get_criterion()

        # Training strategy
        model_saver_plugin = ModelSaverPlugin(
            save_frequency=10,
            strategy_name=strategy_name,
            rand_uuid=rand_uuid,
            save_path=args.save_path,
        )

        cl_strategy = pm.hook.get_strategy(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            evaluator=eval_plugin,
            plugins=[model_saver_plugin],
            device=device,
        )

        # Train and test loop
        results = []

        if isinstance(cl_strategy, JointTraining):
            cl_strategy.train(train_stream)
            results.append(cl_strategy.eval(test_stream))
            save_name = generate_model_save_name(
                save_path=args.save_path,
                strategy=strategy_name,
                rand_uuid=rand_uuid,
                experience=0,
                epoch=cl_strategy.train_epochs,
            )
            model.save_weights(save_name)
        else:
            for i, experience in enumerate(train_stream):
                print("Start of experience: ", experience.current_experience)
                print("Current Classes: ", experience.classes_in_this_experience)

                cl_strategy.train(experience)
                print("Training completed")

                print("Computing accuracy on the whole test set")
                results.append(cl_strategy.eval(test_stream))
                save_name = generate_model_save_name(
                    save_path=args.save_path,
                    strategy=strategy_name,
                    rand_uuid=rand_uuid,
                    experience=i,
                    epoch=cl_strategy.train_epochs,
                )
                model.save_weights(save_name)

        if args.use_wandb:
            wandb_logger.wandb.finish()


if __name__ == "__main__":
    main()
