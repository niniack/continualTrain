import argparse
import importlib.util
import json
import sys

import defaults
import GPUtil
import pluggy
import profiler
import shortuuid
import spec
import torch
from avalanche.evaluation.metrics import accuracy_metrics
from avalanche.logging import InteractiveLogger, WandBLogger
from avalanche.training.plugins import EvaluationPlugin, LRSchedulerPlugin
from avalanche.training.supervised.joint_training import JointTraining
from torch.profiler import profile, record_function
from utils import (
    DS_SEED,
    MODEL_SEEDS,
    ModelSaverPlugin,
    generate_model_save_name,
    verify_model_save_weights,
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
    parser.add_argument("--use_wandb", action="store_true", help="Uses WandB logging")
    parser.add_argument(
        "--train_experiences", type=int, help="Number of experiences to train on"
    )
    parser.add_argument(
        "--eval_experiences", type=int, help="Number of experiences to evaluate on"
    )
    parser.add_argument(
        "--exclude_gpus", nargs="*", type=int, help="The Device IDs of GPUs to exclude"
    )
    parser.add_argument(
        "--profile", action="store_true", help="Profile the main training loop."
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
            maxLoad=100,
            maxMemory=0.85,
            attempts=5,
            interval=5,
            verbose=False,
            excludeID=[],
        )
    except RuntimeError as e:
        sys.exit(f"Error obtaining GPU: {e}")

    device = torch.device(f"cuda:{deviceID[0]}" if deviceID is not None else "cpu")

    # Verify model can save
    temp_model = pm.hook.get_model(device=device, seed=0)
    if not verify_model_save_weights(temp_model):
        print("Error: Model does not have a save_weights method.")
        del temp_model  # cleanup
        sys.exit(1)  # exit the program with an error code
    del temp_model
    torch.cuda.empty_cache()

    # Set up printing locally
    interactive_logger = InteractiveLogger()

    # Benchmark obtained from the plugin manager
    dataset = pm.hook.get_dataset(root_path="/datasets", seed=DS_SEED)
    train_stream = (
        dataset.train_stream[: int(args.train_experiences)]
        if hasattr(args, "train_experiences")
        else dataset.train_stream
    )
    test_stream = (
        dataset.test_stream[: int(args.eval_experiences)]
        if hasattr(args, "eval_experiences")
        else dataset.test_stream
    )

    # Defaults to none
    val_stream = None
    if hasattr(dataset, "val_stream"):
        val_stream = (
            dataset.val_stream[: int(args.train_experiences)]
            if hasattr(args, "train_experiences")
            else dataset.val_stream
        )
    else:
        print("No validation set!")

    # Collate function
    collate_fn = pm.hook.get_collate()

    # TRAINING
    for model_seed in MODEL_SEEDS:
        # Set up interactive logging
        loggers = [interactive_logger]

        # Set up evaluator, which accepts loggers
        eval_plugin = pm.hook.get_evaluator(loggers=loggers)

        # Model from plugin manager
        model = pm.hook.get_model(device=device, seed=model_seed)

        # Set up Avalanche strategy plugins
        model_saver_plugin = ModelSaverPlugin(
            save_frequency=10,
            strategy_name=strategy_name,
            rand_uuid=rand_uuid,
            save_path=args.save_path,
        )
        plugins = [model_saver_plugin]

        # Criterion for training
        criterion = pm.hook.get_criterion()

        # Optimizer and scheduling
        optimizer = pm.hook.get_optimizer(parameters=model.parameters())
        scheduler = pm.hook.get_scheduler(optimizer=optimizer)

        # Add LR scheduler to plugins, if it exists
        if scheduler:
            plugins.append(scheduler)

        # Training strategy
        cl_strategy = pm.hook.get_strategy(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            evaluator=eval_plugin,
            plugins=plugins,
            device=device,
        )

        # Set up wandb logging, if requested
        if args.use_wandb:
            run_name = f"{rand_uuid}_seed{model_seed}_{strategy_name}_{dataset_name}"
            wandb_params = {"entity": wandb_entity, "name": run_name}

            # Extract and log common hyperparameters
            wandb_config_dict = {
                "init_lr": optimizer.param_groups[0]["lr"],
                "minibatch_size": cl_strategy.train_mb_size,
                "epochs": cl_strategy.train_epochs,
                "model_seed": model_seed,
                "gpu_ID": deviceID[0],
                "model_class": type(model).__name__,
                "optimizer_type": type(optimizer).__name__,
                "experiences": int(dataset.n_experiences),
                "scheduler_type": scheduler.scheduler.__name__,
                "nh_weight": metadata["nh_weight"] if "nh_weight" in metadata else 0,
            }

            for key, value in optimizer.defaults.items():
                wandb_config_dict[f"optimizer_{key}"] = value

            wandb_logger = WandBLogger(
                project_name=wandb_project_name,
                params=wandb_params,
                config=wandb_config_dict,
            )

            loggers.append(wandb_logger)

        # Train and test loop
        results = []
        with profiler.conditional_profiler(
            args.profile,
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
        ) as prof:
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
                val_stream_iter = iter(val_stream) if val_stream is not None else None
                for exp_id, experience in enumerate(train_stream):
                    # Get val exp, if exists
                    if val_stream is not None:
                        val_exp = val_stream[exp_id]

                    # Invoke strategy train method
                    print("Start of experience: ", experience.current_experience)
                    print("Current Classes: ", experience.classes_in_this_experience)

                    cl_strategy.train(
                        experience,
                        eval_streams=[val_exp] if val_stream is not None else [],
                        num_workers=8,
                        collate_fn=collate_fn,
                    )
                    print("Training completed")
                    # LR Scheduler will reset here

                    # Invoke strategy evaluation method
                    print("Evaluating experiences")
                    results.append(cl_strategy.eval(test_stream[exp_id]))

                    # Save model
                    save_name = generate_model_save_name(
                        save_path=args.save_path,
                        strategy=strategy_name,
                        rand_uuid=rand_uuid,
                        experience=exp_id,
                        epoch=cl_strategy.train_epochs,
                    )
                    model.save_weights(save_name)

        profiler.print_profiler_results(prof)

        if args.use_wandb:
            wandb_logger.wandb.finish()


if __name__ == "__main__":
    main()
