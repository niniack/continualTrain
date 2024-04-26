import importlib.util
import os
import sys

import defaults
import ffcv
import GPUtil
import pluggy
import shortuuid
import spec
import tomlkit
import torch
import wandb
from avalanche.benchmarks.utils.ffcv_support import enable_ffcv
from avalanche.logging import InteractiveLogger, WandBLogger
from avalanche.training.supervised.joint_training import JointTraining
from parse_args import parse_args
from utils import (
    DS_SEED,
    MODEL_SEEDS,
    ModelSaverPlugin,
    generate_model_save_name,
    seed_everything,
    verify_model_save_weights,
)

from continualTrain.api.sweep import SweepBase
from continualTrain.api.utils import OPTIONAL_SWEEP_KEYS, read_toml_config

"""
This script imports the given model file and executes training 
in accordance with the configuration file 
"""

# This is bad practice, but it must be done,
# because of how WandB handles its WandB agents.
pm, args = None, None


def main():
    global pm
    global args

    # Pytorch settings
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


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

    # If sweeping, offload execution to `wandb.agent`
    if args.sweep_config_path:
        # Override certain training arguments
        args.use_wandb = True
        args.save_frequency = 0
        args.train_experiences = 2
        args.eval_experiences = 2
        args.enable_ffcv = False
        args.profile = False

        # Load sweep config
        sweep_config = read_toml_config(
            file_path=args.sweep_config_path, sweep=True
        )
        # with open(args.sweep_config_path, "r") as file:
        #     sweep_config = tomlkit.load(file)

        # Grab metadata for project name
        metadata = pm.hook.get_metadata()

        # Clean up WandB sweep schema violations
        count = sweep_config["count"]
        sweep_config.pop("count", None)
        sweep_config.pop("program", None)

        # Define a sweep ID
        sweep_id = wandb.sweep(
            sweep=sweep_config,
            project=metadata["wandb_project_name"],
            entity=metadata["wandb_entity"],
        )

        # Launch an agent that handles training launch
        wandb.agent(
            sweep_id=sweep_id,
            function=train,
            count=count,
        )
    else:
        # Standard train
        train()


def train():
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
    wandb_entity = metadata["wandb_entity"]
    wandb_project_name = metadata["wandb_project_name"]
    workers = len(os.sched_getaffinity(0))

    # Get a GPU
    try:
        deviceID = GPUtil.getFirstAvailable(
            order="memory",
            maxLoad=100,
            maxMemory=0.85,
            attempts=5,
            interval=5,
            verbose=False,
            excludeID=args.exclude_gpus,
        )
    except RuntimeError as e:
        sys.exit(f"Error obtaining GPU: {e}")

    device = pm.hook.get_device(available_id=deviceID)

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
    ds_root = "/datasets"
    benchmark = pm.hook.get_dataset(root_path=ds_root, seed=DS_SEED)

    # custom_decoder_pipeline = {
    #     "field_0": [
    #         ffcv.fields.rgb_image.SimpleRGBImageDecoder(),
    #         RandomHorizontalFlipSeeded(0.5),
    #     ],
    #     "field_1": [
    #         ffcv.fields.basics.IntDecoder(),
    #         ffcv.transforms.ToTensor(),
    #     ],
    #     "field_2": [
    #         ffcv.fields.ndarray.NDArrayDecoder(),
    #         RandomHorizontalFlipSeeded(0.5),
    #         ffcv.transforms.ToTensor(),
    #     ],
    #     "field_3": [
    #         ffcv.fields.basics.IntDecoder(),
    #         ffcv.transforms.ToTensor(),
    #     ],
    # }
    custom_decoder_pipeline = pm.hook.get_ffcv_decoder_pipeline()

    # Super sonic (in theory)
    if args.enable_ffcv:
        enable_ffcv(
            benchmark,
            f"{ds_root}/ffcv",
            device,
            ffcv_parameters=dict(
                num_workers=workers,
                write_mode="proportion",
                compress_probability=0.50,
                max_resolution=256,
                jpeg_quality=90,
                os_cache=True,
                seed=DS_SEED,
            ),
            decoder_def=custom_decoder_pipeline,
            decoder_includes_transformations=False,
            force_overwrite=False,
            print_summary=False,
        )

    # Grab train and test streams from benchmark
    train_stream = (
        benchmark.train_stream[: int(args.train_experiences)]
        if hasattr(args, "train_experiences")
        else benchmark.train_stream
    )
    test_stream = (
        benchmark.test_stream[: int(args.eval_experiences)]
        if hasattr(args, "eval_experiences")
        else benchmark.test_stream
    )

    # Grab validation stream
    val_stream = None  # Defaults to none
    if hasattr(benchmark, "val_stream"):
        val_stream = (
            benchmark.val_stream[: int(args.train_experiences)]
            if hasattr(args, "train_experiences")
            else benchmark.val_stream
        )
    else:
        print("No validation set!")

    # Collate function
    collate_fn = pm.hook.get_collate()

    # Load profiler
    if args.profile:
        # Initialize the profiler
        profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            with_stack=True,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                "/workspace/log_dir"
            ),
        )

        # Start the profiler
        profiler.start()

    # Don't run multiple seeds, just pick the first for a sweep
    global MODEL_SEEDS
    if args.sweep_config_path:
        MODEL_SEEDS = MODEL_SEEDS[:1]

    # TRAINING
    for model_seed in MODEL_SEEDS:
        # Seed everything
        seed_everything(seed=model_seed)

        # Set up interactive logging
        loggers = [interactive_logger]

        # Set up evaluator, which accepts loggers
        eval_plugin = pm.hook.get_evaluator(loggers=loggers)


        # Model from plugin manager
        model = pm.hook.get_model(device=device, seed=model_seed)

        # Set up Avalanche strategy plugins
        plugins = []

        if args.save_frequency > 0:
            model_saver_plugin = ModelSaverPlugin(
                save_frequency=args.save_frequency,
                strategy_name=strategy_name,
                rand_uuid=rand_uuid,
                save_path=args.save_path,
            )
            plugins.append(model_saver_plugin)

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

        ########################################################################
        ############################ WandB Setup ###############################

        # Set up wandb logging
        if args.use_wandb:
            run_name = (
                f"{rand_uuid}_seed{model_seed}_{strategy_name}_{dataset_name}"
            )
            wandb_params = {"entity": wandb_entity, "name": run_name}

            # Extract and log common hyperparameters
            wandb_config_dict = {
                "model_seed": model_seed,
                "gpu_ID": device,
                "model_class": type(model).__name__,
                "optimizer_type": type(optimizer).__name__,
                "experiences": int(benchmark.n_experiences),
                "scheduler_type": (
                    type(scheduler.scheduler).__name__ if scheduler else "None"
                ),
            }

            # Add metadata to config dict
            wandb_config_dict.update(metadata)

            # Initialize WandB through Avalanche
            wandb_logger = WandBLogger(
                project_name=wandb_project_name,
                params=wandb_params,
                config=wandb_config_dict,
            )
            cl_strategy.evaluator.loggers.append(wandb_logger)

            # If sweeping, sweep must hijack certain parameters before proceeding with training
            if args.sweep_config_path:
                hijackables = wandb.config.keys() - wandb_config_dict.keys()

                # Sweep params are already in wandb.config
                # Explicitly mentioning the common ones
                if "learning_rate" in hijackables:
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = wandb.config["learning_rate"]
                if "epochs" in hijackables:
                    cl_strategy.train_epochs = wandb.config["epochs"]
                if "batch_size" in hijackables:
                    cl_strategy.train_mb_size = wandb.config["batch_size"]

                # Automating the less common ones
                for hj in hijackables:
                    if isinstance(cl_strategy, SweepBase):
                        if hj == "si_lambda":
                             param = [wandb.config[hj]]
                        else:
                            param = wandb.config[hj]
                        cl_strategy.set_plugin_attribute(hj, param)

            # If not sweeping, update WandB config for logging
            else:
                wandb.config["learning_rate"] = optimizer.param_groups[0]["lr"]
                wandb.config["batch_size"] = cl_strategy.train_mb_size
                wandb.config["epochs"] = cl_strategy.train_epochs

        ########################################################################
        ########################################################################

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
            if args.save_frequency > 0:
                model.save_weights(save_name)
        else:
            for exp_id, experience in enumerate(train_stream):
                # Get val exp, if exists
                if val_stream is not None:
                    val_exp = val_stream[exp_id]

                # Invoke strategy train method
                print("Start of experience: ", experience.current_experience)
                print(
                    "Current Classes: ",
                    experience.classes_in_this_experience,
                )

                cl_strategy.train(
                    experience,
                    eval_streams=(
                        [val_exp]
                        if val_stream is not None
                        else [test_stream[exp_id]]
                    ),
                    num_workers=workers,
                    persistent_workers=True,
                    collate_fn=collate_fn,
                    ffcv_args={
                        "print_ffcv_summary": False,
                        "batches_ahead": 2,
                    },
                )
                print("Training completed")
                # LR Scheduler will reset here

                # Invoke strategy evaluation method
                print("Evaluating experiences")
                results.append(
                    cl_strategy.eval(
                        test_stream,
                        num_workers=workers,
                        persistent_workers=True,
                        ffcv_args={
                            "print_ffcv_summary": False,
                            "batches_ahead": 2,
                        },
                    )
                )

                # Save model
                save_name = generate_model_save_name(
                    save_path=args.save_path,
                    strategy=strategy_name,
                    rand_uuid=rand_uuid,
                    experience=exp_id,
                    epoch=cl_strategy.train_epochs,
                )
                if args.save_frequency > 0:
                    model.save_weights(save_name)

        if args.use_wandb:
            wandb.finish()

    if args.profile:
        profiler.stop()


if __name__ == "__main__":
    main()
