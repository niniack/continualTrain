import os
import subprocess
from pathlib import Path

from rich import print

from continualTrain.api.utils import check_path_exists


def docker_run_training(config, image_name, run_interactive, run_profiler, run_debug):
    save_path = check_path_exists(config["save_path"], "save_path")
    dataset_path = check_path_exists(config["dataset_path"], "dataset_path")
    training_dir_path = check_path_exists(config["training_dir"], "training_dir")
    hook_impl_files = list(training_dir_path.glob("hook*.py"))

    this_dir = Path(__file__).resolve().parent.parent

    processes = []  # List to keep track of all started processes

    # Set up the environment for the docker command
    docker_environment = [
        "--gpus",
        "all",
        "--ipc=host",
        "--ulimit",
        "memlock=-1",
        "--ulimit",
        "stack=67108864",
        "-e",
        f"WANDB_API_KEY={config['wandb_api_key']}",
        "-e",
        "WANDB_DISABLE_GIT",
        "-v",
        f"{this_dir}:/workspace",
        "-v",
        f"{training_dir_path}:/training_dir",
        "-v",
        f"{os.getenv('HOME')}/.ssh:/root/.ssh",
        "-v",
        f"{save_path}:/save",
        "-v",
        f"{dataset_path}:/datasets",
    ]

    # Optionally add CUDA_LAUNCH_BLOCKING if it's set in the config
    if "enable_cuda_debug" in config and config["enable_cuda_debug"]:
        docker_environment.extend(["-e", "CUDA_LAUNCH_BLOCKING=1"])

    # Now, start the training processes for each hook implementation
    for impl in hook_impl_files:
        # Start with the base command string:
        cmd_str = (
            f"PYTHONPATH=$PYTHONPATH:/training_dir "
            f"poetry run python /workspace/scripts/run_training.py "
            f"/training_dir/{impl.name} "
            f"--save_path /save"
        )

        # Add optional arguments to the command string:
        if config.get("enable_wandb_logging", True):
            cmd_str += " --use_wandb"

        if "train_experiences" in config:
            cmd_str += f" --train_experiences {config['train_experiences']}"

        if "eval_experiences" in config:
            cmd_str += f" --eval_experiences {config['eval_experiences']}"

        if "exclude_gpus_list" in config:
            gpus_str = " ".join(map(str, config["exclude_gpus_list"]))
            cmd_str += f" --exclude_gpus {gpus_str}"

        if run_profiler:
            cmd_str += f" --profile"

        # Construct the full Docker command:
        mode = "-it" if (run_interactive or run_debug) else "-d"
        command = [
            "docker",
            "run",
            mode,
            "--rm",
            *docker_environment,
            image_name,
            *(["/bin/bash"] if run_debug else ["/bin/bash", "-c", cmd_str]),
        ]

        process = subprocess.Popen(command)
        processes.append(process)

    # Wait for all processes to complete
    for process in processes:
        process.wait()
