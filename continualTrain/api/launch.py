import os
import subprocess
from pathlib import Path
from typing import Optional

import tomlkit

from continualTrain.api.utils import ContainerPaths, ContainerTool

"""
This is a `launch` wrapper around `scripts/run_training.py`. 
This script prepares all the arguments depending on the container tool and 
handles instantiating multiple processes. 
"""


def train(
    training_config: dict,
    paths: ContainerPaths,
    image_name: str,
    container_type: ContainerTool,
    run_interactive: bool,
    run_profiler: bool,
    run_debug: bool,
    sweep_config: Optional[dict] = None,
):
    # Grab singularity overlays from training config
    if "overlays_list" in training_config:
        overlays_list = training_config["overlays_list"]
    else:
        overlays_list = None

    # Grab current directory
    this_dir = Path(__file__).resolve().parent.parent

    # Init list to keep track of all started processes
    processes = []

    # Start the training proc for each hook implementation
    for impl in paths.hook_impl_files:
        # Start with the base command string:
        cmd_str = (
            f"PYTHONPATH=$PYTHONPATH:/training_dir &&"
            f"/app/.venv/bin/python /workspace/continualTrain/scripts/run_training.py "
            f"/training_dir/{impl.relative_to(*impl.parts[:impl.parts.index('training')+1])}"
        )

        # Set the save path and frequency
        cmd_str += " --save_path /save"

        save_freq = training_config.get("save_frequency", 10)
        cmd_str += f" --save_frequency {save_freq}"

        # Set sweep config path
        if sweep_config:
            cmd_str += " --sweep_config_path /workspace/sweep.toml"

        # Set WandB
        if training_config.get("enable_wandb_logging", False):
            cmd_str += " --use_wandb"

        # Set FFCV
        if training_config.get("enable_ffcv", False):
            cmd_str += " --enable_ffcv"

        # Set experiences to train for
        if "train_experiences" in training_config:
            cmd_str += (
                f" --train_experiences {training_config['train_experiences']}"
            )

        # Set experiences to eval for
        if "eval_experiences" in training_config:
            cmd_str += (
                f" --eval_experiences {training_config['eval_experiences']}"
            )

        # Set GPUs to exclude
        if "exclude_gpus_list" in training_config:
            gpus_str = " ".join(map(str, training_config["exclude_gpus_list"]))
            cmd_str += f" --exclude_gpus {gpus_str}"

        # Set profiling
        if run_profiler:
            cmd_str += " --profile"

        # Build command string for DOCKER
        if container_type == ContainerTool.docker:
            docker_environment = [
                "--gpus",
                "all",
                "--ipc=host",
                "--ulimit",
                "memlock=-1",
                "--ulimit",
                "stack=67108864",
                "-e",
                f"WANDB_API_KEY={training_config['wandb_api_key']}",
                "-e",
                "WANDB_DISABLE_GIT ",
                "--mount",
                f"type=bind,source={this_dir},target=/workspace/continualTrain",
                "--mount",
                f"type=bind,source={paths.sweep_config_path},target=/workspace/sweep.toml,readonly",
                "--mount",
                f"type=bind,source={paths.train_dir_path},target=/training_dir,readonly",
                "--mount",
                f"type=bind,source={os.getenv('HOME')}/.ssh,target=/root/.ssh,readonly",
                "--mount",
                f"type=bind,source={paths.save_path},target=/save",
                "--mount",
                f"type=bind,source={paths.dataset_path},target=/datasets,readonly",
            ]

            if (
                "enable_cuda_debug" in training_config
                and training_config["enable_cuda_debug"]
            ):
                docker_environment.extend(["-e", "CUDA_LAUNCH_BLOCKING=1"])

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

        # Build command string for SINGULARITY
        elif container_type == ContainerTool.singularity:
            # Singularity environment variables and bind paths
            wandb_api_key = f"WANDB_API_KEY={training_config['wandb_api_key']}"
            wandb_disable_git = "WANDB_DISABLE_GIT=True"
            singularity_environment = f"{wandb_api_key},{wandb_disable_git}"
            bind_paths = (
                f"{this_dir}:/workspace/continualTrain,"
                f"{paths.sweep_config_path}:/workspace/sweep.toml,"
                f"{paths.train_dir_path}:/training_dir,"
                f"{os.getenv('HOME')}/.ssh:/root/.ssh,"
                f"{paths.save_path}:/save,"
                f"{paths.dataset_path}:/datasets"
            )

            # Optionally add CUDA_LAUNCH_BLOCKING if it's set in the training_config
            if (
                "enable_cuda_debug" in training_config
                and training_config["enable_cuda_debug"]
            ):
                singularity_environment += ",CUDA_LAUNCH_BLOCKING=1"

            # Construct the full Singularity command:
            if run_interactive or run_debug:
                flags = ["--nv", "-i"]
            else:
                flags = ["--nv"]

            if run_debug:
                shell = ["/bin/bash"]
            else:
                shell = ["/bin/bash", "-c"]
                shell.append(cmd_str)

            command = [
                "singularity",
                "exec",
                *flags,
                "--env",
                singularity_environment,
                "--bind",
                bind_paths,
                *(
                    [
                        item
                        for overlay in overlays_list
                        for item in ["--overlay", str(overlay)]
                    ]
                    if overlays_list is not None
                    else []
                ),
                f"{os.environ['SCRATCH']}/.singularity/{image_name.split('/')[-1]}.sif",
                *shell,
            ]

        # Run command async and append to keep track of it
        process = subprocess.Popen(command)
        processes.append(process)

    # Wait for all processes to complete
    for process in processes:
        process.wait()


__all__ = ["train"]
