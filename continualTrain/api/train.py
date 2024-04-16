import os
import subprocess
from enum import Enum
from pathlib import Path

from continualTrain.api.utils import check_path_exists


class ContainerType(Enum):
    docker = 1
    singularity = 2


def train(
    config, image_name, container_type, run_interactive, run_profiler, run_debug
):
    # Validate paths in config
    save_path = check_path_exists(config["save_path"], "save_path")
    dataset_path = check_path_exists(config["dataset_path"], "dataset_path")
    train_dir_path = check_path_exists(config["training_dir"], "training_dir")

    # Grab singularity overlays
    if "overlays_list" in config:
        overlays_list = config["overlays_list"]
    else:
        overlays_list = None

    hook_impl_files = []
    if "valid_subdirs" in config:
        for subdir_name in config["valid_subdirs"]:
            subdir_path = train_dir_path / subdir_name
            hook_impl_files.extend(subdir_path.rglob("hook*.py"))
    else:
        hook_impl_files = list(train_dir_path.rglob("hook*.py"))

    # Verify hook files
    if len(hook_impl_files) == 0:
        raise FileNotFoundError(
            """
            No implementation files were found in the directory.
            Make sure all files start with `hook_`
            """
        )

    # Grab current directory
    this_dir = Path(__file__).resolve().parent.parent

    # List to keep track of all started processes
    processes = []

    # Start the training proc for each hook implementation
    for impl in hook_impl_files:
        # Start with the base command string:
        cmd_str = (
            f"PYTHONPATH=$PYTHONPATH:/training_dir &&"
            f"/app/.venv/bin/python /workspace/scripts/run_training.py "
            f"/training_dir/{impl.relative_to(*impl.parts[:impl.parts.index('training')+1])} "
            f"--save_path /save"
        )

        save_freq = config.get("save_frequency", 10)
        cmd_str += f" --save_frequency {save_freq}"

        # Add optional arguments to the command string:
        if config.get("enable_wandb_logging", False):
            cmd_str += " --use_wandb"

        if config.get("enable_ffcv", False):
            cmd_str += " --enable_ffcv"

        if "train_experiences" in config:
            cmd_str += f" --train_experiences {config['train_experiences']}"

        if "eval_experiences" in config:
            cmd_str += f" --eval_experiences {config['eval_experiences']}"

        if "exclude_gpus_list" in config:
            gpus_str = " ".join(map(str, config["exclude_gpus_list"]))
            cmd_str += f" --exclude_gpus {gpus_str}"

        if run_profiler:
            cmd_str += " --profile"

        # Build environment for DOCKER
        if container_type == ContainerType.docker:
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
                f"{train_dir_path}:/training_dir",
                "-v",
                f"{os.getenv('HOME')}/.ssh:/root/.ssh",
                "-v",
                f"{save_path}:/save",
                "-v",
                f"{dataset_path}:/datasets",
            ]

            if "enable_cuda_debug" in config and config["enable_cuda_debug"]:
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

        # Build environment for SINGULARITY
        elif container_type == ContainerType.singularity:
            # Singularity environment variables and bind paths
            singularity_environment = f"WANDB_API_KEY={config['wandb_api_key']},WANDB_DISABLE_GIT=True"
            bind_paths = f"{this_dir}:/workspace,{train_dir_path}:/training_dir,{os.getenv('HOME')}/.ssh:/root/.ssh,{save_path}:/save,{dataset_path}:/datasets"

            # Optionally add CUDA_LAUNCH_BLOCKING if it's set in the config
            if "enable_cuda_debug" in config and config["enable_cuda_debug"]:
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

        process = subprocess.Popen(command)
        processes.append(process)

    # Wait for all processes to complete
    for process in processes:
        process.wait()
