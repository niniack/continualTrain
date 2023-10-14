import argparse
import os
import subprocess
from pathlib import Path

from .utils import (
    OPTIONAL_KEYS,
    REQUIRED_KEYS,
    check_path_exists,
    read_toml_config,
    toml_file,
)

image_name = "continual_train"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a CL model in a Docker container.",
        formatter_class=argparse.RawTextHelpFormatter,  # Use the RawTextHelpFormatter
    )

    required_text = "\n".join(f"  - {key}" for key in REQUIRED_KEYS)
    optional_text = "\n".join(f"  - {key}" for key in OPTIONAL_KEYS)

    help_text = (
        "Path to a TOML configuration. The TOML file must be configured with the following keys:\n"
        "Required:\n"
        f"{required_text}\n"
        "\nOptional:\n"
        f"{optional_text}"
    )

    parser.add_argument(
        "training_config",
        type=toml_file,
        help=help_text,
    )
    args = parser.parse_args()
    return args


def build_docker_image(config):
    dependencies_str = " ".join(config["dependencies"])
    script_dir = Path(__file__).resolve().parent
    parent_dir = script_dir.parent
    docker_dir = parent_dir / "docker"
    command = [
        "docker",
        "build",
        "-t",
        image_name,
        "--build-arg",
        f"ADDITIONAL_DEPS={dependencies_str}",
        "-f",
        str(docker_dir / "Dockerfile"),
        str(parent_dir.parent),
    ]
    subprocess.run(command, check=True)


# fmt: off
def docker_run_training(config):
    
    save_path = check_path_exists(config['save_path'], 'save_path')
    dataset_path = check_path_exists(config['dataset_path'], 'dataset_path')
    training_dir_path = check_path_exists(config['training_dir'], 'training_dir')
    hook_impl_files = list(training_dir_path.glob('*.py'))

    this_dir = Path(__file__).resolve().parent.parent

    processes = []  # List to keep track of all started processes

    # Set up the environment for the docker command
    docker_environment = [ 
        "--gpus", "all",
        "--ipc=host",
        "--ulimit", "memlock=-1",
        "--ulimit", "stack=67108864",
        "-e", f"WANDB_API_KEY={config['wandb_api_key']}",
        "-e", "WANDB_DISABLE_GIT",
        "-v", f"{this_dir}:/workspace",
        "-v", f"{training_dir_path}:/training_dir",
        "-v", f"{os.getenv('HOME')}/.ssh:/root/.ssh",
        "-v", f"{save_path}:/save",
        "-v", f"{dataset_path}:/datasets",  
    ]

    # Now, start the training processes for each hook implementation
    for impl in hook_impl_files:
        # Start with the base command string:
        cmd_str = (
            f"cd /workspace && "
            f"poetry run python /workspace/scripts/run_training.py "
            f"/training_dir/{impl.name} "
            f"--save_path /save"
        )

        # Add optional arguments to the command string:
        if config.get('wandb_enable_logging', True):
            cmd_str += " --use_wandb"

        if 'train_experiences' in config:
            cmd_str += f" --train_experiences {config['train_experiences']}"

        if 'eval_experiences' in config:
            cmd_str += f" --eval_experiences {config['eval_experiences']}"

        # Construct the full Docker command:
        command = [
            "docker", "run", "-it", "--rm",
            *docker_environment,
            image_name, "/bin/bash", "-c",
            cmd_str   # Add the constructed command string here
        ]
        
        process = subprocess.Popen(command)
        processes.append(process)
    
    # Wait for all processes to complete
    for process in processes:
        process.wait()
# fmt: on


def main():
    args = parse_args()
    config = read_toml_config(args.training_config)
    build_docker_image(config)
    docker_run_training(config)


if __name__ == "__main__":
    main()
