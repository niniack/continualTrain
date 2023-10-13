import argparse
import os
import subprocess
from pathlib import Path

from .utils import check_path_exists, read_toml_config, toml_file

image_name = "continual_train"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a CL model in a Docker container."
    )
    parser.add_argument(
        "training_config",
        type=toml_file,
        help="Path to a YAML configuration.",
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
        "-e", f"{config['wandb_api_key']}",
        "-e", "WANDB_DISABLE_GIT",
        "-v", f"{this_dir}:/workspace",
        "-v", f"{training_dir_path}:/training_dir",
        "-v", f"{os.getenv('HOME')}/.ssh:/root/.ssh",
        "-v", f"{save_path}:/save",
        "-v", f"{dataset_path}:/datasets",  
    ]

    # Now, start the training processes for each hook implementation
    for impl in hook_impl_files:
        command = [
            "docker", "run", "-it", "--rm",
            *docker_environment,
            image_name, "/bin/bash", "-c",
            f"cd /workspace && \
            poetry run python /workspace/scripts/run_training.py \
            /training_dir/{impl.name} \
            --save_path /save"
        ]

        if config.get('wandb_enable_logging', True):
            command.extend(["--use_wandb"])
        
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
