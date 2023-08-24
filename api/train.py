import argparse
import subprocess
import os
from pathlib import Path
from .utils import yaml_file, read_yaml_config, check_path_exists

image_name = "continual_train"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a CL model in a Docker container."
    )
    parser.add_argument(
        "training_config",
        type=yaml_file,
        help="Path to a YAML configuration.",
    )
    args = parser.parse_args()
    return args


def build_docker_image():
    script_dir = Path(__file__).resolve().parent
    parent_dir = script_dir.parent
    docker_dir = parent_dir / "docker"
    command = ["docker", "build", "-t", "continual_train", str(docker_dir)]
    subprocess.run(command, check=True)


# fmt: off
def docker_run_training(args):
    config = read_yaml_config(args.training_config)
    
    save_path = check_path_exists(config['save_path'], 'save_path')
    dataset_path = check_path_exists(config['dataset_path'], 'dataset_path')
    training_dir_path = check_path_exists(config['training_dir'], 'training_dir')
    hook_impl_files = list(training_dir_path.glob('*.py'))

    this_dir = Path(__file__).resolve().parent.parent

    processes = []  # List to keep track of all started processes

    for impl in hook_impl_files:
        command = [
            "docker", "run", "-it", "--rm",
            "--gpus", "all",
            "--ipc=host",
            "--ulimit", "memlock=-1",
            "--ulimit", "stack=67108864",
            "--env-file", f"{this_dir}/.env",
            # Mount this directory
            "-v", f"{this_dir}:/workspace",
            "-v", f"{training_dir_path}:/training_dir",
            "-v", f"{os.getenv('HOME')}/.ssh:/root/.ssh",
            "-v", f"{save_path}:/save",
            "-v", f"{dataset_path}:/datasets",  # Mount the provided dataset directory to /datasets in the container
            image_name, "/bin/bash", "-c",
            f"cd /workspace && pip install -e . && \
            python /workspace/scripts/run_training.py \
            /training_dir/{impl.name} \
            --save_path /save"
        ]

        if config.get('use_wandb', False):
            command.extend(["--use_wandb"])
        
        process = subprocess.Popen(command)
        processes.append(process)
    
    # Wait for all processes to complete
    for process in processes:
        process.wait()
# fmt: on


def main():
    args = parse_args()
    build_docker_image()
    docker_run_training(args)


if __name__ == "__main__":
    main()
