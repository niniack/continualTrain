import argparse
import subprocess
import os
from pathlib import Path

image_name = "continual_train"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a CL model in a Docker container."
    )
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="Path to save models",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/mnt",  # you can keep a default value if you want
        help="Path to datasets directory",
    )
    parser.add_argument(
        "--use_wandb", action="store_true", default=True, help="Uses WandB logging"
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
    save_path = Path(args.save_path).resolve()
    
    if not save_path.exists():
        raise ValueError(f"The provided save_path '{save_path}' does not exist!")
    
    project_directory = Path(__file__).resolve().parent.parent

    command = [
        "docker", "run", "-it", "--rm",
        "--gpus", "all",
        "--ipc=host",
        "--ulimit", "memlock=-1",
        "--ulimit", "stack=67108864",
        "--env-file", f"{project_directory}/.env",
        "-v", f"{project_directory}:/workspace",
        "-v", f"{os.getenv('HOME')}/.ssh:/root/.ssh",
        "-v", f"{save_path}:/save",
       "-v", f"{args.dataset_path}:/datasets",  # Mount the provided dataset directory to /datasets in the container
        image_name, "/bin/bash", "-c",
        f"cd /workspace && pip install -e . && \
          python /workspace/scripts/run_training.py \
        --save_path /save"
    ]

    if args.use_wandb:
        command.extend(["--use_wandb"])

    subprocess.run(command, check=True)
# fmt: on


def main():
    args = parse_args()
    build_docker_image()
    docker_run_training(args)


if __name__ == "__main__":
    main()
