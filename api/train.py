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
        "--config", type=str, help="Path to JSON model config file", required=True
    )
    parser.add_argument(
        "--model_file",
        type=str,
        help="Path to Python file defining model",
        required=True,
    )
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="Path to save models",
    )
    args = parser.parse_args()
    return args


def build_docker_image():
    script_dir = os.path.dirname(os.path.realpath(__file__))
    parent_dir = os.path.dirname(script_dir)
    docker_dir = os.path.join(parent_dir, "docker")
    command = ["docker", "build", "-t", "continual_train", docker_dir]
    subprocess.run(command, check=True)


# fmt: off
def docker_run_training(args):
    # Resolve paths to absolute paths
    model_path = Path(args.model_file).resolve().parent
    model_file = Path(args.model_file).resolve().name

    config_path = Path(args.config).resolve().parent
    config_file = Path(args.config).resolve().name

    save_path = Path(args.save_path).resolve()
    
    project_directory = Path(__file__).resolve().parent.parent

    command = [
        "docker", "run", "-it", "--rm",
        "--gpus", "all",
        "--ipc=host",
        "--ulimit", "memlock=-1",
        "--ulimit", "stack=67108864",
        "--env-file", f"{project_directory}/.env",
        "-e", f"PYTHONPATH=/model",
        "-v", f"{project_directory}:/workspace",
        "-v", f"{os.getenv('HOME')}/.ssh:/root/.ssh",
        "-v", f"{model_path}:/model",
        "-v", f"{config_path}:/config",
        "-v", f"{save_path}:/save",
        "-v", "/mnt:/mnt",
        # "-v", f"{os.getenv('HOME')}/.avalanche:/root/.avalanche",
        # "-v", "/home/alodie/datasets:/root/datasets",
        image_name, "/bin/bash", "-c",
        f"cd /workspace && pip install -e . && \
          python /workspace/scripts/run_training.py \
        --config_file /config/{config_file} \
        --model_file /model/{model_file} \
        --save_path /save"
    ]
    
    subprocess.run(command, check=True)
# fmt: on


def main():
    args = parse_args()
    build_docker_image()
    docker_run_training(args)


if __name__ == "__main__":
    main()
