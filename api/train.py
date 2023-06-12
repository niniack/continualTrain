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

    args = parser.parse_args()
    return args


def build_docker_image():
    command = ["docker", "build", "-t", image_name, "./docker"]
    subprocess.run(command, check=True)


# fmt: off
def docker_run_training(args):

    model_path = Path(args.model_file).parent
    model_file = Path(args.model_file).name

    config_path = Path(args.config).parent
    config_file = Path(args.config).name

    print(config_path)

    command = [
        "docker", "run", "-it", "--rm",
        "--gpus", "all",
        "--ipc=host",
        "--ulimit", "memlock=-1",
        "--ulimit", "stack=67108864",
        "-v", f"{os.getcwd()}:/workspace",
        "-v", f"{os.getenv('HOME')}/.ssh:/root/.ssh",
        "-v", f"{model_path}:/model",
        "-v", f"{config_path}:/config",
        "-v", "/mnt:/mnt",
        # "-v", f"{os.getenv('HOME')}/.avalanche:/root/.avalanche",
        # "-v", "/home/alodie/datasets:/root/datasets",
        image_name, "/bin/bash", "-c",
        f"cd /workspace && pip install -e . && \
          python /workspace/scripts/run_training.py \
        --config_file {config_file} \
        --model_file {model_file}"
    ]
    
    subprocess.run(command, check=True)
# fmt: on


def main():
    args = parse_args()
    build_docker_image()
    docker_run_training(args)


if __name__ == "__main__":
    main()
