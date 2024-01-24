import argparse
from enum import Enum
from pathlib import Path

import requests
import toml
from rich import print


class ContainerTool(str, Enum):
    docker = "docker"
    singularity = "singularity"


REQUIRED_KEYS = [
    "save_path",
    "dataset_path",
    "training_dir",
    "enable_wandb_logging",
    "wandb_api_key",
]

OPTIONAL_KEYS = [
    "dependencies_list",
    "train_experiences",
    "eval_experiences",
    "exclude_gpus_list",
    "enable_cuda_debug",
    "overlays_list",
    "enable_ffcv",
]


def get_latest_commit_sha(repo_url, branch="master"):
    # Extract repo details from the URL
    user, repo = repo_url.split("/")[-2:]

    # GitHub API endpoint to get the latest commit SHA for the specified branch
    api_url = f"https://api.github.com/repos/{user}/{repo}/commits/{branch}"

    response = requests.get(api_url)
    response.raise_for_status()  # Ensure we got a successful response

    return response.json()["sha"]


def toml_file(file_path):
    if not file_path.endswith(".toml"):
        raise argparse.ArgumentTypeError(
            f"The file '{file_path}' is not a valid TOML file."
        )
    return file_path


def read_toml_config(file_path: Path) -> dict:
    with open(file_path, "r") as file:
        config = toml.load(file)

    # Check for missing keys
    missing_keys = [key for key in REQUIRED_KEYS if key not in config]
    if missing_keys:
        raise ValueError(
            "Missing required keys in the TOML configuration: {', '.join(missing_keys)}"
        )

    # Verify that 'exclude_gpus_list' is a list of integers formatted as "[x,y,z]"
    if "exclude_gpus_list" in config:
        if not isinstance(config["exclude_gpus_list"], list):
            raise ValueError(
                "'exclude_gpus_list' must be a list in the TOML configuration."
            )

        for gpu in config["exclude_gpus_list"]:
            if not isinstance(gpu, int):
                raise ValueError(
                    f"Invalid value in 'exclude_gpus_list'. Expected all values to be integers, but found {gpu} of type {type(gpu).__name__}."
                )

    return config


def check_path_exists(path, name):
    resolved_path = Path(path).resolve()
    if not resolved_path.exists():
        raise ValueError(
            f"The provided {name} '{resolved_path}' does not exist!"
        )
    return resolved_path
