import argparse
from enum import Enum
from pathlib import Path
from typing import List, NamedTuple, Optional, Union

import requests
import tomlkit

REQUIRED_KEYS = [
    "save_path",
    "dataset_path",
    "training_dir",
    "enable_wandb_logging",
    "wandb_api_key",
    "save_frequency",
]

OPTIONAL_KEYS = [
    "dependencies_list",
    "train_experiences",
    "eval_experiences",
    "exclude_gpus_list",
    "enable_cuda_debug",
    "overlays_list",
    "enable_ffcv",
    "valid_subdirs",
]

REQUIRED_SWEEP_KEYS = [
    "method",
    "program",
    "parameters",
    "count",
]

OPTIONAL_SWEEP_KEYS = [
    "learning_rate",
    "batch_size",
    "epochs",
]


class ContainerTool(str, Enum):
    docker = "docker"
    singularity = "singularity"


class ContainerPaths(NamedTuple):
    save_path: Path
    dataset_path: Path
    train_dir_path: Path
    sweep_config_path: Path
    hook_impl_files: List[Path]


def get_latest_commit_sha(repo_url, branch="master"):
    # Extract repo details from the URL
    user, repo = repo_url.split("/")[-2:]

    # GitHub API endpoint to get the latest commit SHA for the specified branch
    api_url = f"https://api.github.com/repos/{user}/{repo}/commits/{branch}"

    response = requests.get(api_url)
    response.raise_for_status()  # Ensure we got a successful response

    return response.json()["sha"]


def toml_file(file_path):
    """_summary_

    :param file_path: _description_
    :raises argparse.ArgumentTypeError: _description_
    :return: _description_
    """
    if not file_path.endswith(".toml"):
        raise argparse.ArgumentTypeError(
            f"The file '{file_path}' is not a valid TOML file."
        )
    return file_path


def read_toml_config(file_path: Path, sweep: bool = False) -> dict:
    """_summary_

    :param file_path: _description_
    :param sweep: _description_, defaults to False
    :raises ValueError: _description_
    :raises ValueError: _description_
    :raises ValueError: _description_
    :return: _description_
    """

    # Open file
    with open(file_path, "r") as file:
        config = tomlkit.load(file)

    # Check for missing keys
    if sweep:
        missing_keys = [key for key in REQUIRED_SWEEP_KEYS if key not in config]
    else:
        missing_keys = [key for key in REQUIRED_KEYS if key not in config]

    if missing_keys:
        raise ValueError(
            f"Missing required keys in the TOML configuration: {', '.join(missing_keys)}"
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


def _path_exists(path: Union[Path, str], key: str) -> Path:
    resolved_path = Path(path).resolve()
    if not resolved_path.exists():
        raise ValueError(
            f"The provided {key} '{resolved_path}' does not exist!"
        )
    return resolved_path


def validate_configs(
    project_path: str,
    training_config: dict,
    sweep_config: Optional[dict] = None,
) -> ContainerPaths:
    save_path = _path_exists(training_config["save_path"], "save_path")
    dataset_path = _path_exists(training_config["dataset_path"], "dataset_path")
    train_dir_path = _path_exists(
        training_config["training_dir"], "training_dir"
    )

    hook_impl_files = []
    if sweep_config:
        file_to_sweep = sweep_config["program"]
        hook_impl_files = list(train_dir_path.rglob(str(file_to_sweep)))
    else:
        # Gather the valid hook implementation files
        if "valid_subdirs" in training_config:
            for subdir_name in training_config["valid_subdirs"]:
                subdir_path = train_dir_path / subdir_name
                _ = _path_exists(subdir_path, "valid_subdirs")
                hook_impl_files.extend(subdir_path.rglob("hook*.py"))
        else:
            hook_impl_files = list(train_dir_path.rglob("hook*.py"))

    # Verify hook files exist
    if len(hook_impl_files) == 0:
        raise FileNotFoundError(
            """
            No implementation files were found.
            Make sure all files start with `hook_`.
            If you are running a sweep, please make sure the file to sweep is exact.
            """
        )

    return ContainerPaths(
        save_path=save_path,
        dataset_path=dataset_path,
        train_dir_path=train_dir_path,
        sweep_config_path=Path.joinpath(project_path, "sweep.toml"),
        hook_impl_files=hook_impl_files,
    )
