import argparse
from pathlib import Path

import toml

REQUIRED_KEYS = [
    "save_path",
    "dataset_path",
    "training_dir",
    "wandb_enable_logging",
    "wandb_api_key",
]


def toml_file(file_path):
    if not file_path.endswith(".toml"):
        raise argparse.ArgumentTypeError(
            f"The file '{file_path}' is not a valid TOML file."
        )
    return file_path


def read_toml_config(file_path):
    with open(file_path, "r") as file:
        config = toml.load(file)

    # Check for missing keys
    missing_keys = [key for key in REQUIRED_KEYS if key not in config]
    if missing_keys:
        raise ValueError(
            f"Missing required keys in the TOML configuration: {', '.join(missing_keys)}"
        )

    return config


def check_path_exists(path, name):
    resolved_path = Path(path).resolve()
    if not resolved_path.exists():
        raise ValueError(f"The provided {name} '{resolved_path}' does not exist!")
    return resolved_path
