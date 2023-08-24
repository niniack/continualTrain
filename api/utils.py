import yaml  # Make sure to install PyYAML
import argparse
from pathlib import Path


REQUIRED_KEYS = ["save_path", "dataset_path", "use_wandb", "training_dir"]


def yaml_file(file_path):
    if not (file_path.endswith(".yaml") or file_path.endswith(".yml")):
        raise argparse.ArgumentTypeError(
            f"The file '{file_path}' is not a valid YAML file."
        )
    return file_path


def read_yaml_config(file_path):
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)

    if config is None:
        raise ValueError(
            f"Missing required keys in the YAML configuration: {', '.join(REQUIRED_KEYS)}"
        )

    # Check for missing keys
    missing_keys = [key for key in REQUIRED_KEYS if key not in config]
    if missing_keys:
        raise ValueError(
            f"Missing required keys in the YAML configuration: {', '.join(missing_keys)}"
        )

    return config


def check_path_exists(path, name):
    resolved_path = Path(path).resolve()
    if not resolved_path.exists():
        raise ValueError(f"The provided {name} '{resolved_path}' does not exist!")
    return resolved_path
