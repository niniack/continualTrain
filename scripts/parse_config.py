import json
from pathlib import Path
from scripts.utils import Strategy, Dataset
from argparse import Namespace

# Required keys for training
required_keys = [
    # Dataset
    "dataset_name",
    "dataset_path",
    "dataset_seed",
    # Strategy config
    "epochs",
    "num_experiences",
    "batch_size",
    "learning_rate",
    "strategy",
    # Model config
    "model_seeds",
    "model_classes"
    # Logging config
    "use_wandb",
    "wandb_project",
    "wandb_entity",
    "use_multihead",
    "save_frequency",
]


def parse_config(config_file):
    # Read file
    with open(config_file, "r") as f:
        file_config = json.load(f)

    # Build dict
    config_dict = {key: file_config.get(key) for key in required_keys}

    # Remaining keys
    kwarg_dict = {
        key: val for key, val in file_config.items() if key not in required_keys
    }

    # Verify that all required keys in the config are valid
    for key, value in config_dict.items():
        if value is None:
            raise ValueError(
                f"Missing configuration for {key} in config file: {config_file}"
            )

    # Apply dataset transformation
    try:
        config_dict["dataset_name"] = Dataset(config_dict["dataset_name"])
    except:
        raise ValueError(f"{config_dict['dataset_name']} is not a valid dataset")

    # Apply strategy transformation
    try:
        config_dict["strategy"] = Strategy(config_dict["strategy"])
    except:
        raise ValueError(f"{config_dict['strategy']} is not a valid strategy")

    # Create the config namespace
    config = Namespace(**config_dict)

    # Return namespace and dict with remaining items
    return config, kwarg_dict
