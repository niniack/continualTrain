import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Train a model.")

    # Grabs arguments from API
    parser.add_argument(
        "hook_implementation",
        type=str,
        help="Path with your hook implementations to drive the training script.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="Path to save models",
    )
    parser.add_argument(
        "--save_frequency",
        type=int,
        required=True,
        help="Frequency to save models. Anything below 0 is invalid.",
    )
    parser.add_argument(
        "--sweep_config_path",
        type=str,
        help="Path to sweep config",
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Uses WandB logging",
    )
    parser.add_argument(
        "--train_experiences",
        type=int,
        help="Number of experiences to train on",
    )
    parser.add_argument(
        "--eval_experiences",
        type=int,
        help="Number of experiences to evaluate on",
    )
    parser.add_argument(
        "--enable_ffcv",
        action="store_true",
        help="Use FFCV for dataloading",
    )
    parser.add_argument(
        "--exclude_gpus",
        nargs="*",
        type=int,
        help="The Device IDs of GPUs to exclude",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Profile the main training loop.",
    )
    args = parser.parse_args()
    return args
