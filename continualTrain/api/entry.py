from pathlib import Path

import tomlkit
import typer
from rich import print
from typing_extensions import Annotated

from continualTrain.api.docker_panel import (
    docker_build_image,
    docker_run_sweep,
    docker_run_training,
)
from continualTrain.api.singularity_panel import (
    singularity_pull_image,
    singularity_run_training,
)
from continualTrain.api.utils import (
    OPTIONAL_KEYS,
    OPTIONAL_SWEEP_KEYS,
    REQUIRED_KEYS,
    REQUIRED_SWEEP_KEYS,
    ContainerTool,
    read_toml_config,
)

banner_message = """
`barracks` is a CLI for continualTrain, which is built for
compatibility with containerization (Docker and Singularity) and relies on
WandB for logging. With barracks, you can initialize a project, launch training,
or launch WandB sweeps.
"""

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
    help=banner_message,
)


def _overwrite_protector(file_path: Path) -> bool:
    if file_path.is_file():
        print(
            f"[red]\nThere already exists a {file_path.name} configuration file."
            f" This command will not overwrite an existing configuration."
            " Please delete the file and initialize again.\n"
        )
        return True
    else:
        print(f"[green]\nInitialized a {file_path.name} configuration file!\n")
        return False


@app.command()
def initialize(
    project_path: Annotated[
        Path,
        typer.Argument(
            exists=False,
            file_okay=False,
            dir_okay=True,
            writable=True,
            readable=True,
            resolve_path=True,
            help="Directory where training.toml and sweep.toml configuration "
            "files will be initialized. Will not overwrite files.",
        ),
    ],
):
    """Initialize configuration files with template."""

    train_config_name = "training.toml"
    training_config_path = Path(project_path, train_config_name)

    # Initialize training.toml
    if not _overwrite_protector(training_config_path):
        config_dict = {
            key: False if "enable" in key else [] if "list" in key else ""
            for key in REQUIRED_KEYS + OPTIONAL_KEYS
        }

        config_dict["save_path"] = "./model_saves"
        config_dict["dataset_path"] = "/mnt/datasets"
        config_dict["training_dir"] = "./training"
        config_dict["train_experiences"] = 1
        config_dict["eval_experiences"] = 1
        config_dict["wandb_api_key"] = "YOUR_WANDB_API_KEY"

        with training_config_path.open("w", encoding="utf-8") as cf:
            tomlkit.dump(config_dict, cf)

    sweep_config_name = "sweep.toml"
    sweep_config_path = Path(project_path, sweep_config_name)

    # Initialize sweep.toml
    if not _overwrite_protector(sweep_config_path):
        config_dict = {key: "" for key in REQUIRED_SWEEP_KEYS}

        config_dict["method"] = "random"
        config_dict["program"] = "hook_x_.py"
        config_dict["count"] = 1
        config_dict["parameters"] = {}

        for key in OPTIONAL_SWEEP_KEYS:
            config_dict["parameters"][key] = {"values": []}

        with sweep_config_path.open("w", encoding="utf-8") as cf:
            tomlkit.dump(config_dict, cf)


@app.command()
def build(
    image_name: Annotated[
        str, typer.Argument(help="Name of the Docker image.")
    ],
    config: Annotated[
        Path,
        typer.Option(
            exists=True,
            file_okay=True,
            dir_okay=False,
            writable=False,
            readable=True,
            resolve_path=True,
            help="Path to training.toml configuration file",
        ),
    ],
    push: Annotated[
        bool, typer.Option("--push", help="Push the image to Dockerhub.")
    ] = False,
):
    """Build a Docker image based on project configuration and push it to the hub."""

    parsed_config = read_toml_config(config)
    docker_build_image(
        add_deps=parsed_config["dependencies_list"],
        image_name=image_name,
        push=push,
        local_registry="localhost:5000",
    )


@app.command()
def sweep(
    image_name: Annotated[
        str, typer.Argument(help="Name of the Docker image.")
    ],
    project_path: Annotated[
        Path,
        typer.Option(
            exists=True,
            file_okay=False,
            dir_okay=True,
            writable=False,
            readable=True,
            resolve_path=True,
            help="Project path containing training.toml and sweep.toml. "
            "Defaults to the current directory.",
        ),
    ] = ".",
    tool: Annotated[
        ContainerTool,
        typer.Option(
            "--tool",
            "-t",
            help="Choose the containerization tool to use. Defaults to Docker.",
            case_sensitive=False,
        ),
    ] = ContainerTool.docker,
    interactive: Annotated[
        bool,
        typer.Option(
            "--interactive", "-it", help="Make the session interactive."
        ),
    ] = False,
):
    """
    Pull the latest image from the hub and run a WandB sweep for
    exploring hyperparameters
    """

    # Grab configurations
    sweep_config_path = Path.joinpath(project_path, "sweep.toml")
    training_config_path = Path.joinpath(project_path, "training.toml")

    if (
        sweep_config_path.is_file() is False
        or training_config_path.is_file() is False
    ):
        raise ValueError(
            "The provided project path is missing a training.toml or sweep.toml. "
            "Did you need to initialize?"
        )

    sweep_config = read_toml_config(file_path=sweep_config_path, sweep=True)
    training_config = read_toml_config(
        file_path=training_config_path, sweep=False
    )

    if tool == ContainerTool.docker:
        docker_run_sweep(
            project_path=project_path,
            training_config=training_config,
            sweep_config=sweep_config,
            image_name=image_name,
            run_interactive=interactive,
        )

    elif tool == ContainerTool.singularity:
        raise NotImplementedError("Can not run a sweep on Singularity (yet)!")


@app.command()
def train(
    image_name: Annotated[
        str, typer.Argument(help="Name of the Docker image.")
    ],
    project_path: Annotated[
        Path,
        typer.Option(
            exists=True,
            file_okay=False,
            dir_okay=True,
            writable=False,
            readable=True,
            resolve_path=True,
            help="Project path containing training.toml and sweep.toml. "
            "Defaults to the current directory.",
        ),
    ] = ".",
    tool: Annotated[
        ContainerTool,
        typer.Option(
            "--tool",
            "-t",
            help="Choose the containerization tool to use.",
            case_sensitive=False,
        ),
    ] = ContainerTool.docker,
    interactive: Annotated[
        bool,
        typer.Option(
            "--interactive", "-it", help="Make the session interactive."
        ),
    ] = False,
    profiler: Annotated[
        bool, typer.Option("--profile", "-p", help="Run a Torch profiler.")
    ] = False,
    debug: Annotated[
        bool,
        typer.Option(
            "--debug", "-d", help="Debug the container with interactive mode."
        ),
    ] = False,
):
    """Pull the latest image from the hub and train"""

    # Grab configurations
    training_config_path = Path.joinpath(project_path, "training.toml")

    if training_config_path.is_file() is False:
        raise ValueError(
            "The provided project path is missing a training.toml. "
            "Did you need to initialize?"
        )

    training_config = read_toml_config(
        file_path=training_config_path, sweep=False
    )

    if tool == ContainerTool.docker:
        docker_run_training(
            project_path=project_path,
            training_config=training_config,
            image_name=image_name,
            run_interactive=interactive,
            run_profiler=profiler,
            run_debug=debug,
        )
    elif tool == ContainerTool.singularity:
        singularity_run_training(
            training_config, image_name, interactive, profiler, debug
        )


@app.command()
def singpull(
    image_name: Annotated[str, typer.Argument(help="Name of the image.")],
    local_registry: Annotated[
        str, typer.Option(help="URL of the local registry")
    ] = "10.224.35.137:5000",
):
    """Pull a Singularity image based on an existing Docker image."""

    singularity_pull_image(image_name, local_registry)


if __name__ == "__main__":
    app()
