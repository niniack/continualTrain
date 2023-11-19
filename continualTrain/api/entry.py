from pathlib import Path

import toml
import typer
from rich import print
from typing_extensions import Annotated

from continualTrain.api import singularity_train
from continualTrain.api.docker_build import build_docker_image
from continualTrain.api.docker_train import docker_run_training
from continualTrain.api.singularity_train import (
    singularity_pull_image,
    singularity_run_training,
)
from continualTrain.api.utils import (
    OPTIONAL_KEYS,
    REQUIRED_KEYS,
    ContainerTool,
    read_toml_config,
)

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)


@app.command()
def initialize(
    config_path: Annotated[
        Path,
        typer.Argument(
            exists=False,
            file_okay=False,
            dir_okay=True,
            writable=True,
            readable=True,
            resolve_path=True,
            help="Directory where training.toml configuration file will be initialized",
        ),
    ]
):
    """Initialize an empty configuration file"""
    config_file = Path(config_path, "training.toml")

    if config_file.is_file():
        print(
            "[bold red]There already exists a training.toml configuration file."
            " Please delete the file and initialize again."
            " This command will not overwrite an existing file.\n"
        )
    else:
        config_dict = {
            key: "false" if "enable" in key else [] if "list" in key else ""
            for key in REQUIRED_KEYS + OPTIONAL_KEYS
        }
        with config_file.open("w", encoding="utf-8") as cf:
            toml.dump(config_dict, cf)


@app.command()
def build(
    image_name: Annotated[str, typer.Argument(help="Name of the Docker image.")],
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
    build_docker_image(
        add_deps=parsed_config["dependencies_list"],
        image_name=image_name,
        push=push,
        local_registry="localhost:5000",
    )


@app.command()
def singpull(
    image_name: Annotated[str, typer.Argument(help="Name of the image.")],
    local_registry: Annotated[
        str, typer.Option(help="URL of the local registry")
    ] = "10.224.35.137:5000",
):
    """Build a Docker image based on project configuration and push it to the hub."""

    singularity_pull_image(image_name, local_registry)


@app.command()
def train(
    image_name: Annotated[str, typer.Argument(help="Name of the Docker image.")],
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
        typer.Option("--interactive", "-it", help="Make the session interactive."),
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

    parsed_config = read_toml_config(config)

    if tool == ContainerTool.docker:
        docker_run_training(parsed_config, image_name, interactive, profiler, debug)
    elif tool == ContainerTool.singularity:
        singularity_run_training(
            parsed_config, image_name, interactive, profiler, debug
        )


if __name__ == "__main__":
    app()
