from pathlib import Path

import typer
from typing_extensions import Annotated

from continualTrain.api import singularity_train
from continualTrain.api.docker_build import build_docker_image
from continualTrain.api.docker_train import docker_run_training
from continualTrain.api.singularity_train import singularity_run_training
from continualTrain.api.utils import ContainerTool, read_toml_config

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)


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
        ),
    ],
    push: Annotated[
        bool, typer.Option("--push", help="Push the image to Dockerhub.")
    ] = False,
):
    """Build a Docker image based on project configuration and push it to the hub."""

    parsed_config = read_toml_config(config)
    build_docker_image(
        add_deps=parsed_config["dependencies"], image_name=image_name, push=push
    )


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
