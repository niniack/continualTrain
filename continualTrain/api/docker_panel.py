import subprocess
from pathlib import Path

from continualTrain.continualTrain.api.launch import ContainerTool, train
from continualTrain.continualTrain.api.utils import validate_configs


def docker_run_sweep(
    project_path: str,
    training_config: dict,
    sweep_config: dict,
    image_name: str,
    run_interactive: bool,
) -> None:
    # Validate paths in config, pass in sweep
    paths = validate_configs(
        project_path=project_path,
        training_config=training_config,
        sweep_config=sweep_config,
    )

    # Launch sweep
    train(
        training_config=training_config,
        paths=paths,
        image_name=image_name,
        container_type=ContainerTool.docker,
        run_interactive=run_interactive,
        run_profiler=False,
        run_debug=False,
        sweep_config=sweep_config,
    )


def docker_run_training(
    project_path: str,
    training_config: dict,
    image_name: str,
    run_interactive: bool,
    run_profiler: bool,
    run_debug: bool,
) -> None:
    # Validate paths in config
    paths = validate_configs(
        project_path=project_path,
        training_config=training_config,
        sweep_config=None,
    )

    # Launch train
    train(
        training_config=training_config,
        paths=paths,
        image_name=image_name,
        container_type=ContainerTool.docker,
        run_interactive=run_interactive,
        run_profiler=run_profiler,
        run_debug=run_debug,
        sweep_config=None,
    )


def docker_build_image(
    add_deps: list, image_name: str, push: bool, local_registry: str = None
) -> None:
    non_git_deps = []
    git_deps = []
    for dep in add_deps:
        if dep.startswith("git+"):
            git_deps.append(dep)
        else:
            non_git_deps.append(dep)

    non_git_deps_str = " ".join(non_git_deps)
    git_deps_str = " ".join(git_deps)

    script_dir = Path(__file__).resolve().parent
    parent_dir = script_dir.parent
    docker_dir = parent_dir / "docker"

    # Not shown here, but you can pass in a CACHEBUSTER build arg
    # This will bust the cache for a github repo
    command = [
        "docker",
        "build",
        "-t",
        image_name,
        "--build-arg",
        f"ADDITIONAL_DEPS={non_git_deps_str}",
        "--build-arg",
        f"GIT_DEPS={git_deps_str}",
        "-f",
        str(docker_dir / "Dockerfile"),
        str(parent_dir.parent),
    ]
    subprocess.run(command, check=True)

    if push:
        if local_registry:
            tagged_image = f"{local_registry}/{image_name}"
            subprocess.run(
                ["docker", "tag", image_name, tagged_image], check=True
            )
            subprocess.run(["docker", "push", tagged_image], check=True)
        else:
            # Push the image to its original registry
            subprocess.run(["docker", "push", image_name], check=True)
