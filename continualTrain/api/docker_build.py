import subprocess
from pathlib import Path

from rich import print

from continualTrain.api.utils import get_latest_commit_sha


def build_docker_image(
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
    LATEST_COMMIT = get_latest_commit_sha(
        "https://github.com/niniack/continualUtils", "dev"
    )

    command = [
        "docker",
        "build",
        "-t",
        image_name,
        "--build-arg",
        f"CACHEBUSTER={LATEST_COMMIT}",
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
            subprocess.run(["docker", "tag", image_name, tagged_image], check=True)
            subprocess.run(["docker", "push", tagged_image], check=True)
        else:
            # Push the image to its original registry
            subprocess.run(["docker", "push", image_name], check=True)
