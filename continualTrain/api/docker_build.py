import subprocess
from pathlib import Path

from continualTrain.api.utils import get_latest_commit_sha


def build_docker_image(add_deps: str, image_name: str, push: bool) -> None:
    dependencies_str = " ".join(add_deps)
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
        f"CACHEBUSTER=${LATEST_COMMIT}",
        "--build-arg",
        f"ADDITIONAL_DEPS={dependencies_str}",
        "-f",
        str(docker_dir / "Dockerfile"),
        str(parent_dir.parent),
    ]
    subprocess.run(command, check=True)

    if push:
        subprocess.run(["docker", "push", image_name], check=True)
