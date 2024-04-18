import os
import subprocess

import requests
from rich import print

from continualTrain.api.launch import ContainerTool, train


def singularity_run_training(
    config, image_name, run_interactive, run_profiler, run_debug
):
    train(
        config,
        image_name,
        ContainerTool.singularity,
        run_interactive,
        run_profiler,
        run_debug,
    )


def singularity_pull_image(image_name, local_registry: str = None):
    save_dir = os.path.join(os.environ["SCRATCH"], ".singularity")
    os.makedirs(save_dir, exist_ok=True)

    # # Check if image needs to be updated
    # if not is_image_update_required(image_name, save_dir):
    #     print(f"No update needed for {image_name}.")
    #     return

    # Extract the image name without any tags for the file name
    image_file_name = image_name.split("/")[-1].split(":")[0] + ".sif"
    save_path = os.path.join(save_dir, image_file_name)

    if local_registry:
        image_path = f"docker://{local_registry}/{image_name}"
    else:
        image_path = f"docker://{image_name}"

    command = f"""
    export SINGULARITY_CACHEDIR=$SCRATCH &&
    export SINGULARITY_TMPDIR=$TMPDIR &&
    export SINGULARITY_NOHTTPS=1 &&
    singularity pull --force --name {save_path} {image_path}
    """
    try:
        subprocess.run(
            command,
            check=True,
            shell=True,
            stdout=None,
            stderr=None,
        )
        print(
            f"Successfully pulled and converted {image_name}. It is saved at {save_path}"
        )
        # Update the stored digest
        update_digest(image_name, save_dir)
    except subprocess.CalledProcessError as e:
        error_message = (
            e.stderr.decode().strip() if e.stderr else "Unknown error"
        )
        print(f"Error pulling image: {error_message}")


def get_docker_image_digest(image_name):
    registry_url = f"https://registry.hub.docker.com/v2/repositories/{image_name}/tags/latest"
    response = requests.get(registry_url, timeout=10)  # timeout in seconds
    if response.status_code == 200:
        return response.json().get("images")[0].get("digest")
    else:
        raise LookupError("Failed to fetch image digest from DockerHub")


def is_image_update_required(image_name, save_dir):
    current_digest = get_docker_image_digest(image_name)
    digest_file = os.path.join(
        save_dir, f"{image_name.replace('/', '_')}_digest.txt"
    )
    if os.path.exists(digest_file):
        with open(digest_file, "r", encoding="utf-8") as file:
            last_digest = file.read().strip()
            return last_digest != current_digest
    return True  # If no digest file found, assume update is needed


def update_digest(image_name, save_dir):
    current_digest = get_docker_image_digest(image_name)
    digest_file = os.path.join(
        save_dir, f"{image_name.replace('/', '_')}_digest.txt"
    )
    with open(digest_file, "w", encoding="utf-8") as file:
        file.write(current_digest)
