import os
import subprocess
from pathlib import Path

import requests
from rich import print

from continualTrain.api.utils import check_path_exists


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
        error_message = e.stderr.decode().strip() if e.stderr else "Unknown error"
        print(f"Error pulling image: {error_message}")


def get_docker_image_digest(image_name):
    registry_url = (
        f"https://registry.hub.docker.com/v2/repositories/{image_name}/tags/latest"
    )
    response = requests.get(registry_url)
    if response.status_code == 200:
        return response.json().get("images")[0].get("digest")
    else:
        raise Exception("Failed to fetch image digest from DockerHub")


def is_image_update_required(image_name, save_dir):
    current_digest = get_docker_image_digest(image_name)
    digest_file = os.path.join(save_dir, f"{image_name.replace('/', '_')}_digest.txt")
    if os.path.exists(digest_file):
        with open(digest_file, "r") as file:
            last_digest = file.read().strip()
            return last_digest != current_digest
    return True  # If no digest file found, assume update is needed


def update_digest(image_name, save_dir):
    current_digest = get_docker_image_digest(image_name)
    digest_file = os.path.join(save_dir, f"{image_name.replace('/', '_')}_digest.txt")
    with open(digest_file, "w") as file:
        file.write(current_digest)


def singularity_run_training(
    config, image_name, run_interactive, run_profiler, run_debug
):
    save_path = check_path_exists(config["save_path"], "save_path")
    dataset_path = check_path_exists(config["dataset_path"], "dataset_path")
    training_dir_path = check_path_exists(config["training_dir"], "training_dir")
    if "overlays_list" in config:
        overlays_list = config["overlays_list"]
    else:
        overlays_list = None
    hook_impl_files = list(training_dir_path.glob("hook*.py"))

    this_dir = Path(__file__).resolve().parent.parent

    processes = []  # List to keep track of all started processes

    # Singularity environment variables and bind paths
    environment = f"WANDB_API_KEY={config['wandb_api_key']},WANDB_DISABLE_GIT=True"
    bind_paths = f"{this_dir}:/workspace,{training_dir_path}:/training_dir,{os.getenv('HOME')}/.ssh:/root/.ssh,{save_path}:/save,{dataset_path}:/datasets"

    # Optionally add CUDA_LAUNCH_BLOCKING if it's set in the config
    if "enable_cuda_debug" in config and config["enable_cuda_debug"]:
        environment += ",CUDA_LAUNCH_BLOCKING=1"

    # Now, start the training processes for each hook implementation
    for impl in hook_impl_files:
        # Start with the base command string:
        cmd_str = (
            # f"PYTHONPATH=$PYTHONPATH:/training_dir &&"
            f"poetry run python /workspace/scripts/run_training.py "
            f"/training_dir/{impl.name} "
            f"--save_path /save"
        )

        # Add optional arguments to the command string:
        if config.get("enable_wandb_logging", True):
            cmd_str += " --use_wandb"

        if "train_experiences" in config:
            cmd_str += f" --train_experiences {config['train_experiences']}"

        if "eval_experiences" in config:
            cmd_str += f" --eval_experiences {config['eval_experiences']}"

        if "exclude_gpus_list" in config:
            gpus_str = " ".join(map(str, config["exclude_gpus_list"]))
            cmd_str += f" --exclude_gpus {gpus_str}"

        if run_profiler:
            cmd_str += " --profile"

        # Construct the full Singularity command:
        if run_interactive or run_debug:
            flags = ["--nv", "-i"]
        else:
            flags = ["--nv"]

        if run_debug:
            shell = ["/bin/bash"]
        else:
            shell = ["/bin/bash", "-c"]
            shell.append(cmd_str)

        command = [
            "singularity",
            "exec",
            *flags,
            "--env",
            environment,
            "--bind",
            bind_paths,
            *(
                [
                    item
                    for overlay in overlays_list
                    for item in ["--overlay", str(overlay)]
                ]
                if overlays_list is not None
                else []
            ),
            f"{os.environ['SCRATCH']}/.singularity/{image_name.split('/')[-1]}.sif",
            *shell,
        ]

        process = subprocess.Popen(command)
        processes.append(process)

    # Wait for all processes to complete
    for process in processes:
        process.wait()
