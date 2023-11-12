import os
import subprocess
from pathlib import Path

from continualTrain.api.utils import check_path_exists


def singularity_pull_image(image_name):
    save_dir = os.path.join(os.environ["HOME"], ".singularity")
    os.makedirs(save_dir, exist_ok=True)

    # Extract the image name without any tags for the file name
    image_file_name = image_name.split("/")[-1].split(":")[0] + ".sif"
    save_path = os.path.join(save_dir, image_file_name)

    command = f"singularity pull --force --name {save_path} docker://{image_name}"
    try:
        subprocess.run(
            command,
            check=True,
            shell=True,
            stdout=None,
            stderr=None,
        )
        print(f"Successfully pulled and converted {image_name}.")
    except subprocess.CalledProcessError as e:
        error_message = e.stderr.decode().strip() if e.stderr else "Unknown error"
        print(f"{error_message}")


def singularity_run_training(
    config, image_name, run_interactive, run_profiler, run_debug
):
    save_path = check_path_exists(config["save_path"], "save_path")
    dataset_path = check_path_exists(config["dataset_path"], "dataset_path")
    training_dir_path = check_path_exists(config["training_dir"], "training_dir")
    hook_impl_files = list(training_dir_path.glob("hook*.py"))

    this_dir = Path(__file__).resolve().parent.parent

    singularity_pull_image(image_name)

    processes = []  # List to keep track of all started processes

    # Singularity environment variables and bind paths
    environment = f"WANDB_API_KEY={config['wandb_api_key']},WANDB_DISABLE_GIT=True"
    bind_paths = f"{this_dir}:/workspace,{training_dir_path}:/training_dir,{os.getenv('HOME')}/.ssh:/root/.ssh,{save_path}:/save,{dataset_path}:/datasets"

    # Optionally add CUDA_LAUNCH_BLOCKING if it's set in the config
    if "cuda_debug" in config and config["cuda_debug"]:
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
        if config.get("wandb_enable_logging", True):
            cmd_str += " --use_wandb"

        if "train_experiences" in config:
            cmd_str += f" --train_experiences {config['train_experiences']}"

        if "eval_experiences" in config:
            cmd_str += f" --eval_experiences {config['eval_experiences']}"

        if "exclude_gpus" in config:
            gpus_str = " ".join(map(str, config["exclude_gpus"]))
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
            f"{Path.home()}/.singularity/{image_name.split('/')[-1]}.sif",
            *shell,
        ]

        print(cmd_str)
        process = subprocess.Popen(command)
        processes.append(process)

    # Wait for all processes to complete
    for process in processes:
        process.wait()
