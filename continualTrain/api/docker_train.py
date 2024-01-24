import os
import subprocess
from pathlib import Path

from rich import print

from continualTrain.api.train import ContainerType, train
from continualTrain.api.utils import check_path_exists


def docker_run_training(
    config, image_name, run_interactive, run_profiler, run_debug
):
    train(
        config,
        image_name,
        ContainerType.docker,
        run_interactive,
        run_profiler,
        run_debug,
    )
