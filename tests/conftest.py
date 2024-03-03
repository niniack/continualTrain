import logging

import pytest
import torch
from avalanche.benchmarks.classic import (
    PermutedMNIST,
    SplitMNIST,
    SplitTinyImageNet,
)
from avalanche.models import SimpleMLP
from datasets import load_dataset
from PIL import Image
from torch.utils.data import TensorDataset
from torchvision import transforms

from continualUtils.benchmarks import SplitClickMe

#################################### DEVICE ####################################

# Define a condition for skipping
skip_if_no_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Test requires GPU, but CUDA is not available.",
)


# Device fixture
@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


#################################### DEVICE ####################################

#################################### MODELS ####################################


# Pretrained fixture
@pytest.fixture
def pretrained_classifier(request, device):
    model_class = request.param["model_class"]
    model = model_class(output_hidden=False).to(device)
    return model


# Custom multihead fixture
@pytest.fixture
def custom_multihead_classifier(
    request, num_classes_per_task, make_multihead, device
):
    model_class = request.param["model_class"]
    num_classes_per_task = num_classes_per_task
    model = model_class(
        num_classes_per_task=num_classes_per_task,
        output_hidden=False,
        make_multihead=make_multihead,
    ).to(device)
    return model


# Simple MLP fixture
@pytest.fixture
def av_simple_mlp():
    model = SimpleMLP(num_classes=10)
    return model


#################################### MODELS ####################################

################################### DATASETS ###################################


# Split MNIST 5 experience fixture
@pytest.fixture
def split_mnist():
    split_mnist = SplitMNIST(
        n_experiences=5, dataset_root="/mnt/datasets/mnist"
    )
    return split_mnist


# Split Permuted MNIST 2 experience fixture
@pytest.fixture
def split_permuted_mnist():
    perm_mnist = PermutedMNIST(n_experiences=2)
    return perm_mnist


@pytest.fixture
def split_tiny_imagenet(request):
    # 200 classes
    split_tiny = SplitTinyImageNet(
        n_experiences=request.param["n_experiences"],
        dataset_root="/mnt/datasets/tinyimagenet",
        seed=42,
        return_task_id=True,
    )
    return split_tiny


@pytest.fixture
def split_clickme_benchmark():
    split_clickme = SplitClickMe(
        n_experiences=20,
        root="/mnt/datasets/clickme",
        seed=42,
        dummy=True,
        return_task_id=True,
    )
    return split_clickme


@pytest.fixture
def img_tensor_list():
    # Define the necessary transformations: convert to tensor and normalize
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),  # Imagenet standards
        ]
    )

    dataset = load_dataset(
        "hf-internal-testing/fixtures_image_utils", split="test"
    )

    image1 = Image.open(dataset[4]["file"])  # Cat image
    image2 = Image.open(dataset[5]["file"])  # Selena with hat image

    # Apply the transformations to the images and add a batch dimension
    tensor1 = transform(image1).unsqueeze(0)
    tensor2 = transform(image2).unsqueeze(0)

    images_tensor = [tensor1, tensor2]

    return images_tensor


@pytest.fixture
def img_tensor_dataset():
    # Define the necessary transformations: convert to tensor and normalize
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),  # Imagenet standards
        ]
    )

    dataset = load_dataset(
        "hf-internal-testing/fixtures_image_utils", split="test"
    )

    image1 = Image.open(dataset[4]["file"])  # Cat image

    # Apply the transformations to the images and add a batch dimension
    tensor1 = transform(image1).unsqueeze(0)

    return TensorDataset(tensor1, torch.Tensor([281]).long())


################################### DATASETS ###################################

################################### LOGGING ####################################


@pytest.fixture(scope="session")
def logger():
    # Create a logger for 'pytest'
    pytest_logger = logging.getLogger("pytest")
    pytest_logger.setLevel(logging.DEBUG)

    return pytest_logger


################################### LOGGING ####################################
