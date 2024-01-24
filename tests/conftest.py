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
from continualUtils.models import PretrainedResNet18

# Define a condition for skipping
skip_if_no_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Test requires GPU, but CUDA is not available.",
)


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def pretrained_resnet18():
    model = PretrainedResNet18(device=torch.device("cpu"))
    return model


@pytest.fixture
def split_permuted_mnist():
    perm_mnist = PermutedMNIST(n_experiences=2)
    return perm_mnist


@pytest.fixture
def split_mnist():
    split_mnist = SplitMNIST(
        n_experiences=5, dataset_root="/mnt/datasets/mnist"
    )
    return split_mnist


@pytest.fixture
def split_tiny_imagenet():
    # 200 classes
    split_tiny = SplitTinyImageNet(
        n_experiences=20,
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


@pytest.fixture
def av_simple_mlp():
    model = SimpleMLP(num_classes=10)
    return model


@pytest.fixture(scope="session")
def logger():
    # Create a logger for 'pytest'
    pytest_logger = logging.getLogger("pytest")
    pytest_logger.setLevel(logging.DEBUG)

    return pytest_logger
