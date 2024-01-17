import pdb

import numpy as np
import PIL
from avalanche.benchmarks.utils.dataset_traversal_utils import (
    single_flat_dataset,
)

from continualUtils.benchmarks.datasets.clickme import (
    ClickMeDataset,
    ClickMeImageNetWrapperDataset,
    make_combined_clickme_dataset,
)


def test_combined_dataset():
    """Test loading the CombinedClickMeDataset"""

    dataset = make_combined_clickme_dataset(
        imagenet_root="/mnt/datasets/fake_imagenet",
        clickme_root="/mnt/datasets/clickme",
        imagenet_split="train",
        clickme_split="dtrain",
    )

    flat_set = single_flat_dataset(dataset)
    assert flat_set is not None


def test_wrapper_dataset():
    """Test loading ClickMeImageNetWrapperDataset with no transforms enabled"""
    dataset = ClickMeImageNetWrapperDataset(
        root="/mnt/datasets/fake_imagenet",
        split="train",
        apply_transform=False,
    )

    flat_set = single_flat_dataset(dataset)
    assert flat_set is not None

    # Call __getitem__
    assert len(dataset[0]) == 4
    image, label, heatmap, token = dataset[0]

    assert isinstance(image, PIL.Image.Image)  # type: ignore
    assert isinstance(label, int)
    assert isinstance(heatmap, np.ndarray)
    assert isinstance(token, int)

    assert heatmap.shape == (256, 256, 1)


def test_clickme_dataset():
    """Test loading ClickMeDataset with no transforms enabled"""
    dataset = ClickMeDataset(
        root="/mnt/datasets/clickme",
        split="dtrain",
        apply_transform=False,
    )

    flat_set = single_flat_dataset(dataset)
    assert flat_set is not None

    # Call __getitem__
    assert len(dataset[0]) == 4
    image, label, heatmap, token = dataset[0]

    assert isinstance(image, PIL.Image.Image)  # type: ignore
    assert isinstance(label, int)
    assert isinstance(heatmap, np.ndarray)
    assert isinstance(token, int)

    assert heatmap.shape == (256, 256, 1)


def test_combined_dataset_consistency():
    """Test that ClickMeImageNetWrapperDataset and ClickMeDataset in
    CombinedDataset load the same data types and shapes."""

    # Initialize datasets
    dataset1 = ClickMeImageNetWrapperDataset(
        root="/mnt/datasets/fake_imagenet",
        split="train",
        apply_transform=False,
    )
    dataset2 = ClickMeDataset(
        root="/mnt/datasets/clickme",
        split="dtrain",
        apply_transform=False,
    )

    # Optional: Check if the datasets are not empty
    assert len(dataset1) > 0
    assert len(dataset2) > 0

    # Number of samples to test
    num_samples_to_test = min(10, len(dataset1), len(dataset2))

    for i in range(num_samples_to_test):
        # Retrieve items from each dataset
        item1 = dataset1[i]
        item2 = dataset2[i]

        # Check for the same number of elements (e.g., image, label, etc.)
        assert len(item1) == len(item2)

        # Check each element
        for elem1, elem2 in zip(item1, item2):
            # Check for type consistency
            assert type(elem1) == type(elem2)

            # If the element is array-like, check their shapes
            if hasattr(elem1, "shape") and hasattr(elem2, "shape"):
                assert elem1.shape == elem2.shape  # type: ignore
