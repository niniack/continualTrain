"""
Dataset access code adapted from the Harmonization project:
https://github.com/serre-lab/Harmonization/blob/main/harmonization/common/clickme_dataset.py
"""
import bisect
import json
from pathlib import Path
from typing import Callable, Literal, Optional

import numpy as np
import torchvision.transforms as tv_transforms
from avalanche.benchmarks.utils import _make_taskaware_classification_dataset
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets

# Dataset hosting info
CLICKME_BASE_URL = (
    "https://connectomics.clps.brown.edu/tf_records/clicktionary_files/"
)
TRAIN_ZIP = "clickme_train.zip"
TEST_ZIP = "clickme_test.zip"
VAL_ZIP = "clickme_val.zip"

LOCAL_PATH = "~/datasets/clickme/"

HEATMAP_INDEX = 2
TOKEN_INDEX = 3

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def make_combined_clickme_dataset(
    clickme_root: str,
    clickme_split: Literal["train", "val", "test", "dtrain", "dtest"],
    imagenet_root: Optional[str] = None,
    imagenet_split: Optional[Literal["train", "val"]] = None,
):
    """Returns a ConcatDataset with both ClickMe and Imagenet"""
    dataset = CombinedClickMeDataset(
        imagenet_root=imagenet_root,
        clickme_root=clickme_root,
        imagenet_split=imagenet_split,
        clickme_split=clickme_split,
    )
    return _make_taskaware_classification_dataset(dataset)


def make_clickme_style_imagenet_dataset(
    root: str, split: Literal["train", "val"]
):
    """Returns ClickMeImageNetWrapperDataset as an Avalanche Dataset"""
    dataset = ClickMeImageNetWrapperDataset(root=root, split=split)
    return _make_taskaware_classification_dataset(dataset)


def make_clickme_dataset(
    root: str, split: Literal["train", "val", "test", "dtrain", "dtest"]
):
    """Returns ClickMe as an Avalanche Dataset"""
    dataset = ClickMeDataset(root=root, split=split)
    return _make_taskaware_classification_dataset(dataset)


class CombinedClickMeDataset(Dataset):
    """
    Dataset class to combine ImageNet and ClickMe datasets with a common transformation.
    Acts as a single dataset, hiding the fact that it is a combination of two datasets.
    """

    def __init__(
        self,
        clickme_root: str,
        clickme_split: Literal["train", "val", "test", "dtrain", "dtest"],
        imagenet_root: Optional[str] = None,
        imagenet_split: Optional[Literal["train", "val"]] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        heatmap_transform: Optional[Callable] = None,
    ):
        # Validate splits
        valid_clickme_splits = ["train", "val", "test", "dtrain", "dtest"]
        if clickme_split not in valid_clickme_splits:
            raise ValueError(
                f"ClickMe split must be one of {valid_clickme_splits}."
            )

        valid_imagenet_splits = ["train", "val"]
        if (
            imagenet_split is not None
            and imagenet_split not in valid_imagenet_splits
        ):
            raise ValueError(
                f"Imagenet split must be one of {valid_imagenet_splits}."
            )

        # Validate imagenet
        if imagenet_split is not None and imagenet_root is None:
            raise ValueError(
                "Imagenet split is given, but no root was provided!"
            )

        # Force the same transforms on both datasets
        # ToTensor() necessarily implies a move to device (from testing)
        # But, flip must occur before that
        if transform is None:
            # Avalanche will automatically swap out
            # ffcv SimpleRGBImageDecoder + tv RandomResizedCrop
            # for ffcv RandomResizedCropRGBImageDecoder
            # as an optimization
            self.transform = tv_transforms.Compose(
                [
                    # tv_transforms.RandomResizedCrop((224, 224), antialias=True),
                    # tv_transforms.RandomHorizontalFlip(p=0.5),
                    tv_transforms.ToTensor(),
                    tv_transforms.Normalize(
                        mean=IMAGENET_MEAN, std=IMAGENET_STD
                    ),
                ]
            )

        # Define composed transforms for heatmap
        if heatmap_transform is None:
            self.heatmap_transform = tv_transforms.Compose(
                [
                    tv_transforms.ToTensor(),
                    tv_transforms.Resize((64, 64), antialias=True),
                    tv_transforms.GaussianBlur(
                        kernel_size=(7, 7), sigma=(7, 7)
                    ),
                    tv_transforms.Resize((224, 224), antialias=True),
                ]
            )

        self.target_transform = target_transform

        # Initialize datasets
        self.datasets = []

        self.datasets.append(
            ClickMeDataset(
                clickme_root,
                clickme_split,
                self.transform,
                self.target_transform,
                apply_transform=False,
            )
        )

        if imagenet_split is not None and imagenet_root is not None:
            self.datasets.append(
                ClickMeImageNetWrapperDataset(
                    imagenet_root,
                    imagenet_split,
                    self.transform,
                    self.target_transform,
                    apply_transform=False,
                )
            )

        # Initialize targets
        self.targets = []
        for dataset in self.datasets:
            self.targets.extend(dataset.targets)

        # Compute cumulative sizes
        self.cumulative_sizes = self._cumsum(self.datasets)

    @staticmethod
    def _cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError("The index is out of range.")

        dataset_index = bisect.bisect_right(self.cumulative_sizes, index)
        sample_index = (
            index - self.cumulative_sizes[dataset_index - 1]
            if dataset_index > 0
            else index
        )

        # Unpack
        image, label, heatmap, token = self.datasets[dataset_index][
            sample_index
        ]

        # NOTE: We force a resize to 224 to avoid using random resized crop
        image = tv_transforms.functional.resize(
            image, (224, 224), antialias=True
        )

        # Transform images
        if self.transform is not None:
            image = self.transform(image)

        # Apply any transformations to labels
        if self.target_transform is not None:
            label = self.target_transform(label)

        # NOTE: Avalanche will yank transform and target_transform
        # in favor of the FFCV decode pipeline it generates.
        # However, heatmap_transform will remain because Avalanche is unaware of it.
        # This means dataset will be stored with heatmap transforms.
        # Don't double apply them.

        # Process heatmap
        if self.heatmap_transform is not None:
            heatmap = self.heatmap_transform(heatmap)

        return image, label, heatmap, token


class ClickMeImageNetWrapperDataset(datasets.ImageNet):
    """Dataset generator that wraps around ImageNet to return ClickMe style dataset."""

    def __init__(
        self,
        root: str,
        split: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        apply_transform: bool = True,
        **kwargs,
    ):
        self.apply_transform = apply_transform

        # If no transform is provided, define the default
        if transform is None:
            transform = tv_transforms.Compose(
                [
                    # tv_transforms.RandomResizedCrop((224, 224), antialias=True),
                    # tv_transforms.RandomHorizontalFlip(p=0.5),
                    tv_transforms.ToTensor(),
                    tv_transforms.Normalize(
                        mean=IMAGENET_MEAN, std=IMAGENET_STD
                    ),
                ]
            )

        super().__init__(
            root,
            split=split,
            transform=transform if apply_transform else None,
            target_transform=target_transform if apply_transform else None,
            **kwargs,
        )

        # Pre-allocate a static heatmap
        self.static_heatmap = np.empty((256, 256, 1), dtype=np.float32)

    def __getitem__(self, index: int):
        # Retrieve the image and label from the ImageNet dataset
        image, label = super().__getitem__(index)

        # NOTE: Transforms will be automatically applied by the parent class

        # Return the shared static heatmap and a placeholder token
        token = int(0)

        return image, label, self.static_heatmap, token


class ClickMeDataset(Dataset):
    """Dataset generator for the ClickMe dataset. Returns torch tensors.

    The ClickMe dataset contains 196,499 unique images.
    """

    def __init__(
        self,
        root: str,
        split: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        apply_transform: bool = True,
    ):
        self.root = root
        self.split = split

        self.split_dir = "clickme_" + split + "/"

        self.full_path = Path(self.root).joinpath(self.split_dir)
        self.files = list(self.full_path.glob("*.npz"))

        self.apply_transform = apply_transform

        # Load targets
        target_path = self.full_path.joinpath("metadata.json")
        if target_path.exists():
            with open(target_path, "r", encoding="utf-8") as meta_file:
                metadata = json.load(meta_file)
                self.targets = [metadata[file.name] for file in self.files]
        else:
            self.targets = [
                int(np.load(str(self.files[x]))["label"])
                for x in range(len(self.files))
            ]

        # Define composed transforms for images
        self.transform = (
            tv_transforms.Compose(
                [
                    # tv_transforms.RandomResizedCrop((224, 224), antialias=True),
                    # tv_transforms.RandomHorizontalFlip(p=0.5),
                    tv_transforms.ToTensor(),
                    tv_transforms.Normalize(
                        mean=IMAGENET_MEAN, std=IMAGENET_STD
                    ),
                ]
            )
            if transform is None
            else transform
        )

        # Define composed transforms for heatmap
        self.heatmap_transform = tv_transforms.Compose(
            [
                tv_transforms.ToTensor(),
                tv_transforms.Resize((64, 64), antialias=True),
                tv_transforms.GaussianBlur(kernel_size=(7, 7), sigma=(7, 7)),
                tv_transforms.Resize((224, 224), antialias=True),
            ]
        )

        # Define composed transforms for labels if needed
        self.target_transform = target_transform

    def __len__(self):
        """
        Returns the size of the dataset
        """
        return len(self.targets)

    def __getitem__(self, index):
        data = np.load(str(self.files[index]))
        image = data["image"]
        label = data["label"]
        heatmap = data["heatmap"]

        # Hardcode np image to PIL
        image = Image.fromarray(image.astype("uint8"), "RGB")

        # Transform images
        if self.apply_transform and self.transform is not None:
            image = self.transform(image)

        # Transform labels
        label = int(label)
        if self.apply_transform and self.target_transform is not None:
            label = self.target_transform(label)

        # Expand heatmap and transform
        heatmap = heatmap[..., np.newaxis]
        if self.apply_transform and self.heatmap_transform is not None:
            heatmap = self.heatmap_transform(heatmap)

        # Instantiate token
        token = int(1)

        return image, label, heatmap, token
