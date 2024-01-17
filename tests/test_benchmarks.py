import pdb

import ffcv
import torch
import torchvision.transforms as tv_transforms
from avalanche.benchmarks import AvalancheDataset, SplitTinyImageNet
from avalanche.benchmarks.utils.data_loader import TaskBalancedDataLoader
from avalanche.benchmarks.utils.ffcv_support import (
    HybridFfcvLoader,
    enable_ffcv,
)
from avalanche.benchmarks.utils.ffcv_support.ffcv_transform_utils import (
    SmartModuleWrapper,
)
from torch.utils.data.sampler import BatchSampler, SequentialSampler
from tqdm import tqdm

from continualUtils.benchmarks import SplitClickMe
from continualUtils.benchmarks.datasets.ffcv_transforms import (
    RandomHorizontalFlipSeeded,
)


def test_load_splitimagenet(device, tmpdir):
    """Test loading SplitImageNet with FFCV"""
    split_tiny_imagenet = SplitTinyImageNet(
        n_experiences=10, dataset_root="/mnt/datasets/tinyimagenet", seed=42
    )
    benchmark_type = "tinyimagenet"
    enable_ffcv(
        benchmark=split_tiny_imagenet,
        write_dir=f"{tmpdir}/ffcv_test_{benchmark_type}",
        device=device,
        ffcv_parameters=dict(num_workers=8),
        print_summary=True,
    )


def test_load_splitclickme():
    """Test loading the SplitClickMe benchmark"""

    ds_root = "/mnt/datasets/clickme"

    split_clickme = SplitClickMe(
        n_experiences=10, root=ds_root, seed=42, dummy=False
    )

    assert split_clickme.train_stream is not None
    assert split_clickme.test_stream is not None
    assert split_clickme.val_stream is not None

    val_exp = next(iter(split_clickme.val_stream), None)
    assert val_exp is not None

    assert hasattr(val_exp, "classes_in_this_experience")

    dataloader = TaskBalancedDataLoader(
        split_clickme.train_stream[0].dataset,
        oversample_small_groups=True,
    )

    batch = next(iter(dataloader))

    assert len(batch) is 5

    image, label, heatmap, token, task = batch


def test_ffcv_clickme(device, tmpdir):
    """Test loading the SplitClickMe benchmark"""
    device = torch.device("cpu")
    dataset_root = "/mnt/datasets/clickme"
    epochs = 5
    batch_size = 16
    num_workers = 2
    # benchmark = SplitTinyImageNet(
    #     n_experiences=10, dataset_root="/mnt/datasets/tinyimagenet", seed=42
    # )
    benchmark = SplitClickMe(
        n_experiences=2,
        root=dataset_root,
        seed=79,
        dummy=True,
        return_task_id=True,
        shuffle=True,
        class_ids_from_zero_in_each_exp=True,
        fixed_class_order=list(range(0, 1000)),
    )

    custom_decoder_pipeline = {
        "field_0": [
            ffcv.fields.rgb_image.SimpleRGBImageDecoder(),
            RandomHorizontalFlipSeeded(0.5),
            # ffcv.fields.rgb_image.RandomResizedCropRGBImageDecoder((224, 224))
        ],
        "field_1": [
            ffcv.fields.basics.IntDecoder(),
            ffcv.transforms.ToTensor(),
        ],
        "field_2": [
            ffcv.fields.ndarray.NDArrayDecoder(),
            RandomHorizontalFlipSeeded(0.5),
            ffcv.transforms.ToTensor(),
            # ffcv.transforms.RandomHorizontalFlip(0.5),
            # ffcv.transforms.ToTorchImage(),
            # SmartModuleWrapper(tv_transforms.Resize((64, 64), antialias=True)),
            # SmartModuleWrapper(
            #     tv_transforms.GaussianBlur(kernel_size=(7, 7), sigma=(7, 7))
            # ),
            # SmartModuleWrapper(
            #     tv_transforms.Resize((224, 224), antialias=True)
            # ),
            # ffcv.transforms.ToDevice(device),
        ],
        "field_3": [
            ffcv.fields.basics.IntDecoder(),
            ffcv.transforms.ToTensor(),
        ],
    }

    # custom_decoder_pipeline = None

    enable_ffcv(
        benchmark=benchmark,
        write_dir=f"{tmpdir}/ffcv_clickme",
        device=device,
        ffcv_parameters=dict(
            num_workers=num_workers,
            write_mode="proportion",
            compress_probability=0.25,
            max_resolution=256,
            jpeg_quality=90,
            os_cache=True,
            seed=10,
        ),
        decoder_def=custom_decoder_pipeline,
        decoder_includes_transformations=False,
        force_overwrite=True,
        print_summary=True,  # Better keep this true on non-benchmarking code
    )

    all_train_dataset = [x.dataset for x in benchmark.train_stream]
    avl_set = AvalancheDataset(all_train_dataset)
    avl_set = avl_set.train()

    ffcv_loader = HybridFfcvLoader(
        dataset=avl_set,
        batch_sampler=BatchSampler(
            SequentialSampler(avl_set),
            batch_size=batch_size,
            drop_last=True,
        ),
        ffcv_loader_parameters=dict(
            num_workers=num_workers,
        ),
        device=device,
        persistent_workers=True,
        print_ffcv_summary=True,
    )

    for _ in tqdm(range(epochs)):
        for batch in tqdm(ffcv_loader):
            # "Touch" tensors to make sure they already moved to GPU
            batch[0][0]
            batch[-1][0]
