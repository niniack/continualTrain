from typing import TYPE_CHECKING, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from avalanche.benchmarks.utils.data import AvalancheDataset
from avalanche.evaluation.metric_definitions import PluginMetric
from avalanche.evaluation.metric_results import (
    MetricResult,
    MetricValue,
    TensorImage,
)
from avalanche.evaluation.metric_utils import get_metric_name
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from typing_extensions import Literal

from continualUtils.benchmarks.datasets.clickme import HEATMAP_INDEX
from continualUtils.explain.losses.harmonizer_loss import compute_score
from continualUtils.explain.tools import compute_saliency_map
from continualUtils.general.rescale_batch import rescale_batch

if TYPE_CHECKING:
    from avalanche.training.templates import SupervisedTemplate


class SaliencyMapSamplePlugin(PluginMetric):
    """Metric used to sample random saliency map outputs.

    :param n_rows: The numbers of raws to use in the grid of images.
    :param n_cols: The numbers of columns to use in the grid of images.
    :param group: If True, images will be grouped by (task, label)
    :param mode: The plugin can be used at train or eval time.
    :return: The corresponding plugins.
    """

    def __init__(
        self,
        *,
        mode: Literal["train", "eval", "both"],
        n_cols: int,
        n_rows: int,
        group: bool = True,
    ):
        super().__init__()
        self.group = group
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.mode = mode

        self.images: List[Tensor] = []
        self.ground_maps: List[Tensor] = []
        self.computed_maps: List[Tensor] = []
        self.labels: List[Tensor] = []
        self.n_wanted_images = self.n_cols * self.n_rows

    def after_training_exp(
        self, strategy: "SupervisedTemplate"
    ) -> "MetricResult":
        if self.mode == "train" or self.mode == "both":
            results = []
            for exp in range(strategy.experience.current_experience + 1):
                try:
                    result = self._make_grid_sample(strategy, exp)
                    if result:
                        results.extend(result)
                except Exception:
                    pass

            return results
        return None

    def after_eval_exp(self, strategy: "SupervisedTemplate") -> "MetricResult":
        pass

    def reset(self) -> None:
        self.images = []
        self.ground_maps = []
        self.computed_maps = []
        self.labels = []

    def result(self) -> List[Tensor]:
        return self.computed_maps

    def __str__(self):
        return "Sample_Saliency_Maps"

    def _make_grid_sample(
        self, strategy: "SupervisedTemplate", experience: int
    ) -> "MetricResult":
        # Load sorted images
        self._load_sorted_images(strategy, experience)

        # Rescale the heatmaps between 0 and 1
        rescaled_images = rescale_batch(self.images).cpu()
        rescaled_maps = rescale_batch(self.computed_maps).cpu()

        # Overlay the heatmaps
        overlayed_images = show_cam_on_image(
            rescaled_images, rescaled_maps, use_rgb=True
        )

        # Create grid from images
        grid = make_grid(
            overlayed_images,
            normalize=False,
            nrow=self.n_cols,
        )

        # Permute for numpy format
        grid = grid.permute(1, 2, 0)

        return [
            MetricValue(
                self,
                name=get_metric_name(
                    self,
                    strategy,
                    add_experience=(self.mode == "eval"),
                    add_task=experience,  # type: ignore
                ),
                value=TensorImage(grid),
                x_plot=strategy.clock.train_iterations,
            )
        ]

    def _load_sorted_images(
        self, strategy: "SupervisedTemplate", experience: int
    ):
        self.reset()
        (
            self.images,
            self.ground_maps,
            self.computed_maps,
            self.labels,
            tasks,
        ) = self._load_data(strategy, experience)
        if self.group:
            self._sort_images(tasks)  # type: ignore

    def _sort_images(self, tasks: List[int]):
        # Zip together all lists with the tasks for sorting
        combined_data = zip(
            tasks,
            self.labels,
            self.images,
            self.ground_maps,
            self.computed_maps,
        )

        # Sort the combined data based on the tasks first, then labels
        sorted_data = sorted(combined_data, key=lambda t: (t[0], t[1]))

        # Unzip the sorted data into their respective lists
        tasks, labels, images, ground_maps, computed_maps = zip(*sorted_data)

        # Assign the sorted lists back to the object's attributes
        self.labels = list(labels)
        self.images = list(images)
        self.ground_maps = list(ground_maps)
        self.computed_maps = list(computed_maps)

    def _load_data(
        self, strategy: "SupervisedTemplate", experience: int
    ) -> Tuple[
        List[Tensor], List[Tensor], List[Tensor], List[Tensor], List[Tensor]
    ]:
        # Grab the dataset from the experience provided
        stream = strategy.experience.origin_stream
        experience_dataset = stream[experience].dataset

        dataloader = self._make_dataloader(
            experience_dataset, strategy.eval_mb_size
        )

        # Initialize
        images: List[Tensor] = []
        ground_maps: List[Tensor] = []
        computed_maps: List[Tensor] = []
        labels: List[Tensor] = []
        tasks: List[Tensor] = []

        for batch in dataloader:
            # Unpack the batch
            *batch_others, batch_tasks = batch
            batch_images, batch_labels = batch_others[:2]
            batch_ground_maps = batch_others[HEATMAP_INDEX]

            # Run saliency map computation
            batch_computed_maps = self._compute_maps(
                batch_images, batch_labels, batch_tasks, strategy
            )

            # Grab only the needed number of samples
            n_missing_images = self.n_wanted_images - len(images)
            images.extend(batch_images[:n_missing_images])
            ground_maps.extend(batch_ground_maps[:n_missing_images])
            computed_maps.extend(batch_computed_maps[:n_missing_images])
            labels.extend(batch_labels[:n_missing_images].tolist())
            tasks.extend(batch_tasks[:n_missing_images].tolist())

            # Return if quota met
            if len(images) == self.n_wanted_images:
                break

        return images, ground_maps, computed_maps, labels, tasks

    def _make_dataloader(
        self, data: AvalancheDataset, mb_size: int
    ) -> DataLoader:
        collate_fn = data.collate_fn if hasattr(data, "collate_fn") else None
        return DataLoader(
            dataset=data,  # type: ignore
            batch_size=min(mb_size, self.n_wanted_images),
            shuffle=False,
            collate_fn=collate_fn,
        )

    def _compute_maps(
        self,
        images: Tensor,
        labels: Tensor,
        tasks: Tensor,
        strategy: "SupervisedTemplate",
    ) -> Tensor:
        # images = preprocess_input(images)
        inputs = images.to(strategy.device).requires_grad_(True)

        if labels.shape[-1] is not strategy.model.num_classes_per_head:
            targets = F.one_hot(labels, strategy.model.num_classes_per_head).to(
                strategy.device
            )
        else:
            targets = labels.to(strategy.device)

        computed_maps = compute_saliency_map(
            pure_function=compute_score,
            model=strategy.model,
            inputs=inputs,
            tasks=tasks,
            targets=targets,
        )
        return computed_maps.detach()


def show_cam_on_image(
    images: Tensor,
    maps: Tensor,
    use_rgb: bool = True,
    colormap: int = cv2.COLORMAP_JET,
    image_weight: float = 0.5,
) -> Tensor:
    """Overlay saliency maps with a colormap on image

    :param images: 4D tensor of images in NCHW
    :param maps: 4D tensor of maps in NCHW, should have 1 channel
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used, defaults to cv2.COLORMAP_JET
    :param image_weight: The final result is image_weight * img + (1-image_weight) * mask, defaults to 0.5
    :return: 4D tensor of overlayed images in NCHW
    """
    # Convert only maps to numpy in [0, 255] integer range
    maps_np = (maps.numpy() * 255).astype(np.uint8)

    # Reshape for applyColorMap: (N * H, W)
    h, w = maps_np.shape[2], maps_np.shape[3]
    heatmaps = maps_np.squeeze(1).reshape(-1, w)

    # Apply colormap to the reshaped maps
    heatmaps = cv2.applyColorMap(heatmaps, colormap)
    if use_rgb:
        heatmaps = cv2.cvtColor(heatmaps, cv2.COLOR_BGR2RGB)

    # Reshape back to original: (N, C, H, W)
    # cv2 returns channel last
    heatmaps = heatmaps.reshape(-1, h, w, 3)
    heatmaps = np.transpose(heatmaps, (0, 3, 1, 2))

    # Convert heatmaps back to [0, 1] range
    heatmaps = heatmaps.astype(np.float32) / 255.0

    # Merge
    cams = (1 - image_weight) * heatmaps + image_weight * images.numpy()

    # Convert back to tensor
    return torch.from_numpy(255 * cams).to(torch.uint8)


def saliency_map_samples_metrics(
    *,
    n_rows: int = 8,
    n_cols: int = 8,
    group: bool = True,
    on_train: bool = True,
    on_eval: bool = False,
) -> List[SaliencyMapSamplePlugin]:
    """
    Create the plugins to log some images samples in grids.
    No data augmentation is shown.
    Only images in strategy.adapted dataset are used. Images added in the
    dataloader (like the replay plugins do) are missed.

    :param n_rows: The numbers of raws to use in the grid of images.
    :param n_cols: The numbers of columns to use in the grid of images.
    :param group: If True, images will be grouped by (task, label)
    :param on_train: If True, will emit some images samples during training.
    :param on_eval: If True, will emit some images samples during evaluation.
    :return: The corresponding plugins.
    """
    plugins: List[SaliencyMapSamplePlugin] = []
    if on_eval:
        plugins.append(
            SaliencyMapSamplePlugin(
                mode="eval", n_rows=n_rows, n_cols=n_cols, group=group
            )
        )
    if on_train:
        plugins.append(
            SaliencyMapSamplePlugin(
                mode="train", n_rows=n_rows, n_cols=n_cols, group=group
            )
        )
    return plugins
