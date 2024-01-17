import copy
from collections import defaultdict
from typing import List

import torch
import torch.nn.functional as F
from avalanche.models import MultiTaskModule
from avalanche.models.utils import avalanche_forward
from avalanche.training.regularization import RegularizationMethod
from torch import Tensor

from continualUtils.explain.tools import (
    compute_saliency_map,
    compute_score,
    standardize_cut,
)


class LwMLoss(RegularizationMethod):
    """Learning without Memorizing (LwM) loss

    This method applies the LwM loss
    """

    def __init__(self, weight, prev_model=None):
        # Regularization weight
        self.weight = weight

        # Set previous model
        self.prev_model = prev_model
        self.expcount = 0
        self.prev_classes_by_task = defaultdict(set)
        self.sample_size = 1

    def __call__(self, mb_x, curr_model):
        return self.weight * self._lwm_penalty(mb_x, curr_model)

    def update(self, experience, model):
        self.expcount += 1
        # Clear out feature before saving model
        model.feature = None
        self.prev_model = copy.deepcopy(model)
        self.prev_model.requires_grad_(False)
        task_ids = experience.dataset.targets_task_labels.uniques

        for task_id in task_ids:
            task_data = experience.dataset.task_set[task_id]
            previous_classes = set(task_data.targets.uniques)

            if task_id not in self.prev_classes_by_task:
                self.prev_classes_by_task[task_id] = previous_classes
            else:
                self.prev_classes_by_task[task_id] = self.prev_classes_by_task[
                    task_id
                ].union(previous_classes)

    def _lwm_penalty(self, mb_x, curr_model):
        if self.prev_model is None:
            return 0

        # Previous model exists
        else:
            loss_att_dist = 0

            # Uses multihead implementation
            if isinstance(self.prev_model, MultiTaskModule):
                self.prev_model.eval()
                curr_model.train()

                # Gradients off
                with torch.no_grad():
                    # Run forward pass on all heads for previous model
                    y_prev: dict = avalanche_forward(
                        model=self.prev_model, x=mb_x, task_labels=None
                    )
                if not y_prev:
                    raise ValueError(
                        "Avalanche forward returned empty dictionaries"
                    )

                # Compute top base class predicted by new model
                top_base_classes_list: List[Tensor] = [torch.empty(0)] * len(
                    y_prev.keys()
                )

                # TODO: Double check this!!!! should it be for yprev or ycurr?
                for task_id, y_prev_for_task_id in y_prev.items():
                    # Store top base classes for each image in each task
                    top_base_classes_list[int(task_id)] = torch.argmax(
                        y_prev_for_task_id, dim=-1
                    )

                # Generate saliency maps for old model
                # Each head gets B saliency maps, B is batch
                # Total is (T-1)*B maps, T is old tasks
                prev_task_maps = [torch.empty(0)] * len(y_prev.keys())
                curr_task_maps = [torch.empty(0)] * len(y_prev.keys())

                for task_id, y_prev in y_prev.items():
                    # Top base classes for this task id
                    top_base_classes_for_task_id = top_base_classes_list[
                        int(task_id)
                    ]
                    one_hot_targets = F.one_hot(
                        top_base_classes_for_task_id,
                        curr_model.num_classes_per_head,
                    )

                    tensor_task = torch.tensor(task_id).unsqueeze(0)

                    # Build B saliency maps for task t
                    curr_map_for_task_id = compute_saliency_map(
                        pure_function=compute_score,
                        model=curr_model,
                        inputs=mb_x,
                        tasks=tensor_task,
                        targets=one_hot_targets,
                    )

                    prev_map_for_task_id = compute_saliency_map(
                        pure_function=compute_score,
                        model=self.prev_model,
                        inputs=mb_x,
                        tasks=tensor_task,
                        targets=one_hot_targets,
                    )

                    # Process B maps
                    prev_map_for_task_id = standardize_cut(prev_map_for_task_id)
                    prev_task_maps[int(task_id)] = prev_map_for_task_id  # type: ignore

                    curr_map_for_task_id = standardize_cut(curr_map_for_task_id)
                    curr_task_maps[int(task_id)] = curr_map_for_task_id  # type: ignore

                # TODO: vmap this
                # Note: len(prev_task_maps) == len(prev_task_maps)
                for i, prev_task_map in enumerate(prev_task_maps):
                    # Flatten maps in
                    prev_task_maps_for_task_id = F.normalize(
                        torch.flatten(prev_task_map, start_dim=1)
                    )
                    curr_task_maps_for_task_id = F.normalize(
                        torch.flatten(curr_task_maps[i], start_dim=1)
                    )
                    loss_att_dist += F.l1_loss(
                        curr_task_maps_for_task_id,
                        prev_task_maps_for_task_id,
                        reduction="mean",
                    )

            else:
                self.prev_model.train()
                curr_model.train()

                # Obtain latest input
                y_curr = avalanche_forward(curr_model, mb_x, None)

                # Run new input through old model
                y_prev = avalanche_forward(self.prev_model, mb_x, None)

                # Previously seen classes
                base_classes = list(self.prev_classes_by_task[0])

                # Mask unseen classes to 0
                mask = torch.ones(y_curr.shape, dtype=torch.bool)
                mask[:, base_classes] = 0
                y_curr[mask] = -float("inf")

                # Get top base classes for new model
                top_base_classes: Tensor = torch.argmax(y_curr, dim=-1)

                curr_map = compute_saliency_map(
                    pure_function=compute_score,
                    model=curr_model,
                    inputs=mb_x,
                    tasks=torch.tensor(0),
                    targets=top_base_classes,
                )

                curr_map = standardize_cut(curr_map)

                prev_map = compute_saliency_map(
                    pure_function=compute_score,
                    model=self.prev_model,
                    inputs=mb_x,
                    tasks=torch.tensor(0),
                    targets=top_base_classes,
                    grad_enabled=False,
                )

                prev_map = standardize_cut(prev_map)

                loss_att_dist += F.l1_loss(curr_map, prev_map, reduction="mean")  # type: ignore

            return loss_att_dist
