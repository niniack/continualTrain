from logging import Logger
from typing import Any

import torch
import torch.nn.functional as F
from avalanche.benchmarks.scenarios.deprecated.dataset_scenario import (
    DatasetScenario,
)
from avalanche.benchmarks.scenarios.deprecated.new_classes.nc_scenario import (
    NCScenario,
)
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import alexnet

from continualUtils.benchmarks.datasets.clickme import HEATMAP_INDEX
from continualUtils.explain.losses.harmonizer_loss import NeuralHarmonizerLoss
from continualUtils.explain.losses.lwm_loss import LwMLoss
from continualUtils.explain.tools import (
    compute_grad_cam,
    compute_pyramidal_mse,
    compute_saliency_map,
    compute_score,
    standardize_cut,
)
from continualUtils.explain.tools.pyramidal import _pyramidal_representation
from continualUtils.models import (
    CustomResNet18,
    CustomResNet50,
    PretrainedResNet18,
    PretrainedResNet50,
)


def test_lwm_loss(
    split_clickme_benchmark: NCScenario | DatasetScenario[Any, Any, Any],
    device: torch.device,
    logger: Logger,
):
    # Define model
    old_model = CustomResNet18(
        device=device,
        num_classes_total=1000,
        num_classes_per_head=50,  # conftest splitclickme is 20 exp
        multihead=True,
    )

    new_model = CustomResNet50(
        device=device,
        num_classes_total=1000,
        num_classes_per_head=50,  # conftest splitclickme is 20 exp
        multihead=True,
    )

    # Set up benchmark
    train_stream = split_clickme_benchmark.train_stream
    exp = train_stream[0]
    exp_set = exp.dataset
    (image, label, heatmap, token, task) = exp_set[0]

    input_img = image.unsqueeze(0).requires_grad_(True).to(device)
    heatmap = heatmap.unsqueeze(0)
    token = token.unsqueeze(0)
    tasks = torch.tensor([task])
    labels = torch.tensor([label])

    # Adapt the models
    old_model.adapt_model(exp)
    new_model.adapt_model(exp)

    # Get loss object
    lwm_loss = LwMLoss(weight=1, prev_model=old_model)
    func_loss_diff_models = lwm_loss(mb_x=input_img, curr_model=new_model)

    lwm_loss = LwMLoss(weight=1, prev_model=new_model)
    func_loss_same_models = lwm_loss(mb_x=input_img, curr_model=new_model)

    logger.debug(func_loss_diff_models)
    logger.debug(func_loss_same_models)

    assert func_loss_diff_models > func_loss_same_models


def test_harmonizer_loss_pretrained(
    pretrained_resnet18: PretrainedResNet18,
    split_clickme_benchmark: NCScenario | DatasetScenario[Any, Any, Any],
):
    # Set up benchmark
    train_stream = split_clickme_benchmark.train_stream
    exp_set = train_stream[0].dataset
    (image, label, heatmap, token, task) = exp_set[0]

    # Define model
    model = pretrained_resnet18

    input_img = image.unsqueeze(0).requires_grad_(True)
    heatmap = heatmap.unsqueeze(0)
    token = token.unsqueeze(0)
    tasks = torch.tensor([task])
    labels = torch.tensor([label])

    sa_map = compute_saliency_map(
        pure_function=compute_score,
        model=model,
        inputs=input_img,
        tasks=tasks,
        targets=F.one_hot(labels, num_classes=model.num_classes_per_head),
    )
    sa_map_preprocess = standardize_cut(sa_map)
    heatmap_preprocess = standardize_cut(heatmap)

    # Get max
    eps = 1e-6
    with torch.no_grad():
        _sa_max = (
            torch.amax(sa_map_preprocess.detach(), dim=(2, 3), keepdim=True)
            + eps
        )
        _hm_max = torch.amax(heatmap_preprocess, dim=(2, 3), keepdim=True) + eps

        # Normalize the true heatmaps according to the saliency maps
        heatmap_preprocess = heatmap_preprocess / _hm_max * _sa_max

    manual_loss = compute_pyramidal_mse(
        sa_map_preprocess, heatmap_preprocess, token
    )

    nh_loss = NeuralHarmonizerLoss(weight=1)
    func_loss = nh_loss(
        mb_x=input_img,
        mb_y=labels,
        mb_heatmap=heatmap,
        model=model,
        mb_tokens=token,
        mb_tasks=tasks,
    )  # type: ignore

    assert torch.allclose(manual_loss, func_loss)


def test_cut_ground_maps(
    split_clickme_benchmark: NCScenario | DatasetScenario[Any, Any, Any]
):
    """Testing the output of standardizing SplitClickMe ground truth maps"""

    train_stream = split_clickme_benchmark.train_stream
    exp_set = train_stream[0].dataset
    ground_map = exp_set[0][HEATMAP_INDEX].unsqueeze(0)

    preprocessed_heatmaps = standardize_cut(ground_map)

    # This should only work pre-relu step
    # mean_val = torch.mean(preprocessed_heatmaps, dim=(2, 3)).item()
    # assert abs(mean_val) < 1e-5, f"Mean value is not close to 0, got {mean_val}"

    assert torch.min(preprocessed_heatmaps) >= 0.0


# def test_grad_cam(img_tensor_list: list[Any]):
#     """Testing saliency map"""

#     model = PretrainedResNet50(device=torch.device("cpu"))
#     transform = transforms.Compose([transforms.Resize((224, 224))])

#     # Apply the transform to each image
#     img1 = transform(img_tensor_list[0])
#     img2 = transform(img_tensor_list[1])

#     # Stack the images to create a batch
#     inputs = torch.cat([img1, img2]).requires_grad_(True)
#     targets = torch.Tensor([281, 281]).long()
#     tasks = torch.Tensor([0, 0]).long()

#     # Grab last layer
#     target_layer_name = "_model.resnet.encoder.stages.3"

#     # One hot the targets
#     targets = F.one_hot(targets, num_classes=1000)

#     # Compute the saliency map
#     grad_cam_map = compute_grad_cam(
#         model=model,
#         inputs=inputs,
#         targets=targets,
#         target_layer_name=target_layer_name,
#         tasks=tasks,
#         grad_enabled=False,
#     )

#     # assert grad_cam_map.shape == (inputs.shape[0], 1, *(inputs.shape[-2:]))

#     # print(torch.mean(grad_cam_map))
#     # print(torch.std(grad_cam_map))

#     # # Visualize the first Grad-CAM heatmap
#     # import matplotlib.pyplot as plt

#     # plt.imshow(
#     #     grad_cam_map[0].squeeze().detach().numpy(), cmap="jet", alpha=0.5
#     # )
#     # plt.savefig("./grad_cam.png")


# def test_perturbed():
#     model = alexnet(pretrained=False)
#     print(model)
#     # Store for activations
#     activations = {}

#     # Function to be called by the hook
#     def hook_fn(module, input, output):
#         activations["linear"] = output  # Store the output

#     # Register the hook
#     # hook = model.linear.register_forward_hook(hook_fn)

#     # Batched inputs
#     xs = torch.rand(1, 3, 256, 256)
#     delta_ys = torch.rand(1, 3, 224, 224)

#     def f_perturbed(params, x, delta_y):
#         # y = torch.func.functional_call(model, params, x)
#         y = model(x)
#         # Run the forward pass to trigger the hook
#         # Modify the activation directly
#         print(params)
#         # modified_activation = activations["linear"].squeeze() + delta_y
#         return y

#     # Vectorize the gradient computation over the batch
#     # Assuming torch.vmap works similar to JAX's vmap
#     grad = torch.func.grad(f_perturbed, argnums=(1), has_aux=False)
#     params_dict = {name: param for name, param in model.named_parameters()}

#     batched_grad = torch.vmap(grad, in_dims=(None, 0, 0), randomness="same")(
#         params_dict, xs, delta_ys
#     )
#     print(batched_grad)
#     # assert torch.sum(batched_grad).item() != 0.0

#     # # Remove the hook when done to prevent memory leak
#     # hook.remove()


def test_saliency_map(img_tensor_list: list[Any]):
    """Testing saliency map"""

    model = PretrainedResNet50(device=torch.device("cpu"))
    inputs = img_tensor_list[0].requires_grad_(True)
    targets = torch.Tensor([281]).long()
    tasks = torch.Tensor([0]).long()

    # One hot the targets
    targets = F.one_hot(targets, num_classes=1000)

    # Compute the saliency map
    saliency_map = compute_saliency_map(
        pure_function=compute_score,
        model=model,
        inputs=inputs,
        targets=targets,
        tasks=tasks,
        grad_enabled=False,
        single_channel=False,
    )

    assert saliency_map.shape == inputs.shape


def test_pyramidal_mse():
    """
    Test the compute_pyramidal_mse function
    """
    heatmaps = torch.rand((10, 1, 224, 224))
    predicted_heatmaps = torch.rand((10, 1, 224, 224))
    tokens = torch.rand((10,))

    loss = compute_pyramidal_mse(heatmaps, predicted_heatmaps, tokens)

    assert isinstance(loss, torch.Tensor)
    assert loss.shape == torch.Size([])  # Checking for scalar tensor

    loss_none = compute_pyramidal_mse(
        heatmaps, predicted_heatmaps, torch.zeros((10,))
    )
    assert torch.allclose(loss_none, torch.zeros(1))

    loss_all = compute_pyramidal_mse(
        heatmaps, predicted_heatmaps, torch.ones((10,))
    )

    assert loss_all.item() >= loss.item() >= loss_none.item()


def test_pyramidal_representation():
    """Test the pyramidal representation"""
    # Define a sample maps tensor
    maps = torch.tensor(
        [
            [
                [
                    [1.0, 2.0, 3.0, 4.0],
                    [5.0, 6.0, 7.0, 8.0],
                    [9.0, 10.0, 11.0, 12.0],
                    [13.0, 14.0, 15.0, 16.0],
                ]
            ]
        ]
    )

    # Apply _pyramidal_representation
    pyramid = _pyramidal_representation(maps, num_levels=2)

    # Check the sizes of the pyramid levels
    assert pyramid[0].shape == (1, 1, 4, 4)
    assert pyramid[1].shape == (1, 1, 2, 2)
    assert pyramid[2].shape == (1, 1, 1, 1)

    # Check some known values (you can expand on this)
    assert torch.allclose(
        pyramid[1][0, 0, 0, 0], torch.tensor(3.5)
    )  # Average of top-left 2x2 block
