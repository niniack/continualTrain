from itertools import product

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision import transforms

from continualUtils.models import (
    CustomCvt13,
    CustomDeiTSmall,
    CustomResNet18,
    CustomResNet50,
    CustomWideResNet,
    PretrainedCvt13,
    PretrainedDeiTSmall,
    PretrainedResNet18,
    SimpleMNISTCNN,
)

from .conftest import skip_if_no_cuda

#################################### SETUP #####################################


# Hook to parameterize tests with specific arguments in this file
def pytest_generate_tests(metafunc):
    pretrained_models = [
        PretrainedCvt13,
        PretrainedDeiTSmall,
        PretrainedResNet18,
    ]
    custom_models = [
        CustomCvt13,
        CustomDeiTSmall,
        CustomResNet18,
    ]
    if metafunc.function.__name__ == "test_compare_counterparts":
        pretrained_params = [
            {"model_class": model_class} for model_class in pretrained_models
        ]
        custom_params = [
            {"model_class": model_class} for model_class in custom_models
        ]
        metafunc.parametrize(
            "pretrained_classifier,custom_multihead_classifier",
            zip(pretrained_params, custom_params),
            indirect=True,
        )
    else:
        if "pretrained_classifier" in metafunc.fixturenames:
            params = [
                {"model_class": model_class}
                for model_class in pretrained_models
            ]
            metafunc.parametrize(
                "pretrained_classifier",
                params,
                indirect=True,
            )

        if "custom_multihead_classifier" in metafunc.fixturenames:
            params = [
                {"model_class": model_class} for model_class in custom_models
            ]
            metafunc.parametrize(
                "custom_multihead_classifier",
                params,
                indirect=True,
            )

    if "split_tiny_imagenet" in metafunc.fixturenames:
        metafunc.parametrize(
            "split_tiny_imagenet",
            [
                {"n_experiences": 20},
            ],
            indirect=True,
        )


#################################### SETUP #####################################


@skip_if_no_cuda
def test_simple_cnn(device, split_mnist):
    model = SimpleMNISTCNN(
        num_classes_per_task=2,
        make_multihead=True,
    ).to(device)

    train_stream = split_mnist.train_stream
    exp_set = train_stream[0].dataset
    image, *_ = exp_set[0]
    image = image.unsqueeze(0).to(device)

    output = model(image, 0)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (1, 2)


@skip_if_no_cuda
def test_pretrained(pretrained_classifier, device, split_tiny_imagenet):
    """
    Test if pretrained classifiers are returning outputs, not multihead
    """
    train_stream = split_tiny_imagenet.train_stream
    exp_set = train_stream[0].dataset
    image, *_ = exp_set[0]
    image = F.interpolate(image.unsqueeze(0), (224, 224)).to(device)

    output = pretrained_classifier(image)

    assert torch.isclose(
        torch.sum(F.softmax(output, dim=1)), torch.tensor(1.0, device=device)
    ), "Softmax output does not sum to 1"
    assert isinstance(output, torch.Tensor)
    assert output.shape == (1, 1000)


@skip_if_no_cuda
def test_pretrained_accuracy(pretrained_classifier, device, img_tensor_list):
    """
    Test accuracy of pretrained classifiers, not multihead
    """

    resizer = transforms.Compose([transforms.Resize((224, 224))])

    imagenet_cat_ids = [281, 282, 283, 284, 285, 286, 287]  # various cats
    expected_cat = torch.argmax(
        pretrained_classifier.forward(resizer(img_tensor_list[0]).to(device))
    )

    imagenet_human_accessories = [
        474,
        515,
        655,
    ]  # cardigan, cowboy hat, miniskirt
    expected_person = torch.argmax(
        pretrained_classifier.forward(resizer(img_tensor_list[1]).to(device))
    )

    assert (
        expected_cat in imagenet_cat_ids
        and expected_person in imagenet_human_accessories
    )


@pytest.mark.parametrize(
    "model",
    [
        CustomResNet50(
            num_classes_per_task=10,
            make_multihead=True,
        ),
        SimpleMNISTCNN(
            num_classes_per_task=10,
            make_multihead=False,
        ),
    ],
)
def test_save_load(model, device, tmpdir):
    # Bring to GPU
    model = model.to(device)

    # Save
    pre_state_dict = model.state_dict()
    model.save_weights(f"{tmpdir}/model")

    # Load
    model.load_weights(f"{tmpdir}/model", device=device)
    post_state_dict = model.state_dict()

    # Function to compare two state dictionaries
    def compare_state_dicts(dict1, dict2):
        for key in dict1:
            # Check if key is present in both dictionaries
            if key not in dict2:
                print(f"Key '{key}' not found in second dictionary")
                return False

            # Check if both values are tensors
            if torch.is_tensor(dict1[key]) and torch.is_tensor(dict2[key]):
                # Check if tensors are on the same device
                if dict1[key].device != dict2[key].device:
                    print(
                        f"Tensor '{key}' is on different devices: {dict1[key].device} and {dict2[key].device}"
                    )
                    return False

                # Check if tensor values are equal
                if not torch.equal(dict1[key], dict2[key]):
                    print(f"Mismatch in values for key '{key}':")
                    print("First dict tensor:", dict1[key])
                    print("Second dict tensor:", dict2[key])
                    return False
            else:
                # Check non-tensor values
                if dict1[key] != dict2[key]:
                    print(
                        f"Mismatch in non-tensor values for key '{key}': {dict1[key]} and {dict2[key]}"
                    )
                    return False

        return True

    # Assert that the two state dictionaries are the same
    assert compare_state_dicts(pre_state_dict, post_state_dict)


@pytest.mark.parametrize("num_classes_per_task", [10])
def test_multihead(
    custom_multihead_classifier,
    num_classes_per_task,
    device,
    split_tiny_imagenet,
):
    """Tests multihead implementation"""

    train_stream = split_tiny_imagenet.train_stream
    exp_set = train_stream[0].dataset
    image, *_ = exp_set[0]
    image = F.interpolate(image.unsqueeze(0), (224, 224)).to(device)

    output = custom_multihead_classifier(image)
    assert isinstance(output, dict), "Output is not dict"

    output = custom_multihead_classifier(image, 0)
    assert isinstance(output, Tensor), "Output is not Tensor"


def test_model_weight_init(custom_multihead_classifier, device):
    """Test weight initialization"""
    epsilon = 1e-3

    # Check if model.model is an instance of torch.nn.Module
    assert isinstance(
        custom_multihead_classifier._model, torch.nn.Module
    ), "model.model is not an instance of torch.nn.Module!"

    # Check Kaiming Initialization for the first module
    for m in custom_multihead_classifier.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)

            # Check if weights' mean is close to 0
            assert (
                torch.abs(m.weight.mean()) < epsilon
            ), f"Mean of weights for {m} not close to 0"

            # Check if weights' variance is close to 2 / fan_in
            assert (
                torch.abs(m.weight.var() - 2 / fan_in) < epsilon
            ), f"Variance of weights for {m} not initialized correctly"

            if m.bias is not None:
                # Check if bias is initialized to 0
                assert (
                    torch.abs(m.bias.mean()) < epsilon
                ), f"Bias of {m} not initialized to 0"

            # Exit loop after checking the first module
            break


@pytest.mark.parametrize(
    "num_classes_per_task, make_multihead", [(1000, False)]
)
def test_compare_counterparts(
    custom_multihead_classifier, pretrained_classifier
):
    # Flatten the model architectures into lists of layers
    custom_layers = [layer for layer in custom_multihead_classifier.modules()]
    pretrained_layers = [layer for layer in pretrained_classifier.modules()]

    # Check if the lengths of the flattened architectures are the same
    assert len(custom_layers) == len(
        pretrained_layers
    ), "The models have a different number of layers"

    # # Iterate through the layers and compare their class types
    # for custom_layer, pretrained_layer in zip(custom_layers, pretrained_layers):
    #     assert type(custom_layer) == type(
    #         pretrained_layer
    #     ), f"Layer mismatch: {custom_layer} vs {pretrained_layer}"
