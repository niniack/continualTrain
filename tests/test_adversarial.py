import torch
import torchattacks
from avalanche.benchmarks.classic import SplitTinyImageNet
from avalanche.training.plugins import EWCPlugin
from avalanche.training.templates import SupervisedTemplate
from torch.nn import CrossEntropyLoss
from torch.optim import SGD

from continualUtils.models import (
    CustomResNet18,
    CustomResNet50,
    PretrainedResNet18,
    PretrainedResNet50,
)
from continualUtils.security import AdversarialAttackLoss


def test_adversarial_attack_loss(device):
    """Test Saliency Guided Loss"""
    adv_attack = AdversarialAttackLoss(
        attack=torchattacks.FGSM, attack_params={"eps": 1e-2}
    )

    tiny_imagenet = SplitTinyImageNet(
        n_experiences=1,
        dataset_root="/mnt/datasets/tinyimagenet",
    )

    model = CustomResNet18(
        num_classes_per_task=200,
        output_hidden=False,
    ).to(device)

    train_stream = tiny_imagenet.train_stream
    exp_set = train_stream[0].dataset
    image, label, token = exp_set[0]
    image = image.unsqueeze(0).to(device)
    label = torch.Tensor([label]).to(device).to(int)

    temp_mb_x = image.requires_grad_(True)
    loss, _, _ = adv_attack(temp_mb_x, label, token, model)
    assert isinstance(loss, torch.Tensor)
    assert loss.item() > 0.0


# def test_adversarial_uap(pretrained_resnet18, img_tensor_dataset):
#     """Test universal adversarial perturbation"""
#     # Define the attack
#     attack = UniversalAdversarialPerturbation()
#     eps = 0.3

#     # Apply the attack
#     transform = attack(
#         img_tensor_dataset,
#         pretrained_resnet18,
#         learning_rate=0.1,
#         iterations=10,
#         eps=eps,
#         device=torch.device("cpu"),
#     )

#     # Test the adversarial noise
#     img, *_ = img_tensor_dataset[0]
#     perturbed_img, *_ = transform(img_tensor_dataset)[0]
#     noise = perturbed_img - img
#     assert torch.all(torch.isclose(noise, torch.zeros_like(noise), atol=eps))


# def test_avalanche_adversarial_plugin(av_simple_mlp, av_split_permuted_mnist):
#     """Test the universal adversarial plugin"""
#     # loggers = []
#     # loggers.append(InteractiveLogger())

#     train_stream, test_stream = av_split_permuted_mnist

#     # Prepare for training & testing
#     optimizer = SGD(av_simple_mlp.parameters(), lr=0.001, momentum=0.9)
#     criterion = CrossEntropyLoss()

#     # Continual learning strategy
#     ewc = EWCPlugin(ewc_lambda=0.1)
#     adv_replay = AdversarialReplayPlugin()
#     strategy = SupervisedTemplate(
#         av_simple_mlp,
#         optimizer,
#         criterion,
#         train_mb_size=4096,
#         train_epochs=2,
#         eval_mb_size=4096,
#         plugins=[ewc, adv_replay],
#     )

#     # train and test loop
#     results = []
#     for train_task in train_stream:
#         strategy.train(train_task, num_workers=8)
#         results.append(strategy.eval(test_stream))
