# import torch
# import torch.nn.functional as F
# from avalanche.benchmarks.classic import SplitMNIST
# from avalanche.models import SimpleMLP
# from avalanche.training import Naive
# from torch import Tensor, nn
# from torch.nn import CrossEntropyLoss
# from torch.optim import SGD

# from continualUtils.schedulers import LRFinderScheduler, LRFinderSchedulerPlugin


# def test_lr_finder():
#     EPOCHS = 5

#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     # model
#     model = SimpleMLP(num_classes=10)

#     # CL Benchmark Creation
#     mnist = SplitMNIST(n_experiences=1, dataset_root="/mnt/datasets/mnist")
#     train_stream = mnist.train_stream
#     test_stream = mnist.test_stream

#     # Prepare for training & testing
#     optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
#     criterion = CrossEntropyLoss()

#     scheduler = LRFinderScheduler(
#         optimizer=optimizer,
#         epochs=EPOCHS,
#         min_lr=1e-3,
#         max_lr=1e-1,
#     )

#     scheduler_plugin = LRFinderSchedulerPlugin(
#         scheduler=scheduler, reset_scheduler=True, step_granularity="iteration"
#     )

#     plugins = [scheduler_plugin]
#     # Continual learning strategy
#     cl_strategy = Naive(
#         model,
#         optimizer,
#         criterion,
#         train_mb_size=32,
#         train_epochs=EPOCHS,
#         plugins=plugins,
#         eval_mb_size=32,
#         device=device,
#     )

#     # train and test loop over the stream of experiences
#     for train_exp in train_stream:
#         cl_strategy.train(train_exp)
