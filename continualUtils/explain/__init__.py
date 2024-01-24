from continualUtils.explain.harmonizer_plugin import *
from continualUtils.explain.lwm_plugin import *
from continualUtils.explain.sgt_plugin import *

# This is the logic/hierarchy for the strategy
# train
#     before_training

#     before_train_dataset_adaptation
#     train_dataset_adaptation
#     after_train_dataset_adaptation
#     make_train_dataloader
#     model_adaptation
#     make_optimizer
#     before_training_exp  # for each exp
#         before_training_epoch  # for each epoch
#             before_training_iteration  # for each iteration
#                 before_forward
#                 after_forward
#                 before_backward
#                 after_backward
#             after_training_iteration
#             before_update
#             after_update
#         after_training_epoch
#     after_training_exp
#     after_training

# eval
#     before_eval
#     before_eval_dataset_adaptation
#     eval_dataset_adaptation
#     after_eval_dataset_adaptation
#     make_eval_dataloader
#     model_adaptation
#     before_eval_exp  # for each exp
#         eval_epoch  # we have a single epoch in evaluation mode
#             before_eval_iteration  # for each iteration
#                 before_eval_forward
#                 after_eval_forward
#             after_eval_iteration
#     after_eval_exp
#     after_eval
