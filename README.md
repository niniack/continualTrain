#### What is continualTrain?

A tool to centralize the exeuction of continual learning models from anywhere for any project, given you have it locally installed and follow the interfacing instructions.

#### What does it contain?

Batteries included: an interface to common strategies and common models, along with a logging module to WandB. If you want to supplement functionality, this should be supported.

#### Good to know

The parent directory of each file provided as an argument to the `train.py` module is mounted. There are three mount: model path, config path, and save path. At the moment, no optional mounts are supported.

All model classes must have a "save_weight" method which takes in a relative path for save location.

#### Base JSON
Ensure that any config JSON has at least the following fields:

```
{
    "dataset_name" : "splitcifar100",
    "dataset_path" : "/mnt/datasets/cifar100",
    "dataset_seed": 42,
    
    "learning_rate": 0.001,
    "weight_decay": 1e-4,
    "epochs": 3,
    "batch_size": 128,
    "momentum": 0.9,
    "num_experiences": 10,
    "strategy": "joint",

    "use_wandb": true,
    "wandb_project": "continualTrain",
    "wandb_entity": "nishantaswani",

    "use_multihead": true,
    "model_seeds": [0,1]
}
```
#### Other JSON Keys
For strategy/plugin specific arguments:
https://avalanche-api.continualai.org/en/v0.3.1/training.html#training-plugins

EWC: `ewc_lambda`
RWALK: `rwalk_lambda, rwalk_alpha, rwalk_delta_t`
MAS: `mas_lambda, mas_alpha`
REPLAY: `replay_mem_size`

#### Other
- 

- Set up a .env file

```env
# This is secret and shouldn't be checked into version control
WANDB_API_KEY=$SECRETY_KEY
WANDB_DISABLE_GIT=true #annoying when false
```
