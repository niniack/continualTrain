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

For strategy/plugin specific arguments:
https://avalanche-api.continualai.org/en/v0.3.1/training.html#training-plugins

All model classes submitted must have a "save_weight" method which takes in a relative path for save location

Set up a .env file

```env
# This is secret and shouldn't be checked into version control
WANDB_API_KEY=$SECRETY_KEY
WANDB_DISABLE_GIT=true #annoying when false
```
