Ensure that any config JSON has at least the following fields:

```
{
    "dataset_name" : "splitcifar100",
    "dataset_path" : "/mnt/datasets/cifar100",
    "dataset_seed": 42,
    
    "epochs": 1,
    "num_experiences": 10,
    "batch_size": 128,
    "learning_rate": 0.001,
    "strategy": "naive",

    "use_wandb": true,
    "wandb_project": "continualTrain",
    "wandb_entity": "nishantaswani",

    "use_multihead": false,
    "model_seeds": [0]
}
```

For strategy/plugin specific arguments:
https://avalanche-api.continualai.org/en/v0.3.1/training.html#training-plugins

All model classes submitted must have a "save_weight" method which takes in a relative path for save location

Set up a .env file

```env
# This is secret and shouldn't be checked into version control
WANDB_API_KEY=$SECRETY_KEY
WANDB_DISABLE_GIT="true" #annoying when false
```