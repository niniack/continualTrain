#### What is continualTrain?

A tool to centralize the execution of continual learning models in Avalanche, from anywhere, for any project, given you follow the interfacing instructions.

#### What does it contain?

This repository doesn't claim to do too much aside from run the main training loop and log things. As such, it contains a training loop skeleton + a logging module.

Bring Your Own Batteries (BYOB): continualTrain is an interface that runs the main training loop inside an already defined docker container, along with a comprehensive logging module to WandB. You bring your own model and strategies.

Everything is a plugin. Some are required (e.g. model, strategy, dataset), and others are not. If you don't give continualTrain a plugin, it will use a default plugin.

#### Environment File

Set up a .env file to do WandB logging

```env
# This is secret and shouldn't be checked into version control
WANDB_API_KEY=$SECRETY_KEY
WANDB_DISABLE_GIT=true #annoying when false
```
