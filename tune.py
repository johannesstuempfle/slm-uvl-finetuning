"""This file iomplements the tuning functionality."""

import wandb
import gin
from train import train
import torch
from evaluation import evaluate

@gin.configurable
def tune(key, sweep_config, parameters_dict, n_runs):
    """Tune models with wandb evaluation.

    Parameters:
        key (string): key for upload in wandb
        sweep_config (dict): sweep config parameters for wandb
        parameters_dict (dict): hyperparameters for tuning
        n_runs (int): number of wandb runs
    """

    # wandb login with given key
    wandb.login(key=key)

    sweep_config['parameters'] = parameters_dict

    # create wandb conf and obj
    sweep_id = wandb.sweep(sweep=sweep_config,
                           project="Forschungsarbeit")

    # actual tuning function
    def tuning():
        """Train one model with tuning parameters."""

        config = {
            'epochs': 1,
            'lr_rate': 0.01,
        }
        if config:
            wandb.init(config=config)

            config = wandb.config

            # Get dataset here
            

            # Create Model here

            # train model here
            train(model_name=config.model_name, supervised=True, merge_model=False, lr=config.lr_rate, 
                  epochs=config.epochs, lora_params=config.lora_params)

            torch.cuda.empty_cache()

            # do evaluation with test dataset here
            eval_result = evaluate(config.model_name)

            wandb.log({"test_results": eval_result})

            torch.cuda.empty_cache()


    # use agent for optimization
    wandb.agent(sweep_id, function=tuning, count=n_runs)

    # clear wandb job
    wandb.finish()