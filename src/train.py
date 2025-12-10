import gin
import src.models as models

@gin.configurable
def train(model_name, train_dataset, eval_dataset, supervised, batch_size, lr, lr_sheduler, 
          epochs, output_dir, lora_params):
    """Actual training function.

    Parameters:
        model_name (string): name of model (same as on huggingface)
        train_dataset (dataset): training dataset 
        eval_dataset (dataset): eval dataset 
        supervised (bool): switch for supervised and unsupervised training/tuning
        batch_size (int): batch size
        lr (float) : learning rate
        lr_sheduler (string): learning rate sheduler
        epochs (int): num. epochs
        output_dir (string): path for savings
        lora_params (dict): lora paremeters

    Returns:
        -
    """

    if supervised: # supervised
        trainer = models.get_trainer_supervised(model_name=model_name, output_dir=output_dir, train_dataset=train_dataset, 
                                         eval_dataset=eval_dataset, epochs=epochs,
                                         batch_size=batch_size, lr_rate=lr, lr_scheduler_type=lr_sheduler,
                                         lora_parameter=lora_params)

    else: # unsupervised
        trainer = models.get_trainer_unsupervised(model_name=model_name, output_dir=output_dir, train_dataset=train_dataset,
                                           eval_dataset=eval_dataset, epochs=epochs,
                                           batch_size=batch_size, lr=lr, lr_scheduler_type=lr_sheduler,
                                           lora_parameter=lora_params)
    
    print("Training...")
    trainer.train()

    trainer.save_model(output_dir)