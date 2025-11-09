from models.models import get_trainer_supervised, get_trainer_unsupervised, merge_models
from input_pipeline.datasets import get_supervised_datasets, get_unsupervised_datasets
from utils import create_output_dir
import gin

DATASET_SUPERVISED = "/home/florian/FA-3908Schatz/input_pipeline/uvl-repos/uvl_dataset-2.json"
GIT_URLS = ["https://github.com/Universal-Variability-Language/uvl-models.git",
            "https://github.tik.uni-stuttgart.de/st191622/uvl_data.git"]
REPO_DIR = "./FA-3908Schatz/input_pipeline/uvl-repos"

USE_LORA = True
USE_PROMPT = False
USE_LAYER_NORM = False

@gin.configurable
def train(model_name, supervised, merge_model, train_eval_split, batch_size, lr, lr_sheduler, 
          epochs, change_embeddings, output_dir, lora_params):
    if supervised:

        train_dataset, eval_dataset = get_supervised_datasets(file=DATASET_SUPERVISED, train_eval_split=train_eval_split)

        if "Llama" in model_name:
            load_in_4bit = True
        else:
            load_in_4bit = False

        trainer = get_trainer_supervised(model_name=model_name, output_dir=output_dir, train_dataset=train_dataset, 
                                         eval_dataset=eval_dataset, load_in_4bit=load_in_4bit, 
                                         use_LoRA=USE_LORA, use_prompt=USE_PROMPT, use_layer_norm=USE_LAYER_NORM, epochs=epochs,
                                         batch_size=batch_size, lr_rate=lr, lr_scheduler_type=lr_sheduler,
                                         change_embedding=change_embeddings, lora_alpha=lora_params["alpha"], 
                                         lora_dropout=lora_params["dropout"], lora_r=lora_params["rank"])

    else: # unsupervised
        train_dataset, eval_dataset = get_unsupervised_datasets(git_urls=GIT_URLS, repo_dir=REPO_DIR, model_name=model_name, 
                                                                seq_length=512, train_eval_split=train_eval_split)


        trainer = get_trainer_unsupervised(model_name=model_name, output_dir=output_dir, train_dataset=train_dataset,
                                           eval_dataset=eval_dataset, load_in_4bit=False, 
                                           use_LoRA=USE_LORA, use_layer_norm=USE_LAYER_NORM, epochs=epochs,
                                           batch_size=batch_size, lr=lr, lr_scheduler_type=lr_sheduler,
                                           change_embedding=change_embeddings, lora_alpha=lora_params["alpha"], 
                                         lora_dropout=lora_params["dropout"], lora_r=lora_params["rank"])

    
    print("Training...")
    trainer.train()

    trainer.save_model(create_output_dir(output_dir, model_name))

    if merge_model:
        merge_models(model_name=model_name, output_dir=output_dir)