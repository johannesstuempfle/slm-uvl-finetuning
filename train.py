from models.models import get_trainer_supervised, get_trainer_unsupervised, merge_models
from input_pipeline.datasets import get_supervised_datasets, get_unsupervised_datasets
from utils import create_output_dir

DATASET_SUPERVISED = "/home/florian/FA-3908Schatz/input_pipeline/uvl-repos/uvl_dataset-2.json"
GIT_URLS = ["https://github.com/Universal-Variability-Language/uvl-models.git",
            "https://github.tik.uni-stuttgart.de/st191622/uvl_data.git"]
REPO_DIR = "./FA-3908Schatz/input_pipeline/uvl-repos"

MODEL = "codellama/CodeLlama-7b-Instruct-hf"#"meta-llama/CodeLlama-7b-hf"# "google/gemma-3-270m-it" / "codellama/CodeLlama-7b-Instruct-hf" 

# Training arguments
TRAIN_EVAL_SPLIT = 0.8
BATCH_SIZE=1                    # batch_size
EPOCHS=10
LR=1e-5                          # learning_rate
LR_SCHEDULER_TYPE="constant"       # lr_scheduler_type
CHANGE_EMBEDDINGS=True          # also change embedding learning
OUTPUT_DIR=f"/home/florian/FA-3908Schatz/models/model_savings"

SYSTEM_PROMPT = "Translate the following text to uvl-Code: "


def train(supervised=False, merge_model=False):
    if supervised:

        train_dataset, eval_dataset = get_supervised_datasets(DATASET_SUPERVISED, SYSTEM_PROMPT, TRAIN_EVAL_SPLIT)

        if "Llama" in MODEL:
            load_in_4bit = True
        else:
            load_in_4bit = False

        trainer = get_trainer_supervised(model_name=MODEL, output_dir=OUTPUT_DIR, train_dataset=train_dataset, 
                                         eval_dataset=eval_dataset, load_in_4bit=load_in_4bit, use_LoRA=True, epochs=EPOCHS,
                                         batch_size=BATCH_SIZE, lr_rate=LR, lr_scheduler_type=LR_SCHEDULER_TYPE,
                                         change_embedding=CHANGE_EMBEDDINGS)

    else: # unsupervised
        train_dataset, eval_dataset = get_unsupervised_datasets(git_urls=GIT_URLS, repo_dir=REPO_DIR, model_name=MODEL, 
                                                                seq_length=512, train_eval_split=0.8)


        trainer = get_trainer_unsupervised(model_name=MODEL, output_dir=OUTPUT_DIR, train_dataset=train_dataset,
                                           eval_dataset=eval_dataset, load_in_4bit=False, use_LoRA=True, epochs=EPOCHS,
                                           batch_size=BATCH_SIZE, lr=LR, lr_scheduler_type=LR_SCHEDULER_TYPE,
                                           change_embedding=CHANGE_EMBEDDINGS)

    
    print("Training...")
    trainer.train()

    trainer.save_model(create_output_dir(OUTPUT_DIR, MODEL))

    if merge_model:
        merge_models(model_name=MODEL, output_dir=OUTPUT_DIR)