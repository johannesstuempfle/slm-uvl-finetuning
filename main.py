from input_pipeline.datasets import get_datasets
from models.models import get_model, create_training_args, get_pipeline
from huggingface_hub import login
from transformers import Trainer
import sys
import torch

HUG_TOKEN = "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"
login(HUG_TOKEN)

GIT_URLS = ["https://github.com/Universal-Variability-Language/uvl-models.git",
            "https://github.tik.uni-stuttgart.de/st191622/uvl_data.git"]
REPO_DIR = "./FA-3908Schatz/input_pipeline/uvl-repos"

MODEL = "google/gemma-3-270m" # "meta-llama/CodeLlama-7b-hf" / "codellama/CodeLlama-7b-Instruct-hf" 
SEQ_LENGTH = 512

TRAIN = True
# Training arguments
TRAIN_EVAL_SPLIT = 0.9
MAX_STEPS=200                   # max_steps
BATCH_SIZE=2                    # batch_size
GR_ACC_STEPS=1                   # gradient_accumulation_steps
LR=5e-4                          # learning_rate
LR_SCHEDULER_TYPE="cosine"       # lr_scheduler_type
WEIGHT_DECAY=0.01                # weight_decay
NUM_WARMUP_STEPS=30              # num_warmup_steps
EVAL_FREQ=100                    # eval_freq
SAVE_FREQ=100                    # save_freq
LOG_FREQ=100                      # log_freq
CHANGE_EMBEDDINGS=False          # also change embedding learning
OUTPUT_DIR=f"/home/florian/FA-3908Schatz/models/model"

# LORA
LORA_R=8                         # lora_r
LORA_ALPHA=32                    # lora_alpha
LORA_DROPOUT=0.0                 # lora_dropout

if __name__ == "__main__":
    if TRAIN:

        train_dataset, eval_dataset = get_datasets(git_urls=GIT_URLS, repo_dir=REPO_DIR, model_name=MODEL, 
                                                   seq_length=SEQ_LENGTH, train_eval_split=TRAIN_EVAL_SPLIT)
       
        model = get_model(model_name=MODEL, load_in_4bit=False, use_LoRA=True)
        
        if CHANGE_EMBEDDINGS:
            model.enable_input_require_grads()

        model.print_trainable_parameters()

        # train_dataset.start_iteration = 0


        training_args = create_training_args(output_dir=OUTPUT_DIR, max_steps=MAX_STEPS, eval_steps=EVAL_FREQ,
                                            save_steps=SAVE_FREQ, logging_steps=LOG_FREQ, batch_size=BATCH_SIZE,
                                            lr=LR, lr_sheduler=LR_SCHEDULER_TYPE, warmup=NUM_WARMUP_STEPS,
                                            gr_acc_steps=GR_ACC_STEPS, decay=WEIGHT_DECAY)
        
        trainer = Trainer(
            model=model, args=training_args, train_dataset=train_dataset, eval_dataset=eval_dataset
        )

    print("Training...")
    trainer.train()