from input_pipeline.datasets import get_uvl_list, clone_repo, create_datasets, chars_token_ratio, ConstantLengthDataset
from models.models import get_auto_model, find_all_linear_names, create_lora_config, create_training_args
from huggingface_hub import login
from transformers import Trainer
from peft import get_peft_model

HUG_TOKEN = "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"
login(HUG_TOKEN)

git_url = "https://github.com/Universal-Variability-Language/uvl-models.git"
repo_dir = "./FA-3908Schatz/input_pipeline/uvl-repo"

#MODEL = "google/gemma-3-270m"
MODEL = "meta-llama/CodeLlama-7b-hf"
DATA_COLUMN = "text"
SEQ_LENGTH = 512

# Training arguments
MAX_STEPS=100                   # max_steps
BATCH_SIZE=2                    # batch_size
GR_ACC_STEPS=1                   # gradient_accumulation_steps
LR=5e-4                          # learning_rate
LR_SCHEDULER_TYPE="cosine"       # lr_scheduler_type
WEIGHT_DECAY=0.01                # weight_decay
NUM_WARMUP_STEPS=30              # num_warmup_steps
EVAL_FREQ=100                    # eval_freq
SAVE_FREQ=100                    # save_freq
LOG_FREQ=10                      # log_freq
CHANGE_EMBEDDINGS=False          # also change embedding learning
OUTPUT_DIR=f"/home/florian/FA-3908Schatz/model"

# LORA
LORA_R=8                         # lora_r
LORA_ALPHA=32                    # lora_alpha
LORA_DROPOUT=0.0                 # lora_dropout

if __name__ == "__main__":
    clone_repo(git_url=git_url, repository_dir=repo_dir)
    data_list = get_uvl_list(repo_dir)
    train_dataset, evaluation_dataset = create_datasets(data_list)

    chars_per_token = chars_token_ratio(data=train_dataset, model_name=MODEL, data_column=DATA_COLUMN)
    print(f"The character to token ratio of the dataset is: {chars_per_token:.2f}")

    train_dataset = ConstantLengthDataset(
        model_name=MODEL,
        dataset=train_dataset,
        infinite=True,
        seq_length=SEQ_LENGTH,
        chars_per_token=chars_per_token,
        content_field=DATA_COLUMN,
    )
    eval_dataset = ConstantLengthDataset(
            model_name=MODEL,
            dataset=evaluation_dataset,
            infinite=False,
            seq_length=SEQ_LENGTH,
            chars_per_token=chars_per_token,
            content_field=DATA_COLUMN,
    )
    
    # create quantified model
    model = get_auto_model(model=MODEL)

    print(model)

    # get all modules that are available for lora
    target_modules_names = find_all_linear_names(model)

    print("Modules that will be used for LoRA: ")
    print(target_modules_names)

    # Set up lora
    peft_config = create_lora_config(alpha=LORA_ALPHA, dropout=LORA_DROPOUT, 
                                     r=LORA_R, target_modules_names=target_modules_names)

    model = get_peft_model(model, peft_config)
    
    if CHANGE_EMBEDDINGS:
        model.enable_input_require_grads()

    model.print_trainable_parameters()

    train_dataset.start_iteration = 0


    training_args = create_training_args(output_dir=OUTPUT_DIR, max_steps=MAX_STEPS, eval_steps=EVAL_FREQ,
                                        save_steps=SAVE_FREQ, logging_steps=LOG_FREQ, batch_size=BATCH_SIZE,
                                        lr=LR, lr_sheduler=LR_SCHEDULER_TYPE, warmup=NUM_WARMUP_STEPS,
                                        gr_acc_steps=GR_ACC_STEPS, decay=WEIGHT_DECAY)
    
    trainer = Trainer(
        model=model, args=training_args, train_dataset=train_dataset, eval_dataset=eval_dataset
    )

    print("Training...")
    trainer.train()