from peft import LoraConfig, PromptTuningConfig, get_peft_model, prepare_model_for_kbit_training, PromptTuningInit, LNTuningConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
    pipeline
)
import torch
import bitsandbytes
import os
from utils import create_output_dir
from trl import SFTConfig, SFTTrainer
from peft import PeftModel

def merge_models(model_name, output_dir):
    model_dir = create_output_dir(dir=output_dir, model_name=model_name)

    # Load base model and tokenizer
    base_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # Load and merge the PEFT adapters onto the base model
    model = PeftModel.from_pretrained(base_model, model_dir)
    model = model.merge_and_unload()

    # Save the merged model and its tokenizer
    model.save_pretrained(os.path.join(model_dir, 'merged'))
    tokenizer.save_pretrained(os.path.join(model_dir, 'merged'))

def get_trainer_supervised(model_name, output_dir, train_dataset, eval_dataset, load_in_4bit=False,
                            use_LoRA=True, use_prompt=False, use_layer_norm=False,
                            lora_alpha=32, lora_dropout=0.0, lora_r=8, epochs=1, batch_size=2, lr_rate=5e-5, 
                            lr_scheduler_type="constant", change_embedding=False):
    def create_SFT_config(output_dir, epochs, batch_size, lr_rate, lr_scheduler_type):
        args = SFTConfig(
            output_dir=output_dir,                          # Directory to save adapters
            num_train_epochs=epochs,                               # Number of training epochs
            per_device_train_batch_size=batch_size,                    # Batch size per device during training
            per_device_eval_batch_size=batch_size,
            logging_strategy="epoch",                         # Log every epoch
            eval_strategy="epoch",                            # Evaluate loss metrics every epoch
            save_strategy="epoch",                            # Save checkpoint every epoch
            learning_rate=lr_rate,                               # Learning rate,
            lr_scheduler_type=lr_scheduler_type,                     # Use constant learning rate scheduler
            max_length=512,                                   # Max sequence length for model and packing of the dataset
            gradient_checkpointing=False,                     # Use gradient checkpointing to save memory
            packing=False,                                    # Groups multiple samples in the dataset into a single sequence
            optim="adamw_torch_fused",                        # Use fused adamw optimizer
            weight_decay=0.01,                                # Added weight decay for regularization
            load_best_model_at_end=True
        )
        return args

    output_dir = create_output_dir(output_dir, model_name)

    # create bitandbytes config
    bnb_config = get_bnb_config(load_in_4bit=load_in_4bit)

    # gather base model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, 
                                                      device_map="auto", attn_implementation='eager')
  
    model.config.pad_token_id = tokenizer.pad_token_id

    if use_LoRA:
    # gather modules that are adaptable by lora
        target_modules_names = find_all_linear_names(model)
        print("Modules that will be used for LoRA: ")
        print(target_modules_names)

        # Set up lora config
        peft_config = create_lora_config(alpha=lora_alpha, dropout=lora_dropout, 
                                        r=lora_r, target_modules_names=target_modules_names)
    
    elif use_prompt:
        peft_config = create_prompt_config(model_name=model_name)

    elif use_layer_norm:
        peft_config = create_layer_norm_config()
    
    # set up sft config
    sft_config = create_SFT_config(output_dir=output_dir, epochs=epochs, batch_size=batch_size,
                                   lr_rate=lr_rate, lr_scheduler_type=lr_scheduler_type)

    # Train and save the LoRA adapters
    if use_LoRA or use_layer_norm or use_prompt:
        trainer = SFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            peft_config=peft_config,
        )
    else:
        trainer = SFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

    return trainer

def get_trainer_unsupervised(model_name, output_dir, train_dataset, eval_dataset, load_in_4bit=False, 
                            use_LoRA=True, use_layer_norm=False,
                            lora_alpha=32, lora_dropout=0.0, lora_r=8, epochs=1, batch_size=2, lr=5e-5, 
                            lr_scheduler_type="constant", change_embedding=False):
    
    output_dir = create_output_dir(output_dir, model_name)

    bnb_config = get_bnb_config(load_in_4bit=load_in_4bit)

    model = get_auto_model(model_name=model_name, bnb_config=bnb_config)

    if use_LoRA:
        target_modules_names = find_all_linear_names(model)
        print("Modules that will be used for LoRA: ")
        print(target_modules_names)

        # Set up lora
        peft_config = create_lora_config(alpha=lora_alpha, dropout=lora_dropout, 
                                        r=lora_r, target_modules_names=target_modules_names)

    elif use_layer_norm:
        peft_config = create_layer_norm_config()


    model = get_peft_model(model, peft_config)

    model.print_trainable_parameters()

    if change_embedding:
        model.enable_input_require_grads()

    training_args = create_training_args(output_dir=output_dir, model_name=model_name, epochs=epochs,
                                        eval_steps=500, logging_steps=250, batch_size=batch_size,
                                        lr=lr, lr_sheduler=lr_scheduler_type, warmup=30,
                                        gr_acc_steps=1, decay=0.01)
    
    trainer = Trainer(
        model=model, args=training_args, train_dataset=train_dataset, eval_dataset=eval_dataset, 
        tokenizer=AutoTokenizer.from_pretrained(model_name)
    )

    return trainer

def get_bnb_config(load_in_4bit=False):
    if load_in_4bit:
        # 4-bit quantization
        compute_dtype = getattr(torch, "bfloat16")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
    else:
        # bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        bnb_config = None
    return bnb_config

def get_auto_model(model_name, bnb_config=None):
    if bnb_config is not None:
        model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                attn_implementation='eager'
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                trust_remote_code=True,
                attn_implementation='eager'
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.config.pad_token = tokenizer.eos_token

    return prepare_model_for_kbit_training(model)

def find_all_linear_names(model):
    lora_module_names = set()
    for name, module in model.named_modules():
        if (isinstance(module, bitsandbytes.nn.Linear4bit) or 
                isinstance(module, bitsandbytes.nn.Linear8bitLt) or
                isinstance(module, torch.nn.modules.linear.Linear)):
            names = name.split(".")
            # model-specific
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)

def create_lora_config(alpha, dropout, r, target_modules_names, bias="none", task_type="CAUSAL_LM"):
    peft_config = LoraConfig(
        lora_alpha=alpha,
        lora_dropout=dropout,
        r=r,
        bias=bias,
        task_type=task_type,
        target_modules=target_modules_names,
    )
    return peft_config

def create_prompt_config(model_name, task_type="CAUSAL_LM"):
    peft_config = PromptTuningConfig(
        prompt_tuning_init=PromptTuningInit.TEXT,
        num_virtual_tokens=8,
        prompt_tuning_init_text="Translate following text to uvl:",
        task_type=task_type,
        tokenizer_name_or_path=model_name
    )
    return peft_config

def create_layer_norm_config(task_type="CAUSAL_LM"):
    peft_config = LNTuningConfig(
        task_type=task_type
    )
    return peft_config

def create_training_args(output_dir, model_name, epochs, eval_steps, logging_steps, 
                         batch_size, lr, lr_sheduler, warmup, gr_acc_steps, decay):
    def create_output_dir(dir, model_name):
        output_dir = os.path.join(dir, model_name.split('/')[-1])
        return output_dir
    training_args = TrainingArguments(
        output_dir=create_output_dir(output_dir, model_name),
        dataloader_drop_last=True,
        num_train_epochs=epochs,
        logging_strategy="epoch",                         # Log every epoch
        eval_strategy="epoch",                            # Evaluate loss metrics every epoch
        save_strategy="epoch",  
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        lr_scheduler_type=lr_sheduler,
        warmup_steps=warmup,
        gradient_accumulation_steps=gr_acc_steps,
        fp16=False,
        bf16=True,
        weight_decay=decay,
        push_to_hub=False,
        include_tokens_per_second=True,
        load_best_model_at_end=True
    )
    return training_args

def get_pipeline(task, model, tokenizer):
    pipe = pipeline(
            task,
            model=model,
            tokenizer = tokenizer,
            dtype=torch.bfloat16,
            device_map="auto"
        )
    return pipe