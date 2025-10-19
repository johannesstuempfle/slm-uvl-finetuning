from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    logging,
    set_seed,
    BitsAndBytesConfig,
    pipeline
)
import torch
import bitsandbytes
import sys
import os

def get_model(model_name, load_in_4bit=False, use_LoRA=True,
              lora_alpha=32, lora_dropout=0.0, lora_r=8, 
              change_embedding=False):

    bnb_config = get_bnb_config(load_in_4bit=load_in_4bit)

    model = get_auto_model(model_name=model_name, bnb_config=bnb_config)

    if use_LoRA:
        target_modules_names = find_all_linear_names(model)
        print("Modules that will be used for LoRA: ")
        print(target_modules_names)

        # Set up lora
        peft_config = create_lora_config(alpha=lora_alpha, dropout=lora_dropout, 
                                        r=lora_r, target_modules_names=target_modules_names)
        model = get_peft_model(model, peft_config)

        model.print_trainable_parameters()

    if change_embedding:
        model.enable_input_require_grads()

    return model

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

def get_auto_model(model_name, bnb_config):
    if bnb_config is not None:
        model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
                use_cache=False,
                trust_remote_code=True,
                attn_implementation='eager'
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                use_cache=False,
                trust_remote_code=True,
                attn_implementation='eager'
        )
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

def create_training_args(output_dir, model_name, epochs, eval_steps, logging_steps, 
                         batch_size, lr, lr_sheduler, warmup, gr_acc_steps, decay):
    def create_output_dir(dir, model_name):
        output_dir = os.path.join(dir, model_name.split('/')[-1])
        return output_dir
    training_args = TrainingArguments(
        output_dir=create_output_dir(output_dir, model_name),
        dataloader_drop_last=True,
        num_train_epochs=epochs,
        eval_strategy="steps",
        save_strategy="epoch",
        eval_steps=eval_steps,
        logging_steps=logging_steps,
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