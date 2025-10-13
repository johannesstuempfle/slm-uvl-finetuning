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

# 4-bit quantization
compute_dtype = getattr(torch, "bfloat16")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=True,
)

def get_auto_model(model):
    model = AutoModelForCausalLM.from_pretrained(
            model,
            # load_in_4bit=True,
            quantization_config=bnb_config,
            device_map="auto",
            #device_map={'':torch.cuda.current_device()},
            use_cache=False,
            trust_remote_code=True,
            attn_implementation='eager'
            # use_flash_attention_2=True,
    )
    return prepare_model_for_kbit_training(model)

def get_auto_tokenizer(model):
    tokenizer = AutoTokenizer.from_pretrained(model)
    return tokenizer

def find_all_linear_names(model):
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, bitsandbytes.nn.Linear4bit):
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

def create_training_args(output_dir, max_steps, eval_steps, save_steps, logging_steps, 
                         batch_size, lr, lr_sheduler, warmup, gr_acc_steps, decay):
    training_args = TrainingArguments(
        output_dir=output_dir,
        dataloader_drop_last=True,
        eval_strategy="steps",
        save_strategy="steps",
        max_steps=max_steps,
        eval_steps=eval_steps,
        save_steps=save_steps,
        logging_steps=logging_steps,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        lr_scheduler_type=lr_sheduler,
        warmup_steps=warmup,
        gradient_accumulation_steps=gr_acc_steps,
        gradient_checkpointing=True,
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