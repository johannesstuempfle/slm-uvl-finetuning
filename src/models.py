import bitsandbytes
import transformers
import torch
import peft
import trl

def get_trainer_supervised(model_name, output_dir, train_dataset, eval_dataset, lora_parameter,
                            epochs=1, batch_size=2, lr_rate=5e-5, lr_scheduler_type="constant"):
    """Creation of supervised trainer object.

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
        trainer (Trainer): actual trainer object
    """
    # quantization config
    quantization_config = transformers.BitsAndBytesConfig(load_in_8bit=True)

    # model and tokenizer
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", 
                                                 quantization_config=quantization_config, 
                                                 trust_remote_code=True)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
  
    # ensure padding
    model.config.pad_token_id = tokenizer.pad_token_id


    # get modules for lora 
    target_modules_names = find_all_linear_names(model)
    print("Modules that will be used for LoRA: ")
    print(target_modules_names)

    # Set up lora config
    peft_config = peft.LoraConfig(lora_alpha=lora_parameter["alpha"], lora_dropout=lora_parameter["dropout"], 
                             r=lora_parameter["rank"], bias="none", task_type="CAUSAL_LM",
                             target_modules=target_modules_names)
    
    # set up sft config
    sft_config = trl.SFTConfig(
            output_dir=output_dir,                          # Directory to save adapters
            num_train_epochs=epochs,                        # Number of training epochs
            per_device_train_batch_size=batch_size,         # Batch size per device during training
            per_device_eval_batch_size=batch_size,          # Batch size per device during evaluation
            logging_strategy="epoch",                       # Log every epoch
            eval_strategy="epoch",                          # Evaluate loss metrics every epoch
            save_strategy="epoch",                          # Save checkpoint every epoch
            learning_rate=lr_rate,                          # Learning rate
            lr_scheduler_type=lr_scheduler_type,            # learning rate scheduler
            max_length=1024,                                # Max sequence length for model and packing of the dataset
            gradient_checkpointing=False,                   # no gradient checkpointing
            packing=False,                                  # no grouping
            optim="adamw_torch_fused",                      # Use fused adamw optimizer
            weight_decay=0.01,                              # Added weight decay for regularization
            load_best_model_at_end=True                     # just load best model in end
        )

    trainer = trl.SFTTrainer(
            model=model,
            processing_class=tokenizer,
            args=sft_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            peft_config=peft_config)

    return trainer

def find_all_linear_names(model):
    """Find Linear modules in given model.

    Parameters:
        model (Transformers Model): model

    Returns:
        lora_module_names (list): list of linear modules
    """
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

def get_trainer_unsupervised(model_name, output_dir, train_dataset, eval_dataset, lora_parameter,
                            epochs=1, batch_size=2, lr=5e-5, lr_scheduler_type="constant"):
    """Creation of supervised trainer object.

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
    # quantization config
    quantization_config = transformers.BitsAndBytesConfig(load_in_8bit=True)

    # model and tokenizer
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", 
                                                 quantization_config=quantization_config, 
                                                 trust_remote_code=True)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # ensure padding
    model.config.pad_token_id = tokenizer.pad_token_id


    # get modules for lora 
    target_modules_names = find_all_linear_names(model)
    print("Modules that will be used for LoRA: ")
    print(target_modules_names)

    # Set up lora config
    peft_config = peft.LoraConfig(lora_alpha=lora_parameter["alpha"], lora_dropout=lora_parameter["dropout"], 
                             r=lora_parameter["rank"], bias="none", task_type="CAUSAL_LM",
                             target_modules=target_modules_names)


    model = peft.get_peft_model(model, peft_config)


    training_args = transformers.TrainingArguments(
        output_dir=output_dir,
        dataloader_drop_last=True,
        num_train_epochs=epochs,                    # epochse
        logging_strategy="epoch",                   # Log every epoch
        eval_strategy="epoch",                      # loss metrics every epoch
        save_strategy="epoch",                      # save every epoch
        per_device_train_batch_size=batch_size,     # batch size during training
        per_device_eval_batch_size=batch_size,      # batch size during evaluation
        learning_rate=lr,                           # learning rate
        lr_scheduler_type=lr_scheduler_type,        # lr sheduler type    
        warmup_steps=30,                            # warm up steps
        gradient_accumulation_steps=1,
        fp16=False,
        bf16=True,
        weight_decay=0.01,
        include_tokens_per_second=True,
        load_best_model_at_end=True
    )
    
    trainer = transformers.Trainer(
        model=model, args=training_args, train_dataset=train_dataset, eval_dataset=eval_dataset, 
        tokenizer=tokenizer
    )

    return trainer