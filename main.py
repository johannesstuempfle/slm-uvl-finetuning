from models.models import get_pipeline
from huggingface_hub import login
from peft import PeftModel
from transformers import AutoTokenizer
import sys
import os
from utils import create_output_dir
from tune import tune
import gin

from train import train

HUG_TOKEN = "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"
login(HUG_TOKEN)

TRAIN = True
TUNE = False

MODEL = "google/gemma-3-270m-it"  #"meta-llama/CodeLlama-7b-hf"# "google/gemma-3-270m-it" / "codellama/CodeLlama-7b-Instruct-hf" 

if __name__ == "__main__":
    gin.parse_config_files_and_bindings(['config.gin'],[])
    if TRAIN:
        train(model_name=MODEL)
    elif TUNE:
        tune()
    else:

        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            pipeline
        )
        MODEL = "codellama/CodeLlama-7b-Instruct-hf" # "codellama/CodeLlama-7b-Instruct-hf", "google/gemma-3-270m-it"
        OUTPUT_DIR=f"/home/florian/FA-3908Schatz/models/model_savings"

        model = AutoModelForCausalLM.from_pretrained(MODEL, device_map="auto")
        model = PeftModel.from_pretrained(model=model, model_id=os.path.join(create_output_dir(OUTPUT_DIR, MODEL)), device_map="auto")
        model = model.merge_and_unload()
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(create_output_dir(OUTPUT_DIR, MODEL)))

        tokenizer.chat_template = """{{ bos_token }}
{%- if messages[0]['role'] == 'system' -%}
    {%- if messages[0]['content'] is string -%}
        {%- set first_user_prefix = messages[0]['content'] + '
' -%}
    {%- else -%}
        {%- set first_user_prefix = messages[0]['content'][0]['text'] + '
' -%}
    {%- endif -%}
    {%- set loop_messages = messages[1:] -%}
{%- else -%}
    {%- set first_user_prefix = "" -%}
    {%- set loop_messages = messages -%}
{%- endif -%}
{%- for message in loop_messages -%}
    {%- if (message['role'] == 'user') != (loop.index0 % 2 == 0) -%}
        {{ raise_exception("Conversation roles must alternate user/assistant/user/assistant/...") }}
    {%- endif -%}
    {%- if (message['role'] == 'assistant') -%}
        {%- set role = "model" -%}
    {%- else -%}
        {%- set role = message['role'] -%}
    {%- endif -%}
    {{ '<start_of_turn>' + role + '
' + (first_user_prefix if loop.first else "") }}
    {%- if message['content'] is string -%}
        {{ message['content'] | trim }}
    {%- elif message['content'] is iterable -%}
        {%- for item in message['content'] -%}
            {%- if item['type'] == 'image' -%}
                {{ '<start_of_image>' }}
            {%- elif item['type'] == 'text' -%}
                {{ item['text'] | trim }}
            {%- endif -%}
        {%- endfor -%}
    {%- else -%}
        {{ raise_exception("Invalid content type") }}
    {%- endif -%}
    {{ '<end_of_turn>
' }}
{%- endfor -%}
{%- if add_generation_prompt -%}
    {{'<start_of_turn>model
'}}
{%- endif -%}"""

        pipeline = get_pipeline(task="text-generation", model=model, tokenizer=tokenizer)


        prompt = "This is an online shop where you can buy clothes. You'll find a wide range of clothing items available. For instance, we offer various types of shirts, pants, and dresses, each in multiple sizes and colors. When it comes to paying for your order, you have several options. You can use a credit card, a debit card, or even pay with cash upon delivery."
        prompt_messages = [
            {"role": "system", "content": "Translate this text to uvl-Code with correct syntax: "},
            {"role": "user", "content": prompt}
        ]

        # Apply the chat template. This will format the messages correctly for the model.
        prompt = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)

        # Generate the output
        output = pipeline(prompt, max_new_tokens=1024)
        model_output_only = output[0]['generated_text'][len(prompt):].strip()

        #print(f"\nDataset text: {user_message_content}")
        #print(f"\nDataset emoji: {dataset_emoji}")
        print(f"\nModel generated output: {model_output_only}")
        print(f"\nModel generated output: {output}")
        """
        sequences = pipeline(
            prompt,
            do_sample=True,
            max_new_tokens=100,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            num_return_sequences=1,
        )
        print(sequences)"""