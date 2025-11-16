from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            pipeline
        )
from peft import PeftModel
import os
from utils import create_output_dir
from models.models import get_pipeline
import gin
from flamapy.interfaces.python.flamapy_feature_model import FLAMAFeatureModel
from flamapy.core.exceptions import FlamaException

CHAT_TEMPLATE = """{{ bos_token }}
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

def parsable(text):
    import io
    from contextlib import redirect_stdout, redirect_stderr

    def save_uvl(filename, text):
        if not filename.endswith(".uvl"):
            dateiname += ".uvl"

        text = text.encode('ascii', 'ignore')

        with open(filename, "wb") as f:
            f.write(text)

    save_uvl("temp.uvl", text)
    
    r = io.StringIO()

    valid = False
    with redirect_stderr(r):
        try:
            fm = FLAMAFeatureModel("temp.uvl")
            valid = fm.satisfiable()
        except FlamaException as e:
            i = 0

    output_err = r.getvalue().split('\n')

    if valid:
        return True, output_err
    else:
        return False, output_err

@gin.configurable
def evaluate(model_name, output_dir, dataset_file, from_finetuned=False):
    def get_test_dataset(file):
        import json
        with open(file) as json_file:
            data = json.load(json_file)
        return data

    output_dir = os.path.join(os.path.abspath(os.getcwd()), output_dir)
    dataset_file = os.path.join(os.path.abspath(os.getcwd()), dataset_file)

    if from_finetuned:
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
        model = PeftModel.from_pretrained(model=model, model_id=os.path.join(create_output_dir(output_dir, model_name)), device_map="auto")
        model = model.merge_and_unload()
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(create_output_dir(output_dir, model_name)))
    else: #online model without finetuning
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_name)


    # tokenizer.chat_template = CHAT_TEMPLATE

    pipeline = get_pipeline(task="text-generation", model=model, tokenizer=tokenizer)

    test_data = get_test_dataset(dataset_file)
    result_data = []
    for sample in test_data["data"]:
        text = sample["text"]
        prompt_messages = [
            {"role": "system", "content": "Can you please translate this text to uvl-Code? Please consider all informations from the text, but make sure, that the correct syntax is used: "},
            {"role": "user", "content": text}
        ]

        # Apply the chat template. This will format the messages correctly for the model.
        prompt = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)

        # Generate the output
        output = pipeline(prompt, max_new_tokens=1024)
        model_output_only = output[0]['generated_text'][len(prompt):].strip()

        model_valid, errors = parsable(model_output_only)

        result_data.append({"text": text, "output": model_output_only, "parsable": model_valid, "errors": errors})

    return result_data