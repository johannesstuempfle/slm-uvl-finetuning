import transformers
import peft
import os
import gin
from flamapy.interfaces.python.flamapy_feature_model import FLAMAFeatureModel
import torch
import re
import json


def parsable(text):
    def save_uvl(filename, text):
        if not filename.endswith(".uvl"):
            dateiname += ".uvl"

        text = text.encode('ascii', 'ignore')

        with open(filename, "wb") as f:
            f.write(text)

    save_uvl("temp.uvl", text)

    try:
        fm = FLAMAFeatureModel("temp.uvl")
        valid = fm.satisfiable()
    except:
        valid = False

    return valid

def evaluate_score(results):
    total = 0
    parsable = 0
    for result in results:
        total += 1
        if result["parsable"] == True:
            parsable +=1
    try:
        return parsable/total
    except:
        return 0

def get_pure_uvl(text):
    pattern = r"'''uvl(.*?)'''"

    # re.DOTALL for newlines
    match = re.search(pattern, text, re.DOTALL)

    if match:
        result = match.group(1).strip()
        return result
    else:
        return text

@gin.configurable
def evaluate(model_name, output_dir, dataset_file, from_finetuned):
    def get_test_dataset(file):
        import json
        with open(file) as json_file:
            data = json.load(json_file)
        return data

    dataset_file = os.path.join(os.path.dirname(__file__), dataset_file)

    if from_finetuned:
        model = transformers.AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
        model = peft.PeftModel.from_pretrained(model=model, model_id=output_dir, device_map="auto")
        model = model.merge_and_unload()
        tokenizer = transformers.AutoTokenizer.from_pretrained(output_dir)
    else: #online model without finetuning
        model = transformers.AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

    pipeline = transformers.pipeline(
            task="text-generation",
            model=model,
            tokenizer = tokenizer,
            dtype=torch.bfloat16,
            device_map="auto"
        )

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

        uvl_code = get_pure_uvl(model_output_only)

        model_valid = parsable(uvl_code)

        result_data.append({"text": text, "output": uvl_code, 
                            "output_raw": model_output_only, 
                            "parsable": model_valid})

    results = {}
    results["score"] = evaluate_score(result_data)
    results["eval_data"] = result_data 

    with open(os.path.join(output_dir, 'training_results.json'), 'w') as f:
        json.dump(results, f)

    return results