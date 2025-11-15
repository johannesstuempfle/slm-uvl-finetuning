from huggingface_hub import login
import sys
import os
from tune import tune
import gin
from evaluation import evaluate

from train import train

HUG_TOKEN = "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"
login(HUG_TOKEN)

TRAIN = False
TUNE = True

MODEL = "google/gemma-3-270m-it"  #"meta-llama/CodeLlama-7b-hf"# "google/gemma-3-270m-it" / "codellama/CodeLlama-7b-Instruct-hf" 

if __name__ == "__main__":
    gin.parse_config_files_and_bindings(['config.gin'],[])
    if TRAIN:
        train(model_name=MODEL)
    elif TUNE:
        tune()
    else:
        # do evaluation with test dataset here
        result = evaluate(MODEL)

        print(result)