from models.models import get_pipeline
from huggingface_hub import login
from peft import PeftModel
from transformers import AutoTokenizer
import sys
import os
from utils import create_output_dir
from tune import tune
import gin
from evaluation import evaluate

from train import train

HUG_TOKEN = "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"
login(HUG_TOKEN)

TRAIN = False
TUNE = False

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

        print(result[0]["result"])