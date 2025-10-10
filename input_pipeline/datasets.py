from git import Repo
import os
import sys
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer
import torch
from torch.utils.data import IterableDataset
from torch.utils.data.dataloader import DataLoader
import random
import numpy as np


def clone_repo(git_url, repository_dir):
   if not os.path.exists(repository_dir):
        Repo.clone_from(git_url, repository_dir)

def get_uvl_list(folder_path):
    def get_uvl_files(folder):
        uvl_files = []
        for root, folderfolders, files in os.walk(folder):
            for file in files:
                if file.endswith(".uvl"):
                    uvl_files.append(os.path.join(root, file))
        return uvl_files

    uvl_list = get_uvl_files(folder_path)

    uvl_content = []

    for file in uvl_list:
        try:
            with open(file, 'r') as f:
                uvl_content.append(f.read())
        except Exception as e:
            print(f"Could not read {file}: {e}")

    print("Imported a total of " + str(len(uvl_content)) +
        " files with size of all uvl example data: " +
        str(sys.getsizeof(uvl_content)) + " Bytes")
    
    return uvl_content

def create_datasets(data_list):
    # generate value pairs from different sources
    data = [{"text": a} for a in data_list]
    #example_data.append({"text": website_text})
    #example_data.append([{"text": a} for a in uvl_content])

    # create dataset
    dataset = Dataset.from_list(data)

    validation_data = dataset.take(30)
    train_data = dataset.skip(30)

    return train_data, validation_data

def chars_token_ratio(data, model_name, data_column, nb_examples=400):
    """
    Estimate the average number of characters per token in the dataset.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    total_characters, total_tokens = 0, 0
    for _, example in tqdm(zip(range(nb_examples), iter(data)), total=nb_examples):
        total_characters += len(example[data_column])
        total_tokens += len(tokenizer(example[data_column]).tokens())

    return total_characters / total_tokens

class ConstantLengthDataset(IterableDataset):
    def __init__(
        self,
        model_name,
        dataset,
        infinite=False,
        seq_length=1024,
        num_of_sequences=1024,
        chars_per_token=4.0,
        content_field="text",
        seed=0,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.concat_token_id = self.tokenizer.eos_token_id
        self.dataset = dataset
        self.seq_length = seq_length
        self.infinite = infinite
        self.current_size = 0
        self.max_buffer_size = seq_length * chars_per_token * num_of_sequences
        self.content_field = content_field
        self.seed = seed

    def __iter__(self):
        iterator = iter(self.dataset)
        more_examples = True
        np_rng = np.random.RandomState(seed=self.seed)
        while more_examples:
            buffer, buffer_len = [], 0
            while True:
                if buffer_len >= self.max_buffer_size:
                    break
                try:
                    buffer.append(next(iterator)[self.content_field])
                    buffer_len += len(buffer[-1])
                except StopIteration:
                    if self.infinite:
                        iterator = iter(self.dataset)
                    else:
                        more_examples = False
                        break
            tokenized_inputs = self.tokenizer(buffer, truncation=False)["input_ids"]
            all_token_ids = []

            for tokenized_input in tokenized_inputs:
                all_token_ids.extend(tokenized_input + [self.concat_token_id])

            examples = []
            for i in range(0, len(all_token_ids), self.seq_length):
                input_ids = all_token_ids[i : i + self.seq_length]
                if len(input_ids) == self.seq_length:
                    examples.append(input_ids)
            random.shuffle(examples)
            for example in examples:
                self.current_size += 1
                yield {
                    "input_ids": torch.LongTensor(example),
                    "labels": torch.LongTensor(example),
                }