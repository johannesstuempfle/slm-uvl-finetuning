from git import Repo
import os
import sys
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer
import torch
from torch.utils.data import Dataset

def get_datasets(git_urls, repo_dir, model_name, seq_length=512, train_eval_split=0.9):
    clone_repositories(git_urls=git_urls, repositories_dir=repo_dir)

    uvl_file_pathes = get_uvl_files(repo_dir)

    # claculate file split for train/eval
    split_index = int(len(uvl_file_pathes)*train_eval_split)

    train_dataset = UvlCodeDataset(model_name, uvl_file_pathes[:split_index], seq_length)
    eval_dataset = UvlCodeDataset(model_name, uvl_file_pathes[split_index:], seq_length)

    return train_dataset, eval_dataset

def clone_repositories(git_urls, repositories_dir):
   for git_url in git_urls:
    repo_name = git_url.split('.git')[0].split('/')[-1]
    repo_dir = os.path.join(repositories_dir, repo_name)
    if not os.path.exists(repo_dir):
        Repo.clone_from(git_url, repo_dir)

def get_uvl_files(folder):
    uvl_files = []
    for root, folderfolders, files in os.walk(folder):
        for file in files:
            if file.endswith(".uvl"):
                uvl_files.append(os.path.join(root, file))
    return uvl_files

def read_uvl_content(file):
    try:
        with open(file, 'r') as f:
            return f.read()
    except Exception as e:
        print(f"Could not read {file}: {e}")

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

def get_max_file_length(model, files):
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    max_length = 0
    for file in files:
        content = read_uvl_content(file)
        tokenized_inputs = tokenizer(content)["input_ids"]
        if len(tokenized_inputs) > max_length:
            max_length = len(tokenized_inputs)
    return max_length

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

class UvlCodeDataset(Dataset):
    def __init__(self, model_name, uvl_file_pathes, max_length):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.uvl_file_pathes = uvl_file_pathes
        self.max_length = max_length

    def __len__(self):
        return len(self.uvl_file_pathes)

    def __getitem__(self, idx):
        file_path = self.uvl_file_pathes[idx]

        content = read_uvl_content(file=file_path)

        tokenized_inputs = self.tokenizer(content, padding='max_length', truncation=True, 
                                          max_length=self.max_length)["input_ids"]
        
        sample = {
                "input_ids": torch.LongTensor(tokenized_inputs),
                "labels": torch.LongTensor(tokenized_inputs),
                }

        return sample