import os
from transformers import AutoTokenizer
import torch
from torch.utils.data import Dataset
import random
import gin
import src.util as util
from datasets import load_dataset

PRE_TEXT = "The following given UVL-Code is written with correct syntax: \n\n"

@gin.configurable
def get_dataset(model_name, supervised, train_eval_split):
    """Return train and eval datasets corresponding to inputs.

    Parameters:
        model_name (string): name of model (same as on huggingface)
        supervised (bool): for supervised is True, for unsupervised is False 
        train_eval_split (float): relative split between train and eval dataset

    Returns:
        train_dataset, eval_dataset: corresponding datasets
    """
    if supervised:
        train_dataset, eval_dataset = get_supervised_datasets(train_eval_split=train_eval_split)
    else:
        train_dataset, eval_dataset = get_unsupervised_datasets(model_name=model_name,
                                                                train_eval_split=train_eval_split)
        
    return train_dataset, eval_dataset

@gin.configurable
def get_unsupervised_datasets(model_name, seq_length, git_urls, repo_dir, train_eval_split):
    """Return train and eval datasets for unsupervised variant.

    Parameters:
        model_name (string): name of model (same as on huggingface)
        seq-length (int): limits the length of training samples
        git_urls (array): strimngs of git-urls to fetch uvl data from
        repo_dir (string): relative path to folder for uvl-datat sync
        train_eval_split (float): relative split between train and eval dataset

    Returns:
        train_dataset, eval_dataset: corresponding unsupervised datasets
    """
    repo_dir = os.path.join(os.path.dirname(__file__), repo_dir)

    util.clone_repositories(git_urls=git_urls, repositories_dir=repo_dir)

    # filter files with sequence length above given sequence maximum length
    uvl_file_pathes = get_uvl_list_with_limit(folder_path=repo_dir, token_limit=seq_length, model=model_name)

    # ensure same shuffling for different runs
    random.seed(10)
    random.shuffle(uvl_file_pathes)

    # calculate file split for train/eval
    split_index = int(len(uvl_file_pathes)*train_eval_split)

    train_dataset = UvlCodeDataset(model_name, uvl_file_pathes[:split_index], seq_length)
    eval_dataset = UvlCodeDataset(model_name, uvl_file_pathes[split_index:], seq_length)

    return train_dataset, eval_dataset

def get_uvl_files(folder):
    """Return filepathes for all uvl files in given folder.

    Parameters:
        folder (string): folder with uvl-data

    Returns:
        uvl_files (array): list of filepathes
    """
    uvl_files = []
    for root, folderfolders, files in os.walk(folder):
        for file in files:
            if file.endswith(".uvl"):
                uvl_files.append(os.path.join(root, file))
    return uvl_files

def read_uvl_content(file):
    """Return string content of uvl file.

    Parameters:
        file (string): file-path with uvl-data

    Returns:
        uvl_data (string): uvl file content
    """
    try:
        with open(file, 'r') as f:
            uvl_data = f.read()
            return uvl_data
    except Exception as e:
        print(f"Could not read {file}: {e}")

def get_uvl_list_with_limit(folder_path, token_limit, model):
    """Return array with uvl content filtered by given limit.

    Parameters:
        folder_path (string): folder with uvl-data
        token_limit (int): size limit of files in tokens
        model (string): huggingface model

    Returns:
        content_limit (array): list of filtered file content
    """
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    files = get_uvl_files(folder=folder_path)
    content_limit = []
    for file in files:
        file_content = read_uvl_content(file)
        tokenized_inputs = tokenizer(file_content)["input_ids"]
        if len(tokenized_inputs) < token_limit:
            content_limit.append(file)
    return content_limit

def get_uvl_list(folder_path):
    """Return array with uvl content.

    Parameters:
        folder_path (string): folder with uvl-data

    Returns:
        uvl_conent (array): list of file contents
    """
    uvl_list = get_uvl_files(folder_path)

    uvl_content = []

    for file in uvl_list:
        try:
            with open(file, 'r') as f:
                uvl_content.append(f.read())
        except Exception as e:
            print(f"Could not read {file}: {e}")
    
    return uvl_content

class UvlCodeDataset(Dataset):
    """Dataset-Class for unsupervised data."""
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

        content = PRE_TEXT + "'''uvl\n" + content + "\n'''"

        tokenized_inputs = self.tokenizer(content, padding='max_length', truncation=True, 
                                          max_length=self.max_length, add_special_tokens=True)["input_ids"]
        
        sample = {
                "input_ids": torch.LongTensor(tokenized_inputs),
                "labels": torch.LongTensor(tokenized_inputs),
                }

        return sample

@gin.configurable
def get_supervised_datasets(file, system_prompt, train_eval_split):
    """Provide a dataset with supervised samples.

    Parameters:
        file (string): relative fileparh with given samples
        system_prompt (string): prompt, that is added to samples
        train_eval_split (float): split between traina and eval samples

    Returns:
        train_dataset, eval_dataset (dataset, dataset): datasets with samples
    """
    file = os.path.join(os.path.dirname(__file__), os.pardir, file)

    def translate(sample):
        return {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{sample['text']}"},
                {"role": "assistant", "content": f"{"'''uvl\n" + sample['code'] + "\n'''"}"}
            ]
        }
    dataset = load_dataset("json", data_files=file, field="data", encoding="utf-8", split="train")

    training_dataset = dataset.map(translate, remove_columns=dataset.features.keys())

    training_dataset_splits = training_dataset.train_test_split(test_size=1-train_eval_split, shuffle=True)
    
    # Set training and evaluation datasets
    train_dataset = training_dataset_splits['train']
    eval_dataset = training_dataset_splits['test']

    return train_dataset, eval_dataset
