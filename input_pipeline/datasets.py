from git import Repo
import os
import sys
from tqdm import tqdm
from transformers import AutoTokenizer
import torch
from torch.utils.data import Dataset
import random
import matplotlib.pyplot as plt
import statistics

def get_unsupervised_datasets(git_urls, repo_dir, model_name, seq_length=512, train_eval_split=0.9):
    clone_repositories(git_urls=git_urls, repositories_dir=repo_dir)

    uvl_file_pathes = get_uvl_files(repo_dir)

    random.seed(10)
    random.shuffle(uvl_file_pathes)

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

def get_uvl_lines_of_code(files):
    lines = 0
    for file in files:
        lines += sum(1 for _ in open(file))
    return lines

def get_uvl_lines_of_code_per_file(files):
    lines = []
    for file in files:
        lines.append(sum(1 for _ in open(file)))
    return lines

def get_uvl_list_with_limit(folder_path, token_limit, model):
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
    uvl_list = get_uvl_files(folder_path)

    uvl_content = []

    for file in uvl_list:
        try:
            with open(file, 'r') as f:
                uvl_content.append(f.read())
        except Exception as e:
            print(f"Could not read {file}: {e}")
    
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
                                          max_length=self.max_length, add_special_tokens=True)["input_ids"]
        
        sample = {
                "input_ids": torch.LongTensor(tokenized_inputs),
                "labels": torch.LongTensor(tokenized_inputs),
                }

        return sample

def get_supervised_datasets(file, system_prompt, train_eval_split=0.8):
    from datasets import load_dataset
    def translate(sample):
        return {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{sample['text']}"},
                {"role": "assistant", "content": f"{sample['code']}"}
            ]
        }
    dataset = load_dataset("json", data_files=file, field="data", encoding="utf-8", split="train")

    training_dataset = dataset.map(translate, remove_columns=dataset.features.keys())

    training_dataset_splits = training_dataset.train_test_split(test_size=1-train_eval_split, shuffle=True)
    
    # Set training and evaluation datasets
    train_dataset = training_dataset_splits['train']
    eval_dataset = training_dataset_splits['test']

    return train_dataset, eval_dataset


def plot_lines_of_code_dist(uvl_path_list, scope):
    uvl_lines_per_file = get_uvl_lines_of_code_per_file(uvl_path_list)

    boundary = 2048

    temp = []
    for file_size in uvl_lines_per_file:
        if file_size > boundary:
            file_size = boundary
        temp.append(file_size)
    
    uvl_lines_per_file = temp

    # define window size, output and axes
    fig, ax = plt.subplots(figsize=[8,6])

    # set plot title
    ax.set_title("Lines of Code Distribution (Clipped to " + 
                 str(boundary) + ")")

    # set x-axis name
    ax.set_xlabel("Lines of Code per File")

    # set y-axis name
    ax.set_ylabel("Number of Files")

    # create histogram within output
    N, bins, patches = ax.hist(uvl_lines_per_file, bins=50, color="#2C8BFF") #initial color of all bins

    # Iterate through all histogram elements
    # each element in this interation is one patch on the histogram, where:
    # - bin_size - number of records in current bin
    # - bin - value of current bin (x-axis)
    # - patch - a rectangle, object of class matplotlib.patches.Patch
    # more details on patch properties: [visit this link][1]
    """
    for bin_size, bin, patch in zip(N, bins, patches):
        if bin_size == <some number>:
            patch.set_facecolor("<some color like #FF000>")
            patch.set_label("something")"""
    plt.show()
    plt.savefig('/home/florian/FA-3908Schatz/input_pipeline/figures/line_of_code_'+scope+'.png', bbox_inches='tight')

if __name__ == "__main__":
    
    DATASET_SUPERVISED = "/home/florian/FA-3908Schatz/input_pipeline/uvl-repos/uvl_dataset-2.json"
    GIT_URLS = ["https://github.com/Universal-Variability-Language/uvl-models.git",
                "https://github.tik.uni-stuttgart.de/st191622/uvl_data.git"]
    REPO_DIR = "./FA-3908Schatz/input_pipeline/uvl-repos"
    MODEL = "google/gemma-3-270m-it" #"meta-llama/CodeLlama-7b-hf"# "google/gemma-3-270m-it" / "codellama/CodeLlama-7b-Instruct-hf" 

    # Doing some analysis of dataset
    clone_repositories(git_urls=GIT_URLS, repositories_dir=REPO_DIR)

    uvl_file_pathes = get_uvl_files(REPO_DIR)
    
    print("Statistics for full dataset:")

    print(" Total number of uvl-Files: " + 
          str(len(uvl_file_pathes)))
    
    print(" Total number of uvl lines of code in files: " + 
          str(get_uvl_lines_of_code(uvl_file_pathes)))

    uvl_list = get_uvl_list(REPO_DIR)

    print(" Total size of all uvl example data: " +
        str(sys.getsizeof(uvl_list)) + " Bytes")

    plot_lines_of_code_dist(uvl_file_pathes, "full")

    lines_of_code = get_uvl_lines_of_code_per_file(uvl_file_pathes)
    print(" Mean lines of code: " + str(statistics.mean(lines_of_code)))
    print(" Median lines of code: " + str(statistics.median(lines_of_code)))

    print("Statistics for limited dataset:")

    uvl_file_pathes = get_uvl_list_with_limit(folder_path=REPO_DIR, token_limit=512, model=MODEL)

    print(" Total number of uvl-Files: " + 
        str(len(uvl_file_pathes)))
    
    print(" Total number of uvl lines of code in files: " + 
          str(get_uvl_lines_of_code(uvl_file_pathes)))

    uvl_list = get_uvl_list(REPO_DIR)

    print(" Total size of all uvl example data: " +
        str(sys.getsizeof(uvl_list)) + " Bytes")

    plot_lines_of_code_dist(uvl_file_pathes, "limited")

    lines_of_code = get_uvl_lines_of_code_per_file(uvl_file_pathes)
    print(" Mean lines of code: " + str(statistics.mean(lines_of_code)))
    print(" Median lines of code: " + str(statistics.median(lines_of_code)))
