import os
from datetime import datetime
from git import Repo

def create_output_dir(dir, model_name):
    """Create an output directory path for specific model

    Parameters:
        dir (string): rel. or abs. path for model data
        model_name (string): name of model (same as on huggingface)

    Returns:
        output_dir (string): output directory with model_name and uniqueue timestamp
    """

    # calculate absolute path from relative path
    if not os.path.isabs(dir):
        output_dir = os.path.join(os.path.dirname(__file__), os.pardir, dir)

    # unique folder name by including time
    now = datetime.now()
    now_str = now.strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = model_name.split('/')[-1] + "_" + now_str

    # model specific path
    output_dir = os.path.join(output_dir, folder_name)

    # check, whether folder already exists and create if not
    os.makedirs(output_dir, exist_ok=True)

    return output_dir

def clone_repositories(git_urls, repositories_dir):
    for git_url in git_urls:
        repo_name = git_url.split('.git')[0].split('/')[-1]
        repo_dir = os.path.join(os.path.dirname(__file__), os.pardir, repositories_dir, repo_name)
        if not os.path.exists(repo_dir):
            Repo.clone_from(git_url, repo_dir)