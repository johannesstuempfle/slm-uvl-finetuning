from input_pipeline.datasets import get_uvl_list, clone_repo, create_datasets, chars_token_ratio, ConstantLengthDataset
from huggingface_hub import login

HUG_TOKEN = "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"
login(HUG_TOKEN)

git_url = "https://github.com/Universal-Variability-Language/uvl-models.git"
repo_dir = "./FA-3908Schatz/input_pipeline/uvl-repo"

MODEL = "google/gemma-3-270m"
DATA_COLUMN = "text"
SEQ_LENGTH = 512

if __name__ == "__main__":
    clone_repo(git_url=git_url, repository_dir=repo_dir)
    data_list = get_uvl_list(repo_dir)
    train_dataset, evaluation_dataset = create_datasets(data_list)

    chars_per_token = chars_token_ratio(data=train_dataset, model_name=MODEL, data_column=DATA_COLUMN)
    print(f"The character to token ratio of the dataset is: {chars_per_token:.2f}")

    train_dataset = ConstantLengthDataset(
        model_name=MODEL,
        dataset=train_dataset,
        infinite=True,
        seq_length=SEQ_LENGTH,
        chars_per_token=chars_per_token,
        content_field=DATA_COLUMN,
    )
    eval_dataset = ConstantLengthDataset(
            model_name=MODEL,
            dataset=evaluation_dataset,
            infinite=False,
            seq_length=SEQ_LENGTH,
            chars_per_token=chars_per_token,
            content_field=DATA_COLUMN,
    )