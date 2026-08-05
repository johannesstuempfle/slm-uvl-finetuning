# Fine-Tuning Small Language Models For Syntactically Valid Universal Variability Language Generation

[![Workshop: GAIV 2026](https://img.shields.io/badge/GAIV-2026-1f6feb)](https://sites.google.com/view/gaiv-2026)
[![Conference: VARIABILITY 2026](https://img.shields.io/badge/VARIABILITY-2026-0a7d5c)](https://conf.researchr.org/home/variability-2026)
[![Paper: accepted](https://img.shields.io/badge/paper-accepted-brightgreen)](https://sites.google.com/view/gaiv-2026)
<!-- Replace once the proceedings are published:
[![DOI](https://img.shields.io/badge/DOI-10.XXXX%2FXXXXXX-blue)](https://doi.org/10.XXXX/XXXXXX) -->
![DOI](https://img.shields.io/badge/DOI-pending-lightgrey)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/downloads/release/python-3120/)

Replication package for the paper *"Fine-Tuning Small Language Models For Syntactically Valid Universal Variability Language Generation"*, accepted at the [1st International Workshop on Generative AI and Variability (GAIV 2026)](https://sites.google.com/view/gaiv-2026), co-located with [VARIABILITY 2026](https://conf.researchr.org/home/variability-2026), September 29 – October 2, 2026, Limassol, Cyprus.

A pipeline for fine-tuning Small Language Models (SLMs) to generate syntactically and semantically correct UVL from natural language descriptions. Training uses **LoRA** via PEFT with optional 8-bit quantization. Dependencies are managed with **[uv](https://github.com/astral-sh/uv)**.

## Prerequisites

- Python 3.12
- NVIDIA GPU with CUDA support
- Hugging Face account (token required for some models)

## Installation

1. **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <project-directory>
    ```

2. **Install uv** (if not already installed):
    ```bash
    # macOS/Linux
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

3. **Install dependencies:**
    ```bash
    uv sync
    ```

## Usage

### Fine-tune and evaluate

Configure the run in `config.gin` (model name, hyperparameters, dataset paths), then:

```bash
uv run main.py
```

Set `finetuning = False` in `config.gin` to skip training and evaluate the base model directly.

For the full UVL training data, download from [UVL-Hub](https://www.uvlhub.io/) into `uvl-repos/`, or add a git URL to `get_unsupervised_datasets.git_urls` in `config.gin`.

Results are saved to `savings/<ModelName>_<timestamp>/training_results.json`.

### Evaluate a plain base model (no fine-tuning)

```bash
uv run baseline.py Qwen/Qwen2.5-7B-Instruct
uv run baseline.py --load-in-8bit google/gemma-3-4b-it
uv run baseline.py --output-dir results/Baseline/my-run Qwen/Qwen2.5-0.5B-Instruct
```

Results are saved to `results/Baseline/<model-name>/full_results.json` by default.

### Evaluation utilities

Re-score existing outputs without re-running inference, or compare individual UVL files:

```bash
# Re-evaluate a single model's outputs
uv run src/evaluate.py reeval results/Supervised/gemma-3-1b-it/

# Re-evaluate all results recursively
uv run src/evaluate.py reeval results/

# Compare two UVL files directly
uv run src/evaluate.py compare generated.uvl ground_truth.uvl

# Batch-compare a directory of generated files against ground truth
uv run src/evaluate.py batch generated_uvl/ uvl-repos/ground_truth/
```

## Results

Results for all evaluated models are in the `results/` directory. The table below reports Parse Rate (fraction of syntactically valid outputs), Feature F1 (feature vocabulary recovery against ground truth), and Semantic F1 (configuration-space agreement, conditional on parseable outputs).

| Model | Mode | Score | Feat. F1 | Sem. F1 |
| :--- | :---: | :---: | :---: | :---: |
| All models | Baseline | 0% | 0% | — |
| Gemma 270m | Supervised | 17% | 51% | 21% |
| | Unsupervised | 0% | 0% | — |
| Gemma 1b | Supervised | 33% | 62% | 100%† |
| | Unsupervised | 0% | 0% | — |
| Gemma 4b | Supervised | 42% | 76% | 38% |
| | Unsupervised | 0% | 0% | — |
| **Qwen2.5-Coder 0.5b** | Supervised | **67%** | 69% | 46% |
| | Unsupervised | 0% | 0% | — |
| Qwen2.5-Coder 1.5b | Supervised | 50% | 73% | 43% |
| | Unsupervised | 0% | 42% | — |
| Qwen2.5-Coder 3b | Supervised | 50% | 77% | 59% |
| | Unsupervised | 0% | 81% | — |
| Qwen2.5-Coder 7b | Supervised | 33% | 68% | 82% |
| | Unsupervised | 8% | 70% | 9%† |
| Qwen2.5 0.5b | Supervised | 33% | 67% | 18% |
| | Unsupervised | 0% | 0% | — |
| Qwen2.5 1.5b | Supervised | 25% | 70% | 89%† |
| | Unsupervised | 0% | — | — |
| Qwen2.5 3b | Supervised | 50% | 73% | 63% |
| | Unsupervised | 0% | 70% | — |
| Qwen2.5 7b | Supervised | 33% | 77% | 81% |
| | Unsupervised | 0% | 82% | — |

† Semantic F1 is averaged over a small number of parseable outputs (1–3 samples); interpret with caution.

### Why baseline models fail

The outputs found in `results/Baseline/` show that even frontier open-weight models (Qwen3.6-27B, gemma-4-31B-it) still struggle with UVL syntax, since it is a niche domain-specific language that is barely represented in pre-training corpora. Qwen3.6-27B (no fine-tuning) produces this for the "car with engine, assistance systems, and colour" prompt:

```
feature Car
root Car
feature Engine
mandatory Engine
feature Diesel
feature Gasoline
feature Electric
alternative Engine
  Diesel
  Gasoline
  Electric
end
feature AssistanceSystems
optional AssistanceSystems
feature LaneAssist
feature ESP
or AssistanceSystems
  LaneAssist
  ESP
end
feature Color
optional Color
```

This is plausible-looking feature-modelling syntax, but it is not valid UVL at all.The model is reproducing patterns from other feature-modelling languages and DSLs it has seen during pre-training rather than UVL specifically, confirming that fine-tuning on UVL data is necessary even for very capable models.

For comparison, the supervised fine-tuned Qwen2.5-Coder-0.5B-Instruct produces this for the same car prompt:

```
features
    Car
        mandatory
            Engine
                alternative
                    Diesel
                    Gasoline
                    Electric
            OptionalAssistances
                alternative
                    LaneAssist
                    ESP
            CarColor
                alternative
                    Red
                    Green
                    Blue
```

This is syntactically valid UVL with a single, well-formed `features` block and correctly nested group keywords, despite the model being 60x smaller than Qwen3.6-27B. A small amount of supervised fine-tuning is more effective at teaching UVL syntax than raw model scale.

## Project Structure

```text
.
├── main.py                 # Pipeline entry point (fine-tune + evaluate)
├── baseline.py             # Evaluate any plain HuggingFace model without fine-tuning
├── config.gin              # All hyperparameters and run configuration
├── src/
│   ├── evaluate.py         # Unified evaluation module (metrics, reeval CLI, UVL pair comparison)
│   ├── dataset.py          # Dataset loading (supervised and unsupervised modes)
│   ├── models.py           # Model and trainer construction (LoRA, quantization)
│   ├── train.py            # Training wrapper
│   └── util.py             # Output directory creation, git repo cloning
├── uvl-repos/
│   ├── uvl_dataset.json    # Supervised training data
│   ├── test_dataset.json   # Evaluation test set
│   └── ground_truth/       # Ground-truth UVL files (auto-generated on first eval run)
├── results/                # Evaluation results per model and training mode
├── savings/                # Fine-tuned LoRA adapters and training results
├── pyproject.toml
└── uv.lock
```

## Citation

The paper is accepted at GAIV 2026 but not yet presented or published, so no DOI is available yet. Until the proceedings appear, please cite:

```bibtex
@inproceedings{stuempfle2026uvl,
  author    = {St{\"u}mpfle, Johannes},
  title     = {Fine-Tuning Small Language Models For Syntactically Valid Universal Variability Language Generation},
  booktitle = {Proceedings of the 1st International Workshop on Generative AI and Variability (GAIV)},
  year      = {2026},
  note      = {To appear. Co-located with VARIABILITY 2026, Limassol, Cyprus},
}
```

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
