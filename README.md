# Custom LLM Fine-Tuning Project

![Python Version](https://img.shields.io/badge/python-3.12%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A streamlined pipeline for fine-tuning Small Language Models (SLMs) on custom datasets. This project leverages **PEFT (Parameter-Efficient Fine-Tuning)** and **LoRA/QLoRA** techniques to ensure training is efficient and accessible on consumer hardware.

Project dependencies and virtual environments are managed using **[uv](https://github.com/astral-sh/uv)** for fast setup.

## 🚀 Features

* **Efficient Training:** Supports QLoRA (8-bit quantization) to minimize VRAM usage.
* **Modular Architecture:** easily adaptable for models different models.
* **Fast Setup:** Uses `uv` for rapid dependency resolution and environment management.
* **Customizable:** Easy configuration of relevant hyperparameters (Rank, Alpha, Dropout, etc.).

## 🛠 Prerequisites

* Python 3.12
* NVIDIA GPU with CUDA support (Recommended)
* [Optional] Hugging Face API Token (for gated models)

## ⚡ Installation & Setup

This project uses **uv** instead of standard pip/venv for better performance.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/](https://github.com/)[YOUR_USERNAME]/[REPO_NAME].git
    cd [REPO_NAME]
    ```

2.  **Install uv (if not already installed):**
    ```bash
    # macOS/Linux
    curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh

    # Windows
    powershell -c "irm [https://astral.sh/uv/install.ps1](https://astral.sh/uv/install.ps1) | iex"
    ```

3.  **Create the virtual environment:**
    ```bash
    uv venv
    ```

4.  **Activate the environment:**
    * *Linux/macOS:*
        ```bash
        source .venv/bin/activate
        ```
    * *Windows:*
        ```bash
        .venv\Scripts\activate
        ```

5.  **Install dependencies:**
    ```bash
    uv sync
    ```

## 🏃‍♂️ Execution

We use **Gin Config** for parameter management and **uv** for seamless execution.

### 1. Configuration
Before running the training, open the `config.gin` file and adjust the parameters to your needs (e.g., model name, hyperparameters, data paths).

### 2. Run with uv
You don't need to manually activate the virtual environment. Simply use `uv run` to execute the main script. This ensures the script runs with the correct dependencies and python version defined in your project.

```bash
uv run main.py
```

## 📂 Project Structure

```text
.
├── savings/                # Training runs
├── uvl-repos/              # folder for/with uvl data
├── src/                    # Source code
├── main.py                 # Main script
├── train.py                # training functions
├── util.py                 # Utility functions
├── models.py               # Model/Training functions
├── dataset.py              # Dataset functions
├── evaluate.py             # Evaluation functions
├── pyproject.toml          # Project dependencies
├── .gitignore
└── README.md
```