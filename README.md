# Lexical Functions to Improve Lexical Competence

This project investigates whether explicit structural injection of French lexical functions, derived from the Meaning-Text Theory (MTT), improves the lexical competence and general natural language understanding of Large Language Models (LLMs). We focus on fine-tuning and evaluating three architectures: **Mistral-7B**, **Qwen2.5-7B**, and **FlauBERT**.

## Prerequisites

- **Python Version**: Python 3.9 or higher is recommended.
- **Dependencies**: All required packages are listed in `requirements.txt`. Install them using:
  ```bash
  pip install -r requirements.txt
  ```

## Project Structure

The repository is organized as follows:

*   **`data/`**: Contains the structured datasets derived from the French Lexical Network (fr-LN) and the benchmarking patterns used for evaluation.
*   **`eval/`**: Scripts for performing model evaluations.
    *   `COLE/`: Subdirectory for the COLE benchmark repository.
    *   `fl_benchmark.py`: Core logic for testing models on Lexical Function tasks.
    *   `run_eval_fl.sh` & `run_eval_cole.sh`: Utility scripts to run evaluations across all models.
    *   `generate_*_report.py`: Scripts to aggregate raw results into summary reports.
*   **`fine-tuning/`**: Contains scripts for Low-Rank Adaptation (LoRA) fine-tuning.
    *   `save_base_models.py`: Helper script to download and cache base models locally.
    *   `train_*.py`: Model-specific fine-tuning scripts.
    *   `run_training.sh`: Orchestrates the training sessions for different tasks (LF vs. Wikipedia baseline).
*   **`models/`**: Local storage for base models and trained adapters (not included in the repository because they are too large for GitHub; contact us for access).
*   **`report/`**: LaTeX source files and instructions for the final academic report.

## Usage Guide

### 1. Model Preparation
First, ensure the base models are downloaded to the local directory:
```bash
python3 fine-tuning/save_base_models.py
```

### 2. Executing Fine-Tuning
To reproduce the fine-tuning experiments on both the Lexical Function (LF) dataset and the Wikipedia baseline:
```bash
cd fine-tuning
bash run_training.sh
```

### 3. Running Benchmarks
To evaluate the models' performance on specific lexical functions:
```bash
cd eval
bash run_eval_fl.sh
```

To assess general linguistic capabilities using the COLE benchmark:
```bash
cd eval
bash run_eval_cole.sh
```

### 4. Generating Reports
Once the evaluations are complete, you can generate aggregated reports:
```bash
python3 eval/generate_fl_report.py
python3 eval/generate_cole_report.py
```

## Reference
For detailed methodology and experimental setup, please refer to the final report documentation in the `report/` directory.
