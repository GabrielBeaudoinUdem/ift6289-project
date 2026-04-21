import argparse
import gc
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Union
import pandas as pd

# ---------------------------------------------------------------------------
# 0. Setup paths & environment BEFORE any COLE / HF imports
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
COLE_DIR = SCRIPT_DIR / "COLE"
MODELS_DIR = PROJECT_DIR / "models"
RESULTS_DIR = PROJECT_DIR / "cole_benchmark"

# Add COLE repo to sys.path so we can import `src.*`
sys.path.insert(0, str(COLE_DIR))

# Disable wandb (COLE uses it everywhere)
os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_SILENT"] = "true"

# Suppress noisy logs from transformers / datasets
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "warning")
os.environ.setdefault("DATASETS_VERBOSITY", "warning")

# ---------------------------------------------------------------------------
# Now safe to import torch / transformers / COLE
# ---------------------------------------------------------------------------
import torch
from datasets import Dataset as HFDataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

try:
    from peft import PeftModel, PeftConfig
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

# COLE imports
from src.language_model.language_model_abstraction import LanguageModel
from src.task.task import Task, TaskType
from src.task.task_names import COLETasks, BorealTasks

# Stub wandb so compute_metrics doesn't crash
import wandb
wandb.init = lambda *a, **kw: None
wandb.log = lambda *a, **kw: None
wandb.finish = lambda *a, **kw: None

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# Local Model Wrapper

class LocalModel(LanguageModel):
    """
    Wrapper to load a local HuggingFace model and expose it to the COLE
    evaluation pipeline via the LanguageModel abstraction.
    """

    def __init__(
        self,
        model_path: str,
        model_name: str,
        batch_size: int = 1,
    ):
        super().__init__(model_name)
        self._model_path = model_path
        self._batch_size = 1  # Forcé à 1 pour désactiver le padding/batching

        log.info(f"Loading model from {model_path} ...")
        device = self._get_device()
        log.info(f"Using device: {device}")

        # ------------------------------------------------------------------
        # Detect LoRA adapter vs full model
        # ------------------------------------------------------------------
        adapter_config_path = Path(model_path) / "adapter_config.json"
        is_lora = adapter_config_path.exists()

        # ------------------------------------------------------------------
        # Load model with best available configuration
        # ------------------------------------------------------------------
        load_kwargs: Dict = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }

        if device == "cuda" and self._bitsandbytes_available():
            # 4-bit quantization on CUDA
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            load_kwargs["device_map"] = "auto"
            log.info("Using 4-bit quantization (bitsandbytes)")
        elif device == "cuda":
            load_kwargs["torch_dtype"] = torch.float16
            load_kwargs["device_map"] = "auto"
        elif device == "mps":
            load_kwargs["torch_dtype"] = torch.float16
            load_kwargs["device_map"] = "auto"
        else:
            # CPU fallback
            load_kwargs["torch_dtype"] = torch.float32

        if is_lora:
            # LoRA adapter: load base model then attach adapter
            if not PEFT_AVAILABLE:
                raise ImportError(
                    "peft is required to load LoRA adapters. "
                    "Install it with: pip install peft"
                )
            with open(adapter_config_path, "r") as f:
                adapter_cfg = json.load(f)
            base_model_id = adapter_cfg["base_model_name_or_path"]
            log.info(f"LoRA adapter detected. Base model: {base_model_id}")

            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_id, **load_kwargs
            )
            self.model = PeftModel.from_pretrained(
                base_model, model_path
            )
            self.model = self.model.merge_and_unload()
            log.info("LoRA adapter merged into base model.")
        else:
            # Full model: load directly
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, **load_kwargs
            )

        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        # Tokenizer padding has been removed/ignored since we force batch_size=1
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        num_params = sum(p.numel() for p in self.model.parameters())
        log.info(f"Model loaded: {num_params / 1e9:.1f}B parameters")

    # ------------------------------------------------------------------
    # predict() is the main entry point called by COLE's ModelEvaluator
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # predict() is the main entry point called by COLE's ModelEvaluator
    # ------------------------------------------------------------------
    def predict(self, evaluation_dataset: HFDataset, task: Task) -> List:
        """
        Use text-generation for ALL tasks when using CausalLM (Mistral/Qwen).
        """
        log.info(f"Setting up text-generation pipeline for task: {task.task_name}")

        # Determine if it's a generative vs classification task to set max_tokens
        max_tokens = 64 if task.task_type == TaskType.GENERATIVE else 16

        # We use a single generation pipeline for all tasks
        self._pipeline = pipeline(
            task="text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            batch_size=self._batch_size,
            return_full_text=False,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
            truncation=True,
            repetition_penalty=1.15,
        )

        self._current_task = task

        process_dataset = evaluation_dataset.map(
            self._generate_and_parse,
            batched=True,
            batch_size=self._batch_size,
            desc=f"Running evaluation for task: {task.task_name}",
            remove_columns="text",
            load_from_cache_file=False,
        )
        return list(process_dataset["prediction"])

    def generate(self, rows) -> Dict:
        """Required by LanguageModel ABC."""
        return self._generate_and_parse(rows)

    def infer(self, rows) -> Dict:
        """Required by LanguageModel ABC."""
        return self._generate_and_parse(rows)

    def _generate_and_parse(self, rows) -> Dict:

        with torch.no_grad():
            texts = rows["text"]
            outputs = self._pipeline(texts)

            predictions = []
            for output in outputs:
                # Handle both types of pipeline output formats
                if isinstance(output, list):
                    gen_text = output[0]["generated_text"].strip()
                else:
                    gen_text = output["generated_text"].strip()

                parsed = self._parse_label(gen_text, self._current_task)
                predictions.append(parsed)

        return {"prediction": predictions}

    def _parse_label(self, text: str, task: Task) -> str:
        labels = task.dataset.possible_ground_truths
        if not labels:
            # Generative tasks (QA, WSD) -> return raw text
            return text

        # Clean text
        clean_text = text.lower().strip().rstrip(".,!?:")

        # 1. Look for exact match
        for label in labels:
            if clean_text == str(label).lower():
                return label

        # 2. Check if the output STARTS with one of the labels
        for label in labels:
            if clean_text.startswith(str(label).lower()):
                return label

        # 3. Check for label in the first "word"
        first_part = clean_text.split()[0].rstrip(".,!?:") if clean_text else ""
        for label in labels:
            if first_part == str(label).lower():
                return label

        # 4. Exclusive presence check (useful for "0", "1" embedded in long sentences)
        import re
        present_labels = []
        for label in labels:
            # Match the label as a distinct word using word boundaries
            pattern = r'\b' + re.escape(str(label).lower()) + r'\b'
            if re.search(pattern, clean_text):
                present_labels.append(label)
        
        # If exactly one label is found anywhere in the text, we can confidently assume it's the model's choice
        if len(present_labels) == 1:
            return present_labels[0]

        # 5. Search for the label anywhere in the short output (constrained)
        # Only do this for labels that are long enough to avoid false positives (like "0")
        for label in labels:
            if len(str(label)) > 1 and f" {str(label).lower()} " in f" {clean_text} ":
                return label

        # Default fallback to raw text (metrics will handle casting/failure)
        return text


    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _get_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @staticmethod
    def _bitsandbytes_available() -> bool:
        try:
            import bitsandbytes  # noqa: F401
            return True
        except ImportError:
            return False


# Task Factory
# We re-implement a minimal task factory here to avoid relying on
# Python 3.10+ match/case and the prediction/*.py imports in COLE.

TASK_CONFIGS = {
    # Sentiment Analysis
    "allocine":             ("accuracy", TaskType.INFERENCE),
    "mms":                  ("accuracy", TaskType.INFERENCE),
    # NLI
    "fracas":               ("accuracy", TaskType.INFERENCE),
    "gqnli":                ("accuracy", TaskType.INFERENCE),
    "lingnli":              ("accuracy", TaskType.INFERENCE),
    "mnli-nineeleven-fr-mt":("accuracy", TaskType.INFERENCE),
    "rte3-french":          ("accuracy", TaskType.INFERENCE),
    "sickfr":               ("accuracy", TaskType.INFERENCE),
    "xnli":                 ("accuracy", TaskType.INFERENCE),
    # QA
    "fquad":                ("fquad",    TaskType.GENERATIVE),
    "french_boolq":         ("accuracy", TaskType.INFERENCE),
    "piaf":                 ("fquad",    TaskType.GENERATIVE),
    # Paraphrase
    "paws_x":               ("accuracy", TaskType.INFERENCE),
    "qfrblimp":             ("accuracy", TaskType.INFERENCE),
    # Grammar
    "daccord":              ("accuracy", TaskType.INFERENCE),
    "multiblimp":           ("accuracy", TaskType.INFERENCE),
    "qfrcola":              ("accuracy", TaskType.INFERENCE),
    # Semantic Similarity
    "sts22":                ("accuracy", TaskType.INFERENCE),
    # WSD
    "wsd":                  ("em",       TaskType.GENERATIVE),
    # Quebec French
    "qfrcore":              ("accuracy", TaskType.INFERENCE),
    "qfrcort":              ("accuracy", TaskType.INFERENCE),
    # frcoe removed — not available in graalul/COLE-public
    # Coreference
    "wino_x_lm":            ("accuracy", TaskType.INFERENCE),
    "wino_x_mt":            ("accuracy", TaskType.INFERENCE),
}


def build_tasks(task_names: List[str]) -> List[Task]:
    """Build Task objects without the match/case task_factory."""
    tasks = []
    for name in task_names:
        if name not in TASK_CONFIGS:
            log.warning(f"Unknown task '{name}', skipping.")
            continue
        metric, task_type = TASK_CONFIGS[name]
        tasks.append(Task(task_name=name, metric=metric, task_type=task_type))
    return tasks


# Evaluation Loop

def evaluate_model(
    model: LocalModel,
    tasks: List[Task],
    max_examples: Union[int, None] = None,
) -> Dict:
    """
    Run the full COLE evaluation for a single model.
    Returns a results dict with per-task predictions, metrics, and total score.
    """
    all_predictions = []
    all_metrics = []
    task_scores = {}
    csv_rows = []

    for task in tqdm(tasks, desc=f"Evaluating {model.name}"):
        log.info(f"--- Task: {task.task_name} ---")
        try:
            # Build prompts
            if task.dataset.dataset is None:
                log.warning(f"Dataset for task '{task.task_name}' could not be loaded. Skipping.")
                all_predictions.append({task.task_name: []})
                all_metrics.append({task.task_name: {"error": "Dataset could not be loaded"}})
                continue

            if max_examples is not None:
                prompts = task.dataset.prompts[:max_examples]
                gts = task.dataset.ground_truths[:max_examples] if hasattr(task.dataset, 'ground_truths') else [task.dataset.possible_ground_truths] * len(prompts)
            else:
                prompts = task.dataset.prompts[:]
                gts = task.dataset.ground_truths[:] if hasattr(task.dataset, 'ground_truths') else [task.dataset.possible_ground_truths] * len(prompts)


            eval_dataset = HFDataset.from_dict({"text": prompts})

            # Run prediction
            preds = model.predict(evaluation_dataset=eval_dataset, task=task)
            all_predictions.append({task.task_name: preds})
            
            # Save CSV rows
            for p, pr, gt in zip(prompts, preds, gts):
                csv_rows.append({
                    "task_name": task.task_name,
                    "prompt": p,
                    "prediction": pr,
                    "ground_truth": str(gt)
                })

            # Compute metric
            metric_score, warning = task.compute(preds)
            metric_name = task.metric_name
            task_entry = {
                task.task_name: {
                    metric_name: {**metric_score, f"{metric_name}_warning": warning}
                }
            }
            all_metrics.append(task_entry)

            # Extract numeric score for COLE total (usually the first float/int value we find)
            score_val = None
            for k, v in metric_score.items():
                if isinstance(v, (int, float)):
                    score_val = v
                    break
            if score_val is not None:
                task_scores[task.task_name] = score_val

            log.info(f"  -> {metric_name}: {metric_score}")

        except Exception as e:
            log.error(f"Task '{task.task_name}' failed: {e}")
            all_predictions.append({task.task_name: []})
            all_metrics.append({task.task_name: {"error": str(e)}})

        finally:
            # Memory cleanup
            if "eval_dataset" in locals():
                del eval_dataset
            if "prompts" in locals():
                del prompts
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                torch.mps.empty_cache()
            gc.collect()

    # ------------------------------------------------------------------
    # Compute COLE total score
    # ------------------------------------------------------------------
    if task_scores:
        cole_score = sum(task_scores.values()) / len(task_scores)
    else:
        cole_score = 0.0

    completed = len(task_scores)
    failed = len(tasks) - completed

    results = {
        "model_name": model.name,
        "model_path": str(model._model_path),
        "evaluation_date": datetime.now().isoformat(),
        "cole_score": round(cole_score, 4),
        "summary": {
            "total_tasks": len(tasks),
            "completed_tasks": completed,
            "failed_tasks": failed,
        },
        "task_scores": task_scores,
        "tasks_predictions": all_predictions,
        "tasks_metrics": all_metrics,
        "csv_data": pd.DataFrame(csv_rows)
    }

    return results


def save_results(results: Dict, output_dir: Path) -> Path:
    """Save results dict to a JSON file and CSV responses."""
    output_dir.mkdir(parents=True, exist_ok=True)
    model_name_safe = results["model_name"].replace("/", "_").replace(" ", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save CSV
    if "csv_data" in results:
        csv_df = results.pop("csv_data")
        csv_path = output_dir / f"{model_name_safe}_cole_results_{timestamp}.csv"
        csv_df.to_csv(csv_path, index=False)
        log.info(f"Predictions saved to CSV: {csv_path}")

    # Save JSON
    filename = f"{model_name_safe}_cole_results_{timestamp}.json"
    filepath = output_dir / filename

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    log.info(f"Metrics saved to JSON: {filepath}")
    return filepath


# Main Application

def main():
    parser = argparse.ArgumentParser(
        description="COLE Benchmark Evaluation for local models"
    )
    ALL_MODELS = [
        "mistral_base", "mistral_lora_fl", "mistral_lora_wiki",
        "qwen_base", "qwen_lora_fl", "qwen_lora_wiki",
    ]
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        action="append",
        choices=ALL_MODELS,
        help="Model(s) to evaluate. Can be specified multiple times. "
             "If not specified, evaluates all 6 models.",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=100,
        help="Max examples per task. Default: 100. Use 0 for all examples.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for inference. Default: 4.",
    )
    parser.add_argument(
        "--tasks_group",
        type=str,
        default="cole",
        choices=["cole", "all"],
        help="Task group: 'cole' (23 COLE tasks) or 'all' (COLE + Boreal).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help=f"Output directory for results. Default: {RESULTS_DIR}",
    )

    args = parser.parse_args()

    # Determine output directory
    output_dir = Path(args.output_dir) if args.output_dir else RESULTS_DIR

    # Determine which models to evaluate
    if args.model:
        model_dirs = args.model
    else:
        model_dirs = ALL_MODELS

    # Determine which tasks to run
    if args.tasks_group == "cole":
        task_names = [t.value for t in COLETasks]
    else:
        task_names = [t.value for t in COLETasks] + [t.value for t in BorealTasks]

    log.info(f"Building {len(task_names)} tasks...")
    tasks = build_tasks(task_names)
    log.info(f"Loaded {len(tasks)} tasks successfully.")

    # Treat 0 as "use all"
    if args.max_examples == 0:
        args.max_examples = None

    if args.max_examples:
        log.info(f"Capping at {args.max_examples} examples per task.")

    # ------------------------------------------------------------------
    # Evaluate each model
    # ------------------------------------------------------------------
    all_results = []

    for model_dir_name in model_dirs:
        model_path = MODELS_DIR / model_dir_name

        if not model_path.exists():
            log.error(f"Model directory not found: {model_path}")
            continue

        log.info("=" * 60)
        log.info(f"EVALUATING: {model_dir_name}")
        log.info("=" * 60)

        model = LocalModel(
            model_path=str(model_path),
            model_name=model_dir_name,
            batch_size=args.batch_size,
        )

        results = evaluate_model(model, tasks, max_examples=args.max_examples)
        filepath = save_results(results, output_dir)
        all_results.append(results)

        # Print summary
        print("\n" + "=" * 60)
        print(f"  {model_dir_name} — COLE SCORE: {results['cole_score']:.4f}")
        print(f"  Completed: {results['summary']['completed_tasks']}/{results['summary']['total_tasks']}")
        print("=" * 60)
        print("\nPer-task scores:")
        for task_name, score in sorted(results["task_scores"].items()):
            print(f"  {task_name:30s} {score:.4f}")
        print()

        # Free model memory
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()

    # ------------------------------------------------------------------
    # Final summary across all models
    # ------------------------------------------------------------------
    if len(all_results) > 1:
        print("\n" + "=" * 60)
        print("  FINAL COMPARISON")
        print("=" * 60)
        for r in all_results:
            print(f"  {r['model_name']:20s}  COLE Score = {r['cole_score']:.4f}")
        print("=" * 60)


if __name__ == "__main__":
    main()
