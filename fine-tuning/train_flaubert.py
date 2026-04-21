import os
import ast
import json
import random
import argparse
from pathlib import Path

import pandas as pd
import torch

from datasets import Dataset
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    default_data_collator,
    set_seed,
)
from peft import LoraConfig, get_peft_model


MODEL_NAME = "flaubert/flaubert_base_cased"


def parse_args():
    parser = argparse.ArgumentParser(description="Train FlauBERT LoRA with MLM objective")
    parser.add_argument("--task", type=str, required=True, choices=["wiki", "fl"])
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_and_format_data(task):
    if task == "wiki":
        data = []
        with open("data/data_wiki_baseline.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                text = str(item.get("text", "")).strip()
                if text:
                    data.append({"text": text})
        return Dataset.from_list(data)

    elif task == "fl":
        df = pd.read_csv("data/train.csv")
        data = []

        for _, row in df.iterrows():
            try:
                targets = ast.literal_eval(row["targets"])
                if isinstance(targets, list) and len(targets) > 0:
                    target = str(targets[0]).strip()
                else:
                    target = str(row["targets"]).strip()
            except Exception:
                target = str(row["targets"]).strip()

            lexical_function = str(row["lexical_function"]).strip()
            term = str(row["name"]).strip()
            pos = str(row["POS"]).strip()

            if lexical_function and term and pos and target:
                data.append({
                    "lexical_function": lexical_function,
                    "term": term,
                    "pos": pos,
                    "target": target
                })

        return Dataset.from_list(data)

    else:
        raise ValueError("Task must be 'wiki' or 'fl'")


def tokenize_wiki(example, tokenizer, max_length):
    return tokenizer(
        example["text"],
        truncation=True,
        max_length=max_length,
        padding="max_length"
    )


def tokenize_fl(example, tokenizer, max_length):
    """
    Build an MLM-style sample for FL:
    We create a structured sentence and only supervise the target span.
    The target tokens are replaced by [MASK], and labels are set only on that span.
    """
    text = (
        f"Fonction lexicale : {example['lexical_function']}. "
        f"Terme : {example['term']} ({example['pos']}). "
        f"Cible : {example['target']}."
    )

    encoded = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding="max_length"
    )

    input_ids = encoded["input_ids"][:]
    attention_mask = encoded["attention_mask"][:]
    labels = [-100] * len(input_ids)

    target_ids = tokenizer.encode(example["target"], add_special_tokens=False)

    if len(target_ids) == 0:
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

    # Find the target span and mask only that part
    found = False
    for i in range(len(input_ids) - len(target_ids) + 1):
        if input_ids[i:i + len(target_ids)] == target_ids:
            for j in range(len(target_ids)):
                labels[i + j] = target_ids[j]
                input_ids[i + j] = tokenizer.mask_token_id
            found = True
            break

    # If exact span not found after tokenization, keep labels = -100
    # so the example does not break training.
    if not found:
        pass

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


def print_sample(dataset, n=2):
    print("\nSample examples:")
    for i in range(min(n, len(dataset))):
        print(dataset[i])


def main():
    args = parse_args()

    output_dir = Path(f"models/flaubert_lora_{args.task}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Task: {args.task}")
    print(f"Output dir: {output_dir}")

    set_seed(args.seed)
    random.seed(args.seed)

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token

    print("Mask token:", tokenizer.mask_token)
    print("Pad token:", tokenizer.pad_token)

    print("\nLoading data...")
    dataset = load_and_format_data(args.task)

    if args.limit is not None and args.limit > 0:
        dataset = dataset.select(range(min(len(dataset), args.limit)))

    print(f"Dataset size: {len(dataset)}")
    print_sample(dataset, n=2)

    print("\nTokenizing dataset...")
    if args.task == "wiki":
        tokenized_dataset = dataset.map(
            lambda x: tokenize_wiki(x, tokenizer, args.max_length),
            batched=False,
            remove_columns=dataset.column_names
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=0.15
        )

    elif args.task == "fl":
        tokenized_dataset = dataset.map(
            lambda x: tokenize_fl(x, tokenizer, args.max_length),
            batched=False,
            remove_columns=dataset.column_names
        )

        data_collator = default_data_collator

    else:
        raise ValueError(f"Unsupported task: {args.task}")

    print("\nLoading base model...")
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)

    print("Applying LoRA...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_lin", "k_lin", "v_lin", "out_lin"],
        lora_dropout=0.05,
        bias="none",
        #task_type="FEATURE_EXTRACTION"
    )

    model = get_peft_model(model, lora_config)
    model.to(device)
    model.print_trainable_parameters()

    print("\nTraining config:")
    print(f"  epochs = {args.epochs}")
    print(f"  batch_size = {args.batch_size}")
    print(f"  grad_accum = {args.grad_accum}")
    print(f"  learning_rate = {args.lr}")
    print(f"  max_length = {args.max_length}")
    print(f"  limit = {args.limit}")

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    print("\nStart training...")
    trainer.train()

    print("\nSaving model and tokenizer...")
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    print("\nDone.")
    print(f"Saved to: {output_dir}")

    # Quick check of saved adapter config
    adapter_config_path = output_dir / "adapter_config.json"
    if adapter_config_path.exists():
        with open(adapter_config_path, "r", encoding="utf-8") as f:
            adapter_cfg = json.load(f)
        print("\nSaved adapter_config.json:")
        print(json.dumps(adapter_cfg, indent=2, ensure_ascii=False))
    else:
        print("\nWarning: adapter_config.json not found after saving.")


if __name__ == "__main__":
    main()
