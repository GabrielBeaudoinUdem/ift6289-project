import os
import ast
import argparse
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    set_seed
)
from peft import LoraConfig, get_peft_model

def load_and_format_data(task):
    if task == "wiki":
        import json
        data = []
        with open("../data/wiki/data_wiki_baseline.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                data.append({"text": item["text"]})
        return Dataset.from_list(data)
    elif task == "fl":
        df = pd.read_csv("../data/train.csv")
        data = []
        for _, row in df.iterrows():
            # Example: Quelle est la cible pour la fonction lexicale 'Syn' du terme 'rapide' (adj) ?
            instruction = f"Quelle est la cible pour la fonction lexicale '{row['lexical_function']}' du terme '{row['name']}' ({row['POS']}) ?"
            # Ast literal_eval to safely parse string "['target1', 'target2']" to list
            try:
                targets = ast.literal_eval(row['targets'])
                if isinstance(targets, list) and len(targets) > 0:
                    response = targets[0] # We pick the first target for training
                else:
                    response = str(row['targets'])
            except:
                response = str(row['targets'])
            
            data.append({"instruction": instruction, "response": response})
        return Dataset.from_list(data)
    else:
        raise ValueError("Task must be 'wiki' or 'fl'")

def main():
    parser = argparse.ArgumentParser("Mistral LoRA Training")
    parser.add_argument("--task", type=str, required=True, choices=["wiki", "fl"])
    args = parser.parse_args()

    MODEL_NAME = "mistralai/Mistral-7B-v0.1"
    OUTPUT_DIR = f"../models/mistral_lora_{args.task}"

    os.makedirs("../models", exist_ok=True)
    device = torch.device("mps") if torch.backends.mps.is_available() else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    print(f"Using device: {device} for Mistral training (Task: {args.task})")
    
    # Set seed for reproducibility
    set_seed(42)

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load Data
    print("Loading data...")
    dataset = load_and_format_data(args.task)
    if len(dataset) > 5000:
        dataset = dataset.shuffle(seed=42).select(range(5000))
    print(f"Loaded {len(dataset)} examples.")

    def tokenize_function(examples):
        if args.task == "wiki":
            texts = [f"<s> {t} </s>" for t in examples['text']]
        else:
            texts = [
                f"<s>[INST] {inst} [/INST] {resp} </s>" 
                for inst, resp in zip(examples['instruction'], examples['response'])
            ]
        return tokenizer(
            texts, 
            truncation=True, 
            max_length=512, 
            padding="max_length"
        )

    tokenized_dataset = dataset.map(
        tokenize_function, 
        batched=True, 
        remove_columns=dataset.column_names
    )

    # Load Model
    print("Loading Base Model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
    ).to(device)

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    # LoRA Config
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Training args
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        learning_rate=1e-4,
        seed=42,
        data_seed=42,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="no",
        fp16=torch.cuda.is_available(), # Use FP16 on CUDA for performance
        bf16=torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8, # Use BF16 on Ampere+ GPUs
        optim="adamw_torch",
        report_to="none",
        gradient_checkpointing=True # VERY IMPORTANT FOR MAC & VRAM EFFICIENCY
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )

    print("Starting fine-tuning...")
    trainer.train()

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Complete! Model saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
