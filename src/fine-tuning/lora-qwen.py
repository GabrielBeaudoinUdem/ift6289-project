import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model

# -------------------------------------------------
# Configuration
# -------------------------------------------------
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct" 
DATA_PATH = "data_training.jsonl"
OUTPUT_DIR = "./qwen25-7b-lexical"

BATCH_SIZE = 4 
GRAD_ACCUM = 4 
EPOCHS = 3
LR = 1e-4
MAX_LENGTH = 512

# Utilisation du GPU du Mac (change pour cuda sur GPU Nvidia)
DEVICE = torch.device("mps")

# -------------------------------------------------
# Tokenizer & Dataset
# -------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
# Qwen utilise souvent <|endoftext|> comme pad_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def format_qwen_chat(example):
    """
    Formatage au format ChatML (standard pour Qwen)
    """
    text = (
        f"<|im_start|>system\nTu es un expert en lexicologie française (théorie Sens-Texte).<|im_end|>\n"
        f"<|im_start|>user\n{example['instruction']}<|im_end|>\n"
        f"<|im_start|>assistant\n{example['response']}<|im_end|>"
    )
    return {"text": text}

def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length"
    )

dataset = load_dataset("json", data_files=DATA_PATH, split="train")
dataset = dataset.map(format_qwen_chat)
dataset = dataset.map(tokenize, remove_columns=["instruction", "response", "text"])

# -------------------------------------------------
# Modèle (Optimisé pour Mac 42Go)
# -------------------------------------------------
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16, # bfloat16 est recommandé pour Qwen2.5
    trust_remote_code=True
).to(DEVICE)

# Configuration LoRA ciblée sur tous les modules de projection
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# -------------------------------------------------
# Training
# -------------------------------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    logging_steps=10,
    save_strategy="epoch",
    eval_strategy="no",
    bf16=False, 
    fp16=False,
    optim="adamw_torch",
    report_to="none",
    remove_unused_columns=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

print("Lancement de l'entraînement sur Qwen2.5-7B...")
trainer.train()

# -------------------------------------------------
# Sauvegarde
# -------------------------------------------------
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"Fine-tuning terminé. Modèle sauvegardé dans {OUTPUT_DIR}")