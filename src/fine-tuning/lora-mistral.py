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
MODEL_NAME = "mistralai/Mistral-7B-v0.1"
DATA_PATH = "data_training.jsonl"
OUTPUT_DIR = "./mistral-lexical-lora"

# Utilisation du GPU du Mac (change pour cuda sur GPU Nvidia)
DEVICE = torch.device("mps")

# -------------------------------------------------
# Tokenizer
# -------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# -------------------------------------------------
# Dataset : Adaptation au format Mistral
# -------------------------------------------------
def tokenize_function(examples):
    # Formatage spécifique à Mistral [INST] Question [/INST] Réponse
    texts = [
        f"<s>[INST] {inst} [/INST] {resp} </s>" 
        for inst, resp in zip(examples['instruction'], examples['response'])
    ]
    return tokenizer(
        texts, 
        truncation=True, 
        max_length=256, 
        padding="max_length"
    )

dataset = load_dataset("json", data_files=DATA_PATH, split="train")
tokenized_dataset = dataset.map(
    tokenize_function, 
    batched=True, 
    remove_columns=dataset.column_names
)

# -------------------------------------------------
# Modèle
# -------------------------------------------------
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
).to(DEVICE)

# -------------------------------------------------
# Configuration LoRA
# -------------------------------------------------
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

# -------------------------------------------------
# Training arguments (CORRIGÉS)
# -------------------------------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=1e-4,
    logging_steps=10,
    save_strategy="epoch",
    eval_strategy="no",
    fp16=False,
    bf16=False,
    optim="adamw_torch",
    report_to="none",
    gradient_checkpointing=False 
)

# -------------------------------------------------
# Entraînement
# -------------------------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

print("Lancement du fine-tuning sur Mistral...")
trainer.train()

# Sauvegarde finale
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Entraînement terminé ! Modèle sauvegardé dans {OUTPUT_DIR}")