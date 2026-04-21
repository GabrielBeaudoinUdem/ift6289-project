import json
import random
from datasets import load_dataset

random.seed(42)
NUM_SAMPLES = 5000
OUTPUT_FILE = "data_wiki_baseline.jsonl"

try:
    wiki_dataset = load_dataset("wikimedia/wikipedia", "20231101.fr", split="train", streaming=True)
except Exception as e:
    print(f"Erreur avec wikimedia/wikipedia: {e}. Essai alternatif...")
    wiki_dataset = load_dataset("wikipedia", "20220301.fr", split="train", streaming=True)

wiki_dataset = wiki_dataset.shuffle(seed=42, buffer_size=10000)

print(f"Extraction de {NUM_SAMPLES} échantillons en cours...")

samples = []
for item in wiki_dataset:
    if len(samples) >= NUM_SAMPLES:
        break
    
    text = item["text"].strip()
    
    if len(text) < 50:
        continue
        
    sample = {
        "text": text[:4000]
    }
    samples.append(sample)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for s in samples:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")

print(f"Terminé, sauvegardé {len(samples)} exemples dans {OUTPUT_FILE}")
