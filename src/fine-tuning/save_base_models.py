import os
from transformers import AutoModelForCausalLM, AutoTokenizer

def save_models():
    os.makedirs("../models", exist_ok=True)
    
    models = {
        "mistral_base": ("mistralai/Mistral-7B-v0.1", AutoModelForCausalLM),
        "qwen_base": ("Qwen/Qwen2.5-7B", AutoModelForCausalLM)
    }
    
    for name, (repo_id, model_class) in models.items():
        out_dir = f"../models/{name}"
        if os.path.exists(out_dir):
            print(f"Le modèle {name} est déjà sauvegardé dans {out_dir}. Ignition...")
            continue
            
        print(f"Téléchargement de {repo_id} depuis HuggingFace...")
        tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)
        model = model_class.from_pretrained(repo_id, trust_remote_code=True)
        
        tokenizer.save_pretrained(out_dir)
        model.save_pretrained(out_dir)
        print(f"✓ {name} sauvegardé avec succès dans {out_dir}")

if __name__ == "__main__":
    print("=== SAUVEGARDE DES MODÈLES DE BASE ===")
    save_models()
    print("Processus terminé.")
