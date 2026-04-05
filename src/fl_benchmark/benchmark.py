import json
import ast
import time
import random
import torch
import pandas as pd
from itertools import product
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
DATA_PATH       = Path(__file__).resolve().parent.parent / "data" / "test.csv"
SENTENCES_PATH  = Path(__file__).resolve().parent.parent / "data" / "benchmarking_patterns.json"
MODEL_PATH      = Path(__file__).resolve().parent.parent / "models" / "model"
OUTPUT_PATH     = Path(__file__).resolve().parent.parent / "results" / "benchmark_results.csv"

# ----------------------------------------------------------------------------
# Model loading
# ----------------------------------------------------------------------------
 
def load_model(model_path: str | Path , quantize_4bit: bool = True):
    """
    Load a causal LM and its tokenizer from a local path.
 
    Args:
        model_path:    Path to the model directory (HuggingFace format).
        quantize_4bit: If True, load in 4-bit with bfloat16 compute dtype
                       (requires bitsandbytes). Reduces VRAM usage ~4×.
                       Set to False to load in full precision (fp32/fp16).
 
    Returns:
        tokenizer, model
    
    Swap model_path for any HuggingFace model ID to download directly, e.g.:
        "mistralai/Mistral-7B-Instruct-v0.3"
        "meta-llama/Meta-Llama-3-8B-Instruct"
        "croissantllm/CroissantLLMBase"          # French-focused base model
        "OpenLLM-France/Lucie-7B-Instruct"       # French instruction-tuned
    """
    print(f"Loading model from {model_path}...")
 
    if quantize_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            quantization_config=quant_config,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype=torch.float16,
            device_map="auto",
        )
 
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
 
    # Some tokenizers (e.g. LLaMA) have no pad token by default
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
 
    model.eval()
    print("Model loaded.\n")
    return tokenizer, model
 

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def load_patterns(path: Path) -> dict[str, list[str]]:
    """
    Return:
    {
        lf: {
            "patterns": [...],
            "one-shot": "..."
        }
    }
    """
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    return {
        entry["lf"]: {
            "patterns": entry["free_pattern"],
            "one_shot": entry.get("one_shot", "")
        } 
        for entry in raw["patterns"]    
    }

def parse_target(targets_raw) -> list[str]:
    """
    Parse the targets column, which is stored as a string representation of a 
    list.
    """
    if isinstance(targets_raw, list):
        return [t.strip().lower() for t in targets_raw]
    try:
        parsed = ast.literal_eval(targets_raw)
        return [t.strip().lower() for t in parsed]
    except Exception:
        return [targets_raw.strip().lower()]
    
def exact_match(answer: str, targets: list[str]) -> bool:
    return answer.strip().lower() in targets

def contain_match(answer: str, targets: list[str]) -> bool:
    answer_lower = answer.lower()
    return any(t in answer_lower for t in targets)

def is_instruction_tuned(tokenizer) -> bool:
    """
    Heuristic: instruction-tuned models have a chat template set in their
    tokenizer config. Use that to decide how to format the prompt.
    """
    return getattr(tokenizer, "chat_template", None) is not None


def build_prompt(word: str, lf: str, pattern: str,
                 patterns_map: dict[str, dict],
                 k: bool, s: bool,
                 current_example: str,
                 tokenizer) -> str:
    """
    Assemble the full prompt for one (word, lf, pattern, k, s) combination.
 
    - k=True  → prepend the one-shot example for this lexical function
    - s=True  → append the usage sentence from the 'example' column
 
    If the tokenizer has a chat template (instruction-tuned model), the
    prompt is wrapped accordingly. Otherwise, raw Q/R format is used for
    base models.
    """
    lines = []
 
    question = pattern.replace("<W>", word)
 
    if k:
        one_shot = patterns_map.get(lf, {}).get("one_shot", "")
        if one_shot:
            lines.append("Exemple :")
            lines.append(one_shot)
            lines.append("")
 
    if s and pd.notna(current_example) and str(current_example).strip():
        lines.append(f'(Contexte : "{current_example.strip()}")')
 
    lines.append(f"Q : {question}")
    lines.append("R :")
 
    user_content = "\n".join(lines)
 
    if is_instruction_tuned(tokenizer):
        messages = [{"role": "user", "content": user_content}]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
 
    return user_content


def extract_first_answer(text: str) -> str:
    """
    Keep only the first meaningful token sequence before any newline,
    period, or comma. This avoids penalising models that give the right
    word but then add an explanation.
    """
    for sep in ("\n", ".", ",", ";"):
        text = text.split(sep)[0]
    return text.strip()

def generate_answer(prompt: str, tokenizer, model, max_new_tokens: int=64) -> str:
    """
    Run a forward pass and return the generated text (without the prompt).
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample= False,
        pad_token_id=tokenizer.eos_token_id
    )
    new_tokens = output_ids[0][input_len:]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return extract_first_answer(raw)


# ----------------------------------------------------------------------------
# Main benchmark loop
# ----------------------------------------------------------------------------
def run_benchmark(data_path: Path=DATA_PATH, 
                  sentences_path: Path=SENTENCES_PATH,
                  model_path: Path=MODEL_PATH, output_path: Path=OUTPUT_PATH,
                  max_new_tokens: int=64, random_seed:int=42,
                  quantize_4bit: bool = True) -> pd.DataFrame:
    random.seed(random_seed)

    # Load data
    df = pd.read_csv(data_path)
    patterns_map = load_patterns(sentences_path)

    # Load model
    tokenizer, model = load_model(model_path, quantize_4bit=quantize_4bit)
    
    # Condition grid (4 combinaisons of (k,s) )
    conditions = list(product([False, True], repeat=2)) # (k,s)

    records = []
    total = sum(
        len(patterns_map[str(row["lexical_function"])]["patterns"]) * len(conditions)
        for _, row in df.iterrows()
        if str(row["lexical_function"]) in patterns_map
    )
    done = 0

    for _, row in df.iterrows():
        word        = str(row["name"])
        lf          = str(row["lexical_function"])
        targets     = parse_target(row["targets"])
        example     = row.get("example", "")

        # Retrieve the benchmarking pattern for this lf; skip if lf is unknown
        if lf not in patterns_map:
            print(f"[WARN] No pattern found for LF '{lf}', skipping entry '{word}'")
            continue

        lf_patterns = patterns_map[lf]["patterns"]

        # Uses all the patterns for more diversity
        for pattern in lf_patterns:
            for (k, s) in conditions:
                prompt = build_prompt(word=word, lf=lf, pattern=pattern, 
                                      patterns_map=patterns_map,
                                      k=k, s=s, current_example=example)
                
                answer = generate_answer(prompt, tokenizer, model, max_new_tokens)

                em = exact_match(answer, targets)
                cm = contain_match(answer, targets)

                records.append({
                    "word": word,
                    "pos": row.get("POS", ""),
                    "lexical_functions": lf,
                    "targets": str(targets),
                    "k": k,
                    "s": s,
                    "prompt": prompt,
                    "answer": answer,
                    "EM": int(em),
                    "CM": int(cm)
                })

                done += 1
                if done % 20 == 0 or done == total:
                    print(f"    {done}/{total} done...")
        
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(records)
    results.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")
    _print_summary(results)
    return results
    
# ----------------------------------------------------------------------------
# Summary helper
# ----------------------------------------------------------------------------
 
def _print_summary(results: pd.DataFrame) -> None:
    print("\n=== Mean EM / CM per condition ===")
    summary = (
        results
        .groupby(["k", "s"])[["EM", "CM"]]
        .mean()
        .rename(columns={"EM": "EM_mean", "CM": "CM_mean"})
        .reset_index()
    )
    summary["condition"] = summary.apply(
        lambda r: f"k={'1' if r['k'] else '0'}, s={'1' if r['s'] else '0'}", axis=1
    )
    for _, r in summary.iterrows():
        print(f"  {r['condition']} → EM={r['EM_mean']:.3f}  CM={r['CM_mean']:.3f}")
 
    print("\n=== Mean EM / CM per LF × condition ===")
    lf_summary = (
        results
        .groupby(["lexical_functions", "k", "s"])[["EM", "CM"]]
        .mean()
        .reset_index()
    )
    print(lf_summary.to_string(index=False))
 
    print("\n=== Mean EM / CM per LF (averaged over all conditions and patterns) ===")
    lf_global = (
        results
        .groupby("lexical_functions")[["EM", "CM"]]
        .mean()
        .sort_values("EM", ascending=False)
    )
    print(lf_global.to_string())
 
    print("\n=== Mean EM / CM per pattern (averaged over all words and conditions) ===")
    pattern_summary = (
        results
        .groupby(["lexical_functions", "pattern"])[["EM", "CM"]]
        .mean()
        .reset_index()
    )


if __name__ == "__main__":
    run_benchmark(model_path="mistralai/Mistral-7B-v0.1")
