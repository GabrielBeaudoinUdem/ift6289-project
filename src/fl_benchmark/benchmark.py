import json
import ast
import time
import random
import pandas as pd
from itertools import product
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
DATA_PATH       = Path(__file__).resolve().parent.parent / "data" / "test.csv"
SENTENCES_PATH  = Path(__file__).resolve().parent.parent / "data" / "benchmarking_patterns.json"
MODEL_PATH      = Path(__file__).resolve().parent.parent / "models" / "model"
OUTPUT_PATH     = Path(__file__).resolve().parent.parent / "results" / "benchmark_results.csv"

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

def build_prompt(word: str, lf: str, pattern: str, 
                 patterns_map: dict[str, list[str]], 
                 df:pd.DataFrame, k: bool, s:bool, 
                 current_example: str) -> str:
    """
    Assemble the full prompt for one (word, lf, k, s) combination.

    - k=True -> use the one-shot example provided for this lexical function
    - s=True -> append the usage sentence from the 'example' column
    """
    lines = []
    
    # Main question
    question = pattern.replace("<W>", word)
    lines.append(f"Q: {question}")

    # One-shot example (k)
    if k:
        one_shot_example = patterns_map.get(lf, {}).get("one_shot", "")
        if one_shot_example:
            lines.append("Example :")
            lines.append(one_shot_example)
            lines.append("")

    # In-context example (s)
    if s and pd.notna(current_example) and str(current_example).strip():
        lines.append(f'(Contexte: "{current_example.strip()}")')

    lines.append("R:")
    return "\n".join(lines)

def generate_answer(prompt: str, tokenizer, model, max_new_tokens: int=64) -> str:
    """
    Run a forward pass and return the generated text (without the prompt).
    """
    inputs = tokenizer(prompt, return_tensors="pt".to(model.device))
    input_len = inputs["input_ids"].shape[1]

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample= False,
        pad_token_id=tokenizer.eos_token_id
    )
    new_tokens = output_ids[0][input_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True.strip())


# ----------------------------------------------------------------------------
# Main benchmark loop
# ----------------------------------------------------------------------------
def run_benchmark(data_path: Path=DATA_PATH, 
                  sentences_path: Path=SENTENCES_PATH,
                  model_path: Path=MODEL_PATH, output_path: Path=OUTPUT_PATH,
                  max_new_tokens: int=64, random_seed:int=42) -> pd.DataFrame:
    random.seed(random_seed)

    # Load data
    df = pd.read_csv(data_path)
    patterns_map = load_patterns(sentences_path)

    # Load model
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.form_pretrained(str(model_path))
    model = AutoModelForCausalLM.from_pretrained(str(model_path))
    model.eval()
    print("Model loaded.\n")
    
    # Condition grid (4 combinaisons of (k,s) )
    conditions = list(product([False, True], repeat=2)) # (k,s)

    records = []
    total = len(df) * len(conditions)
    done = 0

    for _, row in df.iterrows():
        word = str(row["name"])
        lf = str[row["lexical_function"]]
        targets = parse_target(row["targets"])
        example = row.get("example", "")

        # Retrieve the benchmarking pattern for this lf; skip if lf is unknown
        if lf not in patterns_map:
            print(f"[WARN] No pattern found for LF '{lf}', skipping entry '{word}'")
            continue

        lf_patterns = patterns_map[lf]["patterns"]

        # Uses all the patterns for more diversity
        for pattern in lf_patterns:
            for (k, s) in conditions:
                prompt = build_prompt(word=word, lf=lf, pattern=pattern, 
                                      patterns_map=patterns_map, df=df,
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
                    "prompt": prompt
                    "answer": answer,
                    "EM": int(em),
                    "CM": int(cm),
                })

                done += 1
                if done % 20 == 0 or done == total:
                    print(f"    {done}/{total} done...")
        
        results = pd.DataFrame(records)
        results.to_csv(output_path, index=False)
        print(f"\nResults saved to {output_path}")
        _print_summary(results)
        return results
    
# ----------------------------------------------------------------------------
# Summary helper
# ----------------------------------------------------------------------------
 
def _print_summary(results: pd.DataFrame) -> None:
    print("\n=== Summary (mean EM / CM per condition) ===")
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
 
    print("\n=== Summary per LF × condition ===")
    lf_summary = (
        results
        .groupby(["lexical_function", "k", "s"])[["EM", "CM"]]
        .mean()
        .reset_index()
    )
    print(lf_summary.to_string(index=False))


if __name__ == "__main__":
    run_benchmark()
