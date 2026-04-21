import argparse
import json
import ast
from typing import Union
import time
import random
import torch
import pandas as pd
from itertools import product
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

DATA_PATH       = Path(__file__).resolve().parent.parent / "data" / "test.csv"
SENTENCES_PATH  = Path(__file__).resolve().parent.parent / "data" / "benchmarking_patterns.json"
MODEL_PATH      = Path(__file__).resolve().parent.parent / "models" / "model"
OUTPUT_PATH     = Path(__file__).resolve().parent.parent / "fl_benchmark" / "benchmark_results.csv"

def load_model(model_path: Union[str, Path], quantize_4bit: bool = True, device: str = "auto"):
    """
    Load a causal LM and its tokenizer from a local path.
    """
    print(f"Loading model from {model_path}...")
    
    # Auto-detect MPS for Mac if device is "auto"
    if device == "auto":
        if torch.backends.mps.is_available():
            print("  Using MPS (Apple Silicon).")
            # BitsAndBytes doesn't support MPS yet, so we disable quantization if it's on
            if quantize_4bit:
                print("  [WARN] bitsandbytes (4-bit) is not supported on MPS. Falling back to bfloat16/float16.")
                quantize_4bit = False
            device = "mps"
        elif torch.cuda.is_available():
            print("  Using CUDA.")
            device = "auto" # Let accelerate handle it
        else:
            print("  Using CPU.")
            device = "cpu"

    if quantize_4bit:
        try:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            model = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                quantization_config=quant_config,
                device_map="auto",
            )
        except ImportError:
            print("  [WARN] bitsandbytes not found. Falling back to float16.")
            model = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                device_map=device if device != "mps" else None
            )
            if device == "mps":
                model = model.to("mps")
    else:
        # For MPS, we handle device_map manually or set to None as "auto" can be finicky
        dtype = torch.float16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            dtype=dtype,
            device_map=device if device != "mps" else None,
        )
        if device == "mps":
            model = model.to("mps")
 
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
 
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
 
    model.eval()
    print("Model loaded.\n")
    return tokenizer, model
 


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


def run_benchmark(data_path: Path=DATA_PATH, 
                   sentences_path: Path=SENTENCES_PATH,
                   model_path: Path=MODEL_PATH, output_path: Path=OUTPUT_PATH,
                   max_new_tokens: int=64, random_seed:int=42,
                   quantize_4bit: bool = True, device: str = "auto",
                   max_examples: int = None) -> pd.DataFrame:
    random.seed(random_seed)

    # Load data
    df = pd.read_csv(data_path)
    if max_examples:
        df = df.head(max_examples)
        print(f"  [QUICK MODE] Running on first {max_examples} rows.")

    patterns_map = load_patterns(sentences_path)

    # Load model
    tokenizer, model = load_model(model_path, quantize_4bit=quantize_4bit, device=device)
    
    # Condition grid (4 combinaisons of (k,s) )
    conditions = list(product([False, True], repeat=2)) # (k,s)

    records = []
    total = sum(
        len(patterns_map[str(row["lexical_function"])]["patterns"]) * len(conditions)
        for _, row in df.iterrows()
        if str(row["lexical_function"]) in patterns_map
    )
    done = 0

    start_time = time.time()
    for _, row in df.iterrows():
        word        = str(row["name"])
        lf          = str(row["lexical_function"])
        targets     = parse_target(row["targets"])
        example     = row.get("example", "")

        if lf not in patterns_map:
            print(f"[WARN] No pattern found for LF '{lf}', skipping entry '{word}'")
            continue

        lf_patterns = patterns_map[lf]["patterns"]

        for pattern in lf_patterns:
            for (k, s) in conditions:
                prompt = build_prompt(word=word, lf=lf, pattern=pattern, 
                                      patterns_map=patterns_map,
                                      k=k, s=s, current_example=example,
                                      tokenizer=tokenizer)
                
                answer = generate_answer(prompt, tokenizer, model, max_new_tokens)

                em = exact_match(answer, targets)
                cm = contain_match(answer, targets)

                records.append({
                    "word": word,
                    "pos": row.get("POS", ""),
                    "lexical_functions": lf,
                    "pattern": pattern,
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
                    elapsed = time.time() - start_time
                    time_per_step = elapsed / done
                    remaining = (total - done) * time_per_step
                    
                    # Format as MM:SS
                    rem_mins = int(remaining // 60)
                    rem_secs = int(remaining % 60)
                    elapsed_mins = int(elapsed // 60)
                    elapsed_secs = int(elapsed % 60)
                    
                    print(f"    {done}/{total} done... (Elapsed: {elapsed_mins:02d}:{elapsed_secs:02d}, Est. Remaining: {rem_mins:02d}:{rem_secs:02d})")
        
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(records)
    results.to_csv(output_path, index=False)
    
    total_elapsed = time.time() - start_time
    mins = int(total_elapsed // 60)
    secs = int(total_elapsed % 60)
    print(f"\nResults saved to {output_path}")
    print(f"Total time: {mins}m {secs}s")
    _print_summary(results)
    return results
    
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
    parser = argparse.ArgumentParser(description="Run FL Benchmark")
    parser.add_argument("--model_path", type=str, default=str(MODEL_PATH), help="Path to the model directory")
    parser.add_argument("--data_path", type=str, default=str(DATA_PATH), help="Path to test.csv")
    parser.add_argument("--patterns_path", type=str, default=str(SENTENCES_PATH), help="Path to benchmarking_patterns.json")
    parser.add_argument("--output_path", type=str, default=None, help="Path to save results")
    parser.add_argument("--max_new_tokens", type=int, default=64, help="Max tokens to generate")
    parser.add_argument("--quantize_4bit", action="store_true", help="Enable 4-bit quantization (if supported)")
    parser.add_argument("--device", type=str, default="auto", help="Device to use (auto, mps, cuda, cpu)")
    parser.add_argument("--quick", action="store_true", help="Run only 5 examples for testing")

    args = parser.parse_args()

    model_path = Path(args.model_path)
    if not args.output_path:
        # Default to fl_benchmark/<model_name>.csv
        model_name = model_path.name
        output_path = Path(__file__).resolve().parent.parent / "fl_benchmark" / f"{model_name}.csv"
    else:
        output_path = Path(args.output_path)

    run_benchmark(
        data_path=Path(args.data_path),
        sentences_path=Path(args.patterns_path),
        model_path=model_path,
        output_path=output_path,
        max_new_tokens=args.max_new_tokens,
        quantize_4bit=args.quantize_4bit,
        device=args.device,
        max_examples=5 if args.quick else None
    )
