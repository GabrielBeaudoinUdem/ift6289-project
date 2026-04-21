import json
import ast
import random
import torch
import pandas as pd
import torch.nn.functional as F
from itertools import product
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForMaskedLM
from peft import PeftModel

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
DATA_PATH      = Path("data/test.csv")
SENTENCES_PATH = Path("data/benchmarking_patterns.json")
OUTPUT_DIR     = Path("results")

BASE_MODEL = "flaubert/flaubert_base_cased"


MODEL_VARIANTS = {
    "base": None,
    "wiki": "models/flaubert_lora_wiki",
    "fl": "models/flaubert_lora_fl",
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def load_patterns(path: Path) -> dict[str, dict]:
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
    if isinstance(targets_raw, list):
        return [str(t).strip().lower() for t in targets_raw]
    try:
        parsed = ast.literal_eval(targets_raw)
        return [str(t).strip().lower() for t in parsed]
    except Exception:
        return [str(targets_raw).strip().lower()]


def exact_match(answer: str, targets: list[str]) -> bool:
    return answer.strip().lower() in targets


def contain_match(answer: str, targets: list[str]) -> bool:
    answer_lower = answer.lower()
    return any(t in answer_lower for t in targets)


def build_prompt(word: str, lf: str, pattern: str,
                 patterns_map: dict[str, dict],
                 k: bool, s: bool,
                 current_example: str) -> str:
    lines = []

    question = pattern.replace("<W>", word)

    if k:
        one_shot = patterns_map.get(lf, {}).get("one_shot", "")
        if one_shot:
            lines.append("Exemple :")
            lines.append(one_shot)
            lines.append("")

    if s and pd.notna(current_example) and str(current_example).strip():
        lines.append(f'(Contexte : "{str(current_example).strip()}")')

    lines.append(f"Q : {question}")
    lines.append("R :")

    return "\n".join(lines)


def load_model(variant_name: str):
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token

    model = AutoModelForMaskedLM.from_pretrained(BASE_MODEL)

    adapter_path = MODEL_VARIANTS[variant_name]
    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, adapter_path)

    model.to(DEVICE)
    model.eval()
    return tokenizer, model


def build_candidate_pool(df: pd.DataFrame) -> dict[str, list[str]]:
    """
    For each LF, gather all unique target words appearing in the benchmark set.
    """
    lf_to_candidates = {}

    for _, row in df.iterrows():
        lf = str(row["lexical_function"])
        targets = parse_target(row["targets"])

        if lf not in lf_to_candidates:
            lf_to_candidates[lf] = []

        for t in targets:
            if t not in lf_to_candidates[lf]:
                lf_to_candidates[lf].append(t)

    return lf_to_candidates


@torch.no_grad()
def score_candidate_pll(prompt: str, candidate: str, tokenizer, model, max_length: int = 512) -> float:
    """
    Pseudo-log-likelihood for MLM.
    Score the candidate appended after the prompt.
    """
    full_text = prompt.strip() + " " + candidate.strip()

    enc_full = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=max_length)
    enc_prompt = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)

    input_ids = enc_full["input_ids"][0]
    attention_mask = enc_full["attention_mask"][0]
    prompt_len = enc_prompt["input_ids"].shape[1]

    special_ids = set(tokenizer.all_special_ids)
    candidate_positions = []

    for pos in range(prompt_len, input_ids.shape[0]):
        tok_id = input_ids[pos].item()
        if tok_id not in special_ids:
            candidate_positions.append(pos)

    if len(candidate_positions) == 0:
        return -1e9

    total_logprob = 0.0

    for pos in candidate_positions:
        masked_ids = input_ids.clone()
        true_id = masked_ids[pos].item()
        masked_ids[pos] = tokenizer.mask_token_id

        outputs = model(
            input_ids=masked_ids.unsqueeze(0).to(DEVICE),
            attention_mask=attention_mask.unsqueeze(0).to(DEVICE)
        )
        logits = outputs.logits[0, pos]
        log_probs = F.log_softmax(logits, dim=-1)
        total_logprob += log_probs[true_id].item()

    return total_logprob / len(candidate_positions)


@torch.no_grad()
def predict_answer(prompt: str, lf: str, lf_to_candidates: dict[str, list[str]], tokenizer, model) -> str:
    candidates = lf_to_candidates.get(lf, [])
    if not candidates:
        return ""

    best_candidate = None
    best_score = -1e18

    for cand in candidates:
        score = score_candidate_pll(prompt, cand, tokenizer, model)
        if score > best_score:
            best_score = score
            best_candidate = cand

    return best_candidate if best_candidate is not None else ""


# ----------------------------------------------------------------------------
# Main benchmark loop
# ----------------------------------------------------------------------------
def run_benchmark(variant_name: str,
                  data_path: Path = DATA_PATH,
                  sentences_path: Path = SENTENCES_PATH) -> pd.DataFrame:
    print(f"\n===== Running FL benchmark for variant: {variant_name} =====")

    df = pd.read_csv(data_path)
    patterns_map = load_patterns(sentences_path)
    tokenizer, model = load_model(variant_name)
    lf_to_candidates = build_candidate_pool(df)

    conditions = list(product([False, True], repeat=2))  # (k, s)

    records = []
    total = sum(
        len(patterns_map[str(row["lexical_function"])]["patterns"]) * len(conditions)
        for _, row in df.iterrows()
        if str(row["lexical_function"]) in patterns_map
    )
    done = 0

    for _, row in df.iterrows():
        word = str(row["name"])
        lf = str(row["lexical_function"])
        targets = parse_target(row["targets"])
        example = row.get("example", "")

        if lf not in patterns_map:
            print(f"[WARN] No pattern found for LF '{lf}', skipping entry '{word}'")
            continue

        lf_patterns = patterns_map[lf]["patterns"]

        for pattern in lf_patterns:
            for (k, s) in conditions:
                prompt = build_prompt(
                    word=word,
                    lf=lf,
                    pattern=pattern,
                    patterns_map=patterns_map,
                    k=k,
                    s=s,
                    current_example=example
                )

                answer = predict_answer(prompt, lf, lf_to_candidates, tokenizer, model)

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
                    print(f"{done}/{total} done...")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"flaubert_{variant_name}_fl_results.csv"

    results = pd.DataFrame(records)
    results.to_csv(output_path, index=False)

    print(f"\nSaved to {output_path}")
    print_summary(results)
    return results


# ----------------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------------
def print_summary(results: pd.DataFrame):
    print("\n=== Mean EM / CM per condition ===")
    summary = (
        results.groupby(["k", "s"])[["EM", "CM"]]
        .mean()
        .reset_index()
    )
    print(summary.to_string(index=False))

    print("\n=== Mean EM / CM per LF ===")
    lf_summary = (
        results.groupby("lexical_functions")[["EM", "CM"]]
        .mean()
        .sort_values("EM", ascending=False)
    )
    print(lf_summary.to_string())


if __name__ == "__main__":
    for variant in ["base", "wiki", "fl"]:
        run_benchmark(variant)

