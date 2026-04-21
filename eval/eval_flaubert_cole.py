import os
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForMaskedLM
from peft import PeftModel

# ===== PATH CONFIG =====
BASE_MODEL = "flaubert/flaubert_base_cased"
INPUT_CSV = "data/flaubert_cole.csv"
OUTPUT_DIR = "./results/flaubert_cole"

WIKI_ADAPTER_PATH = "./models/flaubert_lora_wiki"
FL_ADAPTER_PATH = "./models/flaubert_lora_fl"

os.makedirs(OUTPUT_DIR, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def normalize_text(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def load_model(mode):
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token

    model = AutoModelForMaskedLM.from_pretrained(BASE_MODEL)

    print(f"\n[DEBUG] mode = {mode}")

    if mode == "wiki":
        print(f"[DEBUG] loading wiki adapter from: {WIKI_ADAPTER_PATH}")
        model = PeftModel.from_pretrained(model, WIKI_ADAPTER_PATH)
        model = model.merge_and_unload()
    elif mode == "fl":
        print(f"[DEBUG] loading fl adapter from: {FL_ADAPTER_PATH}")
        model = PeftModel.from_pretrained(model, FL_ADAPTER_PATH)
        model = model.merge_and_unload()
    else:
        print("[DEBUG] using base model only")

    print("[DEBUG] final model class:", type(model))

    model.to(DEVICE)
    model.eval()
    return tokenizer, model


def get_candidates_by_task(df):
    task2cands = {}
    for task, subdf in df.groupby("task_name"):
        vals = []
        for x in subdf["ground_truth"].dropna().tolist():
            s = normalize_text(x)
            if s not in vals:
                vals.append(s)
        task2cands[task] = vals
    return task2cands


@torch.no_grad()
def score_candidate_pll(prompt, candidate, tokenizer, model, max_length=512):
    """
    Robust PLL scoring for MLM:
    - Encode prompt and candidate separately
    - Concatenate token ids explicitly
    - Score only candidate token positions
    """
    prompt = normalize_text(prompt)
    candidate = normalize_text(candidate)

    if not candidate:
        return -1e9

    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
    cand_ids = tokenizer.encode(candidate, add_special_tokens=False)

    if len(cand_ids) == 0:
        return -1e9

    input_ids = prompt_ids + cand_ids
    input_ids = input_ids[:max_length]

    # candidate may be truncated
    candidate_start = min(len(prompt_ids), len(input_ids))
    candidate_positions = list(range(candidate_start, len(input_ids)))

    if len(candidate_positions) == 0:
        return -1e9

    attention_mask = [1] * len(input_ids)

    input_ids_tensor = torch.tensor(input_ids, dtype=torch.long)
    attention_mask_tensor = torch.tensor(attention_mask, dtype=torch.long)

    total_logprob = 0.0
    counted = 0

    for pos in candidate_positions:
        true_id = input_ids_tensor[pos].item()

        if true_id in tokenizer.all_special_ids:
            continue

        masked_ids = input_ids_tensor.clone()
        masked_ids[pos] = tokenizer.mask_token_id

        outputs = model(
            input_ids=masked_ids.unsqueeze(0).to(DEVICE),
            attention_mask=attention_mask_tensor.unsqueeze(0).to(DEVICE)
        )

        logits = outputs.logits[0, pos]
        log_probs = F.log_softmax(logits, dim=-1)
        total_logprob += log_probs[true_id].item()
        counted += 1

    if counted == 0:
        return -1e9

    return total_logprob / counted


@torch.no_grad()
def predict_label(prompt, task_name, task2cands, tokenizer, model, debug=False):
    candidates = task2cands.get(task_name, [])
    if not candidates:
        return ""

    best_cand = None
    best_score = -1e18

    if debug:
        print(f"\n[DEBUG] task = {task_name}")
        print(f"[DEBUG] prompt = {prompt}")
        print(f"[DEBUG] candidates = {candidates}")

    for cand in candidates:
        score = score_candidate_pll(prompt, cand, tokenizer, model)
        if debug:
            print(f"[DEBUG] candidate = {cand!r} | score = {score:.6f}")
        if score > best_score:
            best_score = score
            best_cand = cand

    return best_cand if best_cand is not None else ""


def run_mode(mode):
    print(f"\n===== Running mode: {mode} =====")
    df = pd.read_csv(INPUT_CSV)

    required_cols = {"task_name", "prompt", "ground_truth"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    task2cands = get_candidates_by_task(df)
    tokenizer, model = load_model(mode)

    predictions = []
    correctness = []

    for i, row in df.iterrows():
        task_name = normalize_text(row["task_name"])
        prompt = normalize_text(row["prompt"])
        gt = normalize_text(row["ground_truth"])

        debug = (i < 3)  # only print details for first 3 rows
        pred = predict_label(prompt, task_name, task2cands, tokenizer, model, debug=debug)

        predictions.append(pred)
        correctness.append(int(normalize_text(pred) == gt))

        if (i + 1) % 50 == 0:
            print(f"{i+1}/{len(df)} done")

    out_df = df.copy()
    out_df["prediction"] = predictions
    out_df["is_correct"] = correctness

    out_path = os.path.join(OUTPUT_DIR, f"flaubert_{mode}_cole_results.csv")
    out_df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    for mode in ["base", "wiki", "fl"]:
        run_mode(mode)
