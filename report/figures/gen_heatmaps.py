import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
import os

def parse_md_table(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    headers = [h.strip() for h in lines[0].split('|') if h.strip()]
    rows = []
    for line in lines[2:]:
        if line.strip():
            rows.append([c.strip() for c in line.split('|') if c.strip()])
    return headers, rows

# Standard Model Order
MODEL_ORDER = [
    "flaubert_base", "flaubert_wiki", "flaubert_fl",
    "qwen_base", "qwen_lora_wiki", "qwen_lora_fl",
    "mistral_base", "mistral_lora_wiki", "mistral_lora_fl"
]

MODEL_DISPLAY_NAMES = {
    "flaubert_base": "FlauBERT Base",
    "flaubert_wiki": "FlauBERT Wiki",
    "flaubert_fl": "FlauBERT LF",
    "qwen_base": "Qwen2.5 Base",
    "qwen_lora_wiki": "Qwen2.5 Wiki",
    "qwen_lora_fl": "Qwen2.5 LF",
    "mistral_base": "Mistral Base",
    "mistral_lora_wiki": "Mistral Wiki",
    "mistral_lora_fl": "Mistral LF"
}

def clean_val(v):
    v = re.sub(r'\*\*', '', v)
    v = v.replace('%', '')
    try:
        return float(v)
    except ValueError:
        return 0.0

def create_heatmap(headers, rows, output_path, title, figsize=(10, 8)):
    # Find mapping
    col_mapping = []
    for model in MODEL_ORDER:
        found = False
        for i, h in enumerate(headers):
            if model in h or h in model:
                col_mapping.append(i)
                found = True
                break
        if not found:
            col_mapping.append(None)

    data = []
    y_labels = []
    for row in rows:
        row_name = re.sub(r'\*\*', '', row[0])
        y_labels.append(row_name)
        row_vals = []
        for idx in col_mapping:
            if idx is not None and idx < len(row):
                row_vals.append(clean_val(row[idx]))
            else:
                row_vals.append(0.0)
        data.append(row_vals)

    df = pd.DataFrame(data, columns=[MODEL_DISPLAY_NAMES[m] for m in MODEL_ORDER], index=y_labels)

    plt.figure(figsize=figsize)
    sns.heatmap(df, annot=True, fmt=".1f", cmap="YlGnBu", cbar=False)
    
    for i, label in enumerate(y_labels):
        if "TOTAL" in label.upper() or "SCORE" in label.upper():
            plt.axhline(i, color='black', lw=3)
            
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Heatmap saved to {output_path}")

# COLE
cole_headers, cole_rows = parse_md_table('../../eval/cole_benchmark/cole_aggregated_report.md')
create_heatmap(cole_headers, cole_rows, 'cole_heatmap.png', 'Performance Heatmap: COLE Benchmark', figsize=(12, 10))

# FL
fl_headers, fl_rows = parse_md_table('../../eval/fl_benchmark/fl_aggregated_report.md')
# FL has 37 rows, needs more vertical space
create_heatmap(fl_headers, fl_rows, 'fl_heatmap.png', 'Performance Heatmap: Lexical Functions (FL)', figsize=(12, 16))
