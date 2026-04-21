import pandas as pd
import argparse
from pathlib import Path
import ast
import re

def clean_answer(ans):
    if pd.isna(ans):
        return ""
    ans = str(ans).lower().strip()
    # Remove Qwen chat template hallucination
    if ans.endswith("assistant"):
        ans = ans[:-len("assistant")].strip()
    return ans

def recompute_metrics(df):
    if 'targets' not in df.columns:
        print("    -> Skipping recompute (missing 'targets' column)")
        return df

    ans_col = 'answer' if 'answer' in df.columns else 'prediction'
    if ans_col not in df.columns:
        print(f"    -> Skipping recompute (no 'answer' or 'prediction' column)")
        return df

    ems, cms = [], []
    for _, row in df.iterrows():
        try:
            targets = ast.literal_eval(row['targets'])
        except:
            targets = [str(row['targets']).strip().lower()]
        
        ans = clean_answer(row[ans_col])
        
        # Exact Match (EM)
        em = 1 if ans in targets else 0
            
        # Contain match (CM) "comme du monde"
        cm = 0
        for t in targets:
            pattern = r'\b' + re.escape(t) + r'\b'
            if re.search(pattern, ans):
                cm = 1
                break
        
        # Fallback to naive substring if regex failed 
        if cm == 0 and any(t in ans for t in targets):
            cm = 1
            
        ems.append(em)
        cms.append(cm)
        
    df['EM'] = ems  
    df['CM'] = cms  
    print(f"    -> Recomputed: mean EM {df['EM'].mean():.3f}, mean CM {df['CM'].mean():.3f}")
    return df

def generate_report(csv_dir: Path):
    if not csv_dir.exists():
        print(f"Error: {csv_dir} not found.")
        return

    csv_files = list(csv_dir.glob("*.csv"))
    if not csv_files:
        print(f"Error: No CSV files found in {csv_dir}.")
        return

    print(f"Processing {len(csv_files)} files from {csv_dir}...")
    
    lf_scores = {}
    model_totals = {}
    model_names = []

    for csv_file in sorted(csv_files):
        print(f"  {csv_file.name}")
        df = pd.read_csv(csv_file)
        model_name = csv_file.stem
        model_names.append(model_name)
        
        # Recompute correct metrics
        df = recompute_metrics(df)
        
        # Ensure EM and CM columns exist for grouping
        if 'CM' not in df.columns:
            df['CM'] = 0
        if 'EM' not in df.columns:
            df['EM'] = 0
            
        df.to_csv(csv_file, index=False)
        
        # Calculate per-LF CM mean
        lf_means = df.groupby('lexical_functions')['CM'].mean()
        
        for lf, score in lf_means.items():
            if lf not in lf_scores:
                lf_scores[lf] = {}
            lf_scores[lf][model_name] = score
            
        model_totals[model_name] = df['CM'].mean()

    # Sort models by total Contain Match descending
    model_names.sort(key=lambda m: model_totals[m], reverse=True)

    # Build Markdown
    report = []

    # Table Header
    header = "| Fonction Lexicale | " + " | ".join(model_names) + " |"
    sep = "| :--- | " + " | ".join([":---:" for _ in model_names]) + " |"
    report.append(header)
    report.append(sep)
    
    # Rows for each Lexical Function
    for lf in sorted(lf_scores.keys()):
        row_str = f"| **{lf}** | "
        scores = []
        for model in model_names:
            score = lf_scores[lf].get(model, 0.0)
            scores.append(f"{score:.1%}")
        row_str += " | ".join(scores) + " |"
        report.append(row_str)
        
    # Total row
    tot_str = "| **TOTAL (Moyenne)** | "
    tot_scores = [f"**{model_totals[m]:.1%}**" for m in model_names]
    tot_str += " | ".join(tot_scores) + " |"
    report.append(tot_str)

    output_file = csv_dir / "fl_aggregated_report.md"
    with open(output_file, "w") as f:
        f.write("\n".join(report))
        
    print(f"Aggregated report saved to {output_file}")
    return "\n".join(report)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate FL Report")
    parser.add_argument("--dir", type=str, default="fl_benchmark", help="Directory containing CSVs")
    args = parser.parse_args()
    
    target_dir = Path(__file__).resolve().parent.parent / args.dir
    result = generate_report(target_dir)
