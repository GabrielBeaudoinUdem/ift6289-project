import argparse
import json
import hashlib
from pathlib import Path

def generate_report(results_dir: Path):
    if not results_dir.exists():
        print(f"Erreur : Le répertoire {results_dir} est introuvable. Avez-vous lancé l'évaluation COLE ?")
        return

    # Discover all results files
    all_files = list(results_dir.glob("*_cole_results*"))
    
    # Check for duplicate files
    file_hashes = {}
    duplicates = []
    for f in all_files:
        if f.suffix == '.csv':
            with open(f, 'rb') as rb:
                h = hashlib.md5(rb.read()).hexdigest()
                if h in file_hashes:
                    duplicates.append((f.name, file_hashes[h]))
                else:
                    file_hashes[h] = f.name
    
    if duplicates:
        print("ATTENTION : Des fichiers de résultats identiques ont été détectés :")
        for f1, f2 in duplicates:
            print(f"  - {f1} est identique à {f2}")
        print("Cela peut fausser la comparaison entre les modèles.\n")

    # Group files by their base name (ignoring extension)
    file_groups = {}
    for f in all_files:
        if f.suffix in ['.json', '.csv']:
            file_groups.setdefault(f.stem, {})[f.suffix] = f

    if not file_groups:
        print(f"Aucun résultat COLE trouvé dans {results_dir}.")
        return

    print(f"Analyse de {len(file_groups)} modèles COLE trouvés...")

    task_scores = {}
    model_totals = {}
    model_names = []

    for stem in sorted(file_groups.keys()):
        group = file_groups[stem]
        json_file = group.get('.json')
        csv_file = group.get('.csv')
        
        data = {}
        if json_file:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        
        if csv_file:
            import pandas as pd
            df = pd.read_csv(csv_file)
            
            # If JSON is missing, compute data from CSV
            if not json_file:
                if 'task_name' in df.columns and 'is_correct' in df.columns:
                    task_means = df.groupby('task_name')['is_correct'].mean().to_dict()
                    data['task_scores'] = task_means
                    if task_means:
                        data['cole_score'] = sum(task_means.values()) / len(task_means)
                    # Clean model name
                    m_name = stem.replace('_cole_results', '')
                    # If it has a timestamp like model_20260410_..., strip it for the display name if possible
                    m_name_clean = '_'.join(m_name.split('_')[:-2]) if '_' in m_name and m_name.split('_')[-2].isdigit() else m_name
                    data['model_name'] = m_name_clean
                else:
                    print(f"    -> Skipping {stem}: missing 'task_name' or 'is_correct' in CSV")
                    continue
            
            # Recompute fracas score logic
            if 'task_name' in df.columns and 'fracas' in df['task_name'].values:
                fracas_mask = df['task_name'] == 'fracas'
                
                def fix_fracas(row):
                    ans = str(row['ground_truth']).lower()
                    if "yes" in ans:
                        expected = "0"
                    elif ans.startswith("no"):
                        expected = "2"
                    else:
                        expected = "1"
                    return 1 if str(row['prediction']).strip() == expected else 0
                    
                new_correct = df[fracas_mask].apply(fix_fracas, axis=1)
                
                needs_update = False
                if 'is_correct' in df.columns:
                    needs_update = not df.loc[fracas_mask, 'is_correct'].equals(new_correct)
                else:
                    needs_update = True
                    
                if needs_update:
                    df.loc[fracas_mask, 'is_correct'] = new_correct
                    df.to_csv(csv_file, index=False)
                    
                    new_fracas_score = float(new_correct.mean())
                    if 'task_scores' not in data:
                        data['task_scores'] = {}
                    data['task_scores']['fracas'] = new_fracas_score
                    
                    scores = list(data['task_scores'].values())
                    data['cole_score'] = sum(scores) / len(scores)
                        
                    if json_file:
                        with open(json_file, 'w', encoding='utf-8') as f:
                            json.dump(data, f, indent=2, ensure_ascii=False)
                    print(f"    -> Recomputed fracas score for {stem}: {new_fracas_score:.2%}")

        model_name = data.get("model_name", stem)
        if model_name not in model_names:
            model_names.append(model_name)
            
        scores = data.get("task_scores", {})
        for task, score in scores.items():
            if task not in task_scores:
                task_scores[task] = {}
            task_scores[task][model_name] = score
            
        model_totals[model_name] = data.get("cole_score", 0.0)

    # Sort models by total COLE SCORE descending
    model_names.sort(key=lambda m: model_totals.get(m, 0.0), reverse=True)

    # Build Markdown
    report = []

    # Table Header
    header = "| Tâche COLE | " + " | ".join(model_names) + " |"
    sep = "| :--- | " + " | ".join([":---:" for _ in model_names]) + " |"
    report.append(header)
    report.append(sep)
    
    # Rows for each Task
    for task in sorted(task_scores.keys()):
        row_str = f"| **{task}** | "
        cols = []
        for model in model_names:
            score = task_scores[task].get(model, None)
            if score is not None:
                # Convert to percentage
                cols.append(f"{score:.2%}")
            else:
                cols.append("-")
        row_str += " | ".join(cols) + " |"
        report.append(row_str)
        
    # Total row
    tot_str = "| **SCORE COLE (TOTAL)** | "
    tot_cols = []
    for m in model_names:
        tot = model_totals.get(m, 0.0)
        tot_cols.append(f"**{tot:.2%}**")
            
    tot_str += " | ".join(tot_cols) + " |"
    report.append(tot_str)

    output_file = results_dir / "cole_aggregated_report.md"
    with open(output_file, "w") as f:
        f.write("\n".join(report))
        
    print(f"Rapport sauvegardé avec succès dans : {output_file}")
    return "\n".join(report)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate COLE Report from JSON results")
    parser.add_argument("--dir", type=str, default="cole_benchmark", help="Dossier (relatif à la racine du projet) contenant les JSONs de résultats")
    args = parser.parse_args()
    
    target_dir = Path(__file__).resolve().parent.parent / args.dir
    result = generate_report(target_dir)
