import os
import json
import pandas as pd
import numpy as np
from collections import defaultdict
import glob

def load_human_eval_data(human_eval_dir, dataset_type):
    """
    Load human evaluation data for the given dataset type
    """
    human_eval_data = {}
    
    # Determine which directory to use based on dataset_type
    if dataset_type == "base":
        eval_dir = os.path.join(human_eval_dir, "human-eval-base")
    elif dataset_type == "masked":
        eval_dir = os.path.join(human_eval_dir, "human-eval-masked")
    elif dataset_type == "shuffled":
        eval_dir = os.path.join(human_eval_dir, "human-eval-shuffled")
    elif dataset_type == "masked_shuffled":
        eval_dir = os.path.join(human_eval_dir, "human-eval-masked-shuffled")
    else:
        print(f"Unknown dataset type: {dataset_type}")
        return {}
    
    csv_files = glob.glob(os.path.join(eval_dir, "*.csv"))
    
    for csv_file in csv_files:
        model_name = os.path.basename(csv_file).split('.')[0]  # Extract model name
        
        # Read the CSV file
        df = pd.read_csv(csv_file)
        
        # Store human evaluation scores for each row
        for idx, row in df.iterrows():
            # Check if 'id' is in the row's index and handle both 'id' and 'row{id}' formats
            if 'id' in row:
                if isinstance(row['id'], str) and row['id'].startswith('row'):
                    row_id = row['id']
                else:
                    row_id = f"row{row['id']}"
            else:
                row_id = f"row{idx}"
                
            human_eval_data[(model_name, row_id)] = {
                'cause_score': row.get('cause_score', 0),
                'effect_score': row.get('effect_score', 0),
                'avg_score': (row.get('cause_score', 0) + row.get('effect_score', 0)) / 2
            }
    
    return human_eval_data

def calculate_f1_metrics(result_dir, human_eval_dir, output_dir):
    """
    Calculate F1 metrics for all models and datasets
    """
    results_by_dataset = defaultdict(list)
    
    # Get all dataset types (base, masked, shuffled, masked_shuffled)
    dataset_types = [d for d in os.listdir(result_dir) if os.path.isdir(os.path.join(result_dir, d))]
    
    for dataset_type in dataset_types:
        print(f"Processing dataset: {dataset_type}")
        dataset_path = os.path.join(result_dir, dataset_type)
        
        # Load human evaluation data for this dataset type
        human_eval_data = load_human_eval_data(human_eval_dir, dataset_type)
        
        # Get all model directories for this dataset
        model_dirs = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
        
        for model_dir in model_dirs:
            model_path = os.path.join(dataset_path, model_dir)
            model_name = model_dir.split('-')[0]
            
            print(f"  Processing model: {model_name}")
            
            jsonl_files = glob.glob(os.path.join(model_path, "*.jsonl"))
            judgments = {}
            for jsonl_file in jsonl_files:
                judge_name = os.path.basename(jsonl_file).split('_')[1]
                
                with open(jsonl_file, 'r') as f:
                    rows = [json.loads(line) for line in f if line.strip()]
                
                # Store judgments for each row
                for row in rows:
                    row_id = row['id']
                    if row_id not in judgments:
                        judgments[row_id] = {
                            'llm_judgments': [],
                            'pred_cause': row.get('pred_cause', ""),
                            'pred_effect': row.get('pred_effect', ""),
                            'orig_cause': row.get('orig_cause', ""),
                            'orig_effect': row.get('orig_effect', ""),
                        }
                    
                    # Add this judge's scores
                    cause_score = row.get('cause_score', 0)
                    effect_score = row.get('effect_score', 0)
                    judgments[row_id]['llm_judgments'].append({
                        'judge': judge_name,
                        'cause_score': cause_score,
                        'effect_score': effect_score,
                        'avg_score': (cause_score + effect_score) / 2 if not pd.isna(cause_score) and not pd.isna(effect_score) else 0
                    })
            
            # Calculate TP, FP, TN, FN for each row
            tp, fp, tn, fn = 0, 0, 0, 0
            
            for row_id, data in judgments.items():
                # Check if LLM provided an extraction
                has_extraction = (data['pred_cause'] != "" and data['pred_effect'] != "")
                # Get human evaluation score for this model and row
                human_eval = None
                if (model_name, row_id) in human_eval_data:
                    human_eval = human_eval_data[(model_name, row_id)]['avg_score']
                else:
                    # Fall back to using LLM judgments if no human-eval available
                    cause_scores = [j['cause_score'] for j in data['llm_judgments'] if not pd.isna(j['cause_score'])]
                    effect_scores = [j['effect_score'] for j in data['llm_judgments'] if not pd.isna(j['effect_score'])]
                    
                    if cause_scores and effect_scores:
                        human_eval = (np.mean(cause_scores) + np.mean(effect_scores)) / 2
                    else:
                        human_eval = 0
                
                # Get majority score from LLM judges (>= 0.5)
                cause_votes = [j['cause_score'] >= 0.5 for j in data['llm_judgments'] if not pd.isna(j['cause_score'])]
                effect_votes = [j['effect_score'] >= 0.5 for j in data['llm_judgments'] if not pd.isna(j['effect_score'])]
                
                majority_cause = sum(cause_votes) > len(cause_votes) / 2 if cause_votes else False
                majority_effect = sum(effect_votes) > len(effect_votes) / 2 if effect_votes else False
                majority_positive = majority_cause and majority_effect
                
                # Check original data to determine if causality was present
                causality_present = not (pd.isna(data['orig_cause']) or pd.isna(data['orig_effect']) or 
                                        data['orig_cause'] == "" or data['orig_effect'] == "")
                
                # Apply TP, FP, TN, FN classification rules
                if has_extraction:
                    if human_eval >= 0.5 and majority_positive:
                        tp += 1
                    else:
                        fp += 1
                else:
                    if causality_present and human_eval > 0:
                        fn += 1
                    else:
                        tn += 1
            
            # Calculate metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # Store results for this dataset type
            results_by_dataset[dataset_type].append({
                'Model': model_name,
                'TP': tp,
                'FP': fp,
                'TN': tn,
                'FN': fn,
                'Precision': precision,
                'Recall': recall,
                'F1': f1
            })
    
    # Output results both as a combined file and separate files per dataset
    
    # Combined file
    all_results = []
    for dataset_type, results in results_by_dataset.items():
        for result in results:
            all_results.append({
                'Dataset': dataset_type,
                **result
            })
    
    combined_df = pd.DataFrame(all_results)
    os.makedirs(output_dir, exist_ok=True)
    combined_df.to_csv(os.path.join(output_dir, "f1_metrics_all.csv"), index=False)
    
    # Separate files per dataset
    for dataset_type, results in results_by_dataset.items():
        df = pd.DataFrame(results)
        df.to_csv(os.path.join(output_dir, f"f1_metrics_{dataset_type}.csv"), index=False)
    
    return results_by_dataset

def main():#
    result_dir = "runs/grades"
    human_eval_dir = "datasets/human-eval"
    output_dir = "result-summary/f1-metrics"
    
    results = calculate_f1_metrics(result_dir, human_eval_dir, output_dir)
    
    print("\nF1 Score Summary by Dataset and Model:")
    print("=" * 80)
    
    for dataset_type, results_list in results.items():
        print(f"\nDataset: {dataset_type}")
        print("-" * 80)
        print(f"{'Model':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}")
        print("-" * 80)
        
        for result in sorted(results_list, key=lambda x: x['F1'], reverse=True):
            print(f"{result['Model']:<10} {result['Precision']:.4f}     {result['Recall']:.4f}     {result['F1']:.4f}")

if __name__ == "__main__":
    main()