#!/usr/bin/env python3
"""
Summarize evaluation results from both human evaluations and LLM evaluations.
Creates summary tables for each evaluation type and dataset.

Usage:
python summarize_eval_results.py
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import json
import re

# Configuration
BASE_DIR = Path("/Users/sandorvunk/PycharmProjects/Thesis")
HUMAN_EVAL_DIR = BASE_DIR / "datasets/human-eval"
LLM_EVAL_DIR = BASE_DIR / "runs/grades"
OUTPUT_DIR = BASE_DIR / "result-summary"

# Create output directories
(OUTPUT_DIR / "llm-eval").mkdir(exist_ok=True, parents=True)
(OUTPUT_DIR / "human-eval").mkdir(exist_ok=True, parents=True)

# Models and dataset types
MODELS = ["claude", "deepseek", "gemini", "grok3", "llama4", "mistral", "o3", "qwen3"]
DATASET_TYPES = ["base", "masked", "shuffled", "masked-shuffled"]
JUDGES = ["claude", "deepseek", "gemini", "grok", "llama4", "mistral", "openai", "qwen3"]

def summarize_human_evaluations():
    """Summarize human evaluation results"""
    print("\nSummarizing human evaluations...")
    
    # Final results dictionary to store all summaries
    all_results = {
        dataset_type: {
            "by_model": pd.DataFrame(),
            "overall": {}
        } for dataset_type in DATASET_TYPES
    }
    
    # Process each dataset type
    for dataset_type in DATASET_TYPES:
        print(f"Processing dataset type: {dataset_type}")
        eval_dir = HUMAN_EVAL_DIR / f"human-eval-{dataset_type}"
        
        if not eval_dir.exists():
            print(f"  Warning: Directory not found: {eval_dir}")
            continue
        
        # Collect results for all models
        model_results = []
        
        for model in MODELS:
            file_path = eval_dir / f"{model}.csv"
            if not file_path.exists():
                print(f"  Warning: File not found: {file_path}")
                continue
            
            try:
                # Read CSV file
                df = pd.read_csv(file_path)
                
                # Skip if no records or no scores
                if len(df) == 0:
                    print(f"  Warning: No data in {file_path}")
                    continue
                
                # Convert score columns to numeric, coercing any non-numeric values to NaN
                if "cause_score" in df.columns and "effect_score" in df.columns:
                    df["cause_score"] = pd.to_numeric(df["cause_score"], errors="coerce")
                    df["effect_score"] = pd.to_numeric(df["effect_score"], errors="coerce")
                    
                    # Calculate averages only for rows with scores
                    scored_df = df.dropna(subset=["cause_score", "effect_score"])
                    
                    if len(scored_df) > 0:
                        avg_cause = scored_df["cause_score"].mean()
                        avg_effect = scored_df["effect_score"].mean()
                        avg_total = (avg_cause + avg_effect) / 2
                        
                        # Calculate count of non-NaN values
                        count_cause = scored_df["cause_score"].count()
                        count_effect = scored_df["effect_score"].count()
                        
                        # Add to results
                        model_results.append({
                            "Model": model,
                            "Avg Cause Score": round(avg_cause, 3) if not np.isnan(avg_cause) else 0,
                            "Avg Effect Score": round(avg_effect, 3) if not np.isnan(avg_effect) else 0,
                            "Avg Total Score": round(avg_total, 3) if not np.isnan(avg_total) else 0,
                            "Count Cause": count_cause,
                            "Count Effect": count_effect
                        })
                        
                        print(f"  Processed {model}: Cause={round(avg_cause, 3)}, "
                              f"Effect={round(avg_effect, 3)}, "
                              f"Total={round(avg_total, 3)}, "
                              f"Count={count_cause}")
                    else:
                        print(f"  Warning: No scored rows in {file_path}")
                else:
                    print(f"  Warning: Missing score columns in {file_path}")
                      
            except Exception as e:
                print(f"  Error processing {file_path}: {e}")
        
        if model_results:
            # Create DataFrame from results
            df_results = pd.DataFrame(model_results)
            
            # Sort by total score descending
            df_results = df_results.sort_values("Avg Total Score", ascending=False)
            
            # Save to CSV
            output_file = OUTPUT_DIR / "human-eval" / f"human_eval_{dataset_type}_summary.csv"
            df_results.to_csv(output_file, index=False)
            print(f"  Saved summary to {output_file}")
            
            # Calculate overall averages
            overall = {
                "avg_cause": df_results["Avg Cause Score"].mean(),
                "avg_effect": df_results["Avg Effect Score"].mean(),
                "avg_total": df_results["Avg Total Score"].mean()
            }
            
            # Store in results dictionary
            all_results[dataset_type]["by_model"] = df_results
            all_results[dataset_type]["overall"] = overall
        else:
            print(f"  No results found for {dataset_type}")
    
    # Create cross-dataset comparison table
    cross_dataset_rows = []
    for model in MODELS:
        row = {"Model": model}
        
        for dataset_type in DATASET_TYPES:
            if dataset_type in all_results and not all_results[dataset_type]["by_model"].empty:
                model_data = all_results[dataset_type]["by_model"]
                model_row = model_data[model_data["Model"] == model]
                
                if not model_row.empty:
                    row[f"{dataset_type}_cause"] = model_row["Avg Cause Score"].iloc[0]
                    row[f"{dataset_type}_effect"] = model_row["Avg Effect Score"].iloc[0]
                    row[f"{dataset_type}_total"] = model_row["Avg Total Score"].iloc[0]
                else:
                    row[f"{dataset_type}_cause"] = np.nan
                    row[f"{dataset_type}_effect"] = np.nan
                    row[f"{dataset_type}_total"] = np.nan
            else:
                row[f"{dataset_type}_cause"] = np.nan
                row[f"{dataset_type}_effect"] = np.nan
                row[f"{dataset_type}_total"] = np.nan
        
        cross_dataset_rows.append(row)
    
    if cross_dataset_rows:
        # Create DataFrame for cross-dataset comparison
        cross_df = pd.DataFrame(cross_dataset_rows)
        
        # Add averages row
        avg_row = {"Model": "AVERAGE"}
        for col in cross_df.columns:
            if col != "Model":
                avg_row[col] = cross_df[col].mean()
        
        cross_df = pd.concat([cross_df, pd.DataFrame([avg_row])], ignore_index=True)
        
        # Save cross-dataset comparison
        output_file = OUTPUT_DIR / "human-eval" / "human_eval_cross_dataset_comparison.csv"
        cross_df.to_csv(output_file, index=False)
        print(f"Saved cross-dataset comparison to {output_file}")
    
    return all_results

def read_jsonl_file(file_path):
    """Read a JSONL file and return data as a list of dictionaries."""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    print(f"  Warning: Could not parse line in {file_path}")
    except FileNotFoundError:
        print(f"  Warning: File not found: {file_path}")
    except Exception as e:
        print(f"  Error reading {file_path}: {e}")
    
    return data

def extract_scores_from_jsonl(file_path):
    """Extract scores from a JSONL file."""
    scores = []
    data = read_jsonl_file(file_path)
    
    for item in data:
        # Try to find scores in different formats
        if "cause_score" in item and "effect_score" in item:
            cause_score = item.get("cause_score")
            effect_score = item.get("effect_score")
            avg_score = item.get("avg_score")
            
            # If avg_score is not available, calculate from cause and effect
            if avg_score is None and cause_score is not None and effect_score is not None:
                avg_score = (cause_score + effect_score) / 2
                
            scores.append({
                "cause_score": cause_score,
                "effect_score": effect_score,
                "avg_score": avg_score,
                "id": item.get("id", "")
            })
        elif "judge_response" in item:
            # Try to parse judge_response
            try:
                judge_response = json.loads(item["judge_response"])
                cause_score = judge_response.get("cause_score")
                effect_score = judge_response.get("effect_score")
                
                if cause_score is not None and effect_score is not None:
                    avg_score = (cause_score + effect_score) / 2
                    
                    scores.append({
                        "cause_score": cause_score,
                        "effect_score": effect_score,
                        "avg_score": avg_score,
                        "id": item.get("id", "")
                    })
            except:
                print(f"  Warning: Could not parse judge_response in line from {file_path}")
    
    return scores

def find_jsonl_files(directory, judge, model):
    """Find all JSONL files for a specific judge and model."""
    pattern = f"score_{judge}__{model}__results.jsonl"
    return list(directory.glob(pattern))

def summarize_llm_evaluations():
    """Summarize LLM evaluation results."""
    print("\nSummarizing LLM evaluations...")
    
    # Final results dictionary
    all_results = {
        dataset_type: {
            "by_model_judge": pd.DataFrame(),
            "by_model": pd.DataFrame(),
            "by_judge": pd.DataFrame(),
            "overall": {}
        } for dataset_type in DATASET_TYPES
    }
    
    # Process each dataset type
    for dataset_type in DATASET_TYPES:
        print(f"Processing dataset type: {dataset_type}")
        
        # Handle the different directory structure for masked_shuffled
        ds_type = dataset_type.replace("-", "_")
        search_dir = LLM_EVAL_DIR / ds_type
        
        if not search_dir.exists():
            print(f"  Warning: Directory not found: {search_dir}")
            continue
        
        # Model-judge pairs and scores
        model_judge_scores = []
        
        # We need to handle different directory structures
        # 1. Standard structure: grades/{dataset}/model-{dataset}-grades/score_judge__model__results.jsonl
        # 2. Alternative structure: grades/{dataset}/model/score_judge__model__results.jsonl
        
        # Check for both directory patterns
        grade_dirs = []
        
        # Check pattern 1: model-{dataset}-grades
        model_grades_dirs = list(search_dir.glob("*-grades"))
        if model_grades_dirs:
            grade_dirs.extend(model_grades_dirs)
        
        # Check pattern 2: model names as directories
        model_dirs = [search_dir / model for model in MODELS if (search_dir / model).exists()]
        if model_dirs:
            grade_dirs.extend(model_dirs)
        
        # Also check for renamed directories like "llama" instead of "llama4"
        possible_renames = {
            "grok": "grok3",
            "llama": "llama4",
            "qwen": "qwen3"
        }
        
        for dir_name, model_name in possible_renames.items():
            if (search_dir / dir_name).exists():
                grade_dirs.append(search_dir / dir_name)
        
        if not grade_dirs:
            print(f"  Warning: No grade directories found in {search_dir}")
            continue
        
        # Process each grading directory
        for grade_dir in grade_dirs:
            # Try to extract model name from directory
            model_match = re.search(r'(\w+)(?:-grades|-\w+)?$', grade_dir.name)
            model = model_match.group(1) if model_match else grade_dir.name
            
            # Map shortened model names to full names
            if model in possible_renames.keys():
                for short_name, full_name in possible_renames.items():
                    if model == short_name:
                        model = full_name
                        break
            
            # Process all JSONL files in this directory
            for jsonl_file in grade_dir.glob("score_*__*__results.jsonl"):
                # Extract judge and graded model from filename
                parts = jsonl_file.stem.split("__")
                if len(parts) >= 2:
                    # If filename pattern is score_judge__model__results.jsonl
                    judge = parts[0].replace("score_", "")
                    graded_model = parts[1]
                    
                    # Extract scores from JSONL
                    scores = extract_scores_from_jsonl(jsonl_file)
                    
                    if scores:
                        # Calculate averages
                        avg_cause = np.mean([s["cause_score"] for s in scores if s["cause_score"] is not None])
                        avg_effect = np.mean([s["effect_score"] for s in scores if s["effect_score"] is not None])
                        avg_total = np.mean([s["avg_score"] for s in scores if s["avg_score"] is not None])
                        
                        # Count of scores
                        count_cause = sum(1 for s in scores if s["cause_score"] is not None)
                        count_effect = sum(1 for s in scores if s["effect_score"] is not None)
                        count_total = sum(1 for s in scores if s["avg_score"] is not None)
                        
                        # Add to results
                        model_judge_scores.append({
                            "Judge": judge,
                            "Model": graded_model,
                            "Avg Cause Score": round(avg_cause, 3) if not np.isnan(avg_cause) else 0,
                            "Avg Effect Score": round(avg_effect, 3) if not np.isnan(avg_effect) else 0,
                            "Avg Total Score": round(avg_total, 3) if not np.isnan(avg_total) else 0,
                            "Count Cause": count_cause,
                            "Count Effect": count_effect,
                            "Count Total": count_total
                        })
                        
                        print(f"  Processed {judge} judging {graded_model}: "
                              f"Total={round(avg_total, 3) if not np.isnan(avg_total) else 'N/A'}, "
                              f"Count={count_total}")
        
        if model_judge_scores:
            # Create DataFrame from results
            df_results = pd.DataFrame(model_judge_scores)
            
            # Save detailed results
            output_file = OUTPUT_DIR / "llm-eval" / f"llm_eval_{dataset_type}_detailed.csv"
            df_results.to_csv(output_file, index=False)
            print(f"  Saved detailed results to {output_file}")
            
            # Aggregate by model (average across judges)
            model_agg = df_results.groupby("Model").agg({
                "Avg Cause Score": "mean",
                "Avg Effect Score": "mean",
                "Avg Total Score": "mean",
                "Count Cause": "sum",
                "Count Effect": "sum",
                "Count Total": "sum"
            }).reset_index()
            
            # Sort by total score descending
            model_agg = model_agg.sort_values("Avg Total Score", ascending=False)
            
            # Save model aggregation
            output_file = OUTPUT_DIR / "llm-eval" / f"llm_eval_{dataset_type}_by_model.csv"
            model_agg.to_csv(output_file, index=False)
            print(f"  Saved model aggregation to {output_file}")
            
            # Aggregate by judge (average across models)
            judge_agg = df_results.groupby("Judge").agg({
                "Avg Cause Score": "mean",
                "Avg Effect Score": "mean",
                "Avg Total Score": "mean",
                "Count Cause": "sum",
                "Count Effect": "sum",
                "Count Total": "sum"
            }).reset_index()
            
            # Save judge aggregation
            output_file = OUTPUT_DIR / "llm-eval" / f"llm_eval_{dataset_type}_by_judge.csv"
            judge_agg.to_csv(output_file, index=False)
            print(f"  Saved judge aggregation to {output_file}")
            
            # Overall averages
            overall = {
                "avg_cause": df_results["Avg Cause Score"].mean(),
                "avg_effect": df_results["Avg Effect Score"].mean(),
                "avg_total": df_results["Avg Total Score"].mean()
            }
            
            # Store in results dictionary
            all_results[dataset_type]["by_model_judge"] = df_results
            all_results[dataset_type]["by_model"] = model_agg
            all_results[dataset_type]["by_judge"] = judge_agg
            all_results[dataset_type]["overall"] = overall
        else:
            print(f"  No results found for {dataset_type}")
    
    # Create cross-dataset comparison tables
    
    # By model
    cross_dataset_model_rows = []
    for model in MODELS:
        row = {"Model": model}
        
        for dataset_type in DATASET_TYPES:
            if dataset_type in all_results and not all_results[dataset_type]["by_model"].empty:
                model_data = all_results[dataset_type]["by_model"]
                model_row = model_data[model_data["Model"] == model]
                
                if not model_row.empty:
                    row[f"{dataset_type}_cause"] = model_row["Avg Cause Score"].iloc[0]
                    row[f"{dataset_type}_effect"] = model_row["Avg Effect Score"].iloc[0]
                    row[f"{dataset_type}_total"] = model_row["Avg Total Score"].iloc[0]
                else:
                    row[f"{dataset_type}_cause"] = np.nan
                    row[f"{dataset_type}_effect"] = np.nan
                    row[f"{dataset_type}_total"] = np.nan
            else:
                row[f"{dataset_type}_cause"] = np.nan
                row[f"{dataset_type}_effect"] = np.nan
                row[f"{dataset_type}_total"] = np.nan
        
        cross_dataset_model_rows.append(row)
    
    if cross_dataset_model_rows:
        # Create DataFrame for cross-dataset model comparison
        cross_model_df = pd.DataFrame(cross_dataset_model_rows)
        
        # Add averages row
        avg_row = {"Model": "AVERAGE"}
        for col in cross_model_df.columns:
            if col != "Model":
                avg_row[col] = cross_model_df[col].mean()
        
        cross_model_df = pd.concat([cross_model_df, pd.DataFrame([avg_row])], ignore_index=True)
        
        # Save cross-dataset model comparison
        output_file = OUTPUT_DIR / "llm-eval" / "llm_eval_cross_dataset_by_model.csv"
        cross_model_df.to_csv(output_file, index=False)
        print(f"Saved cross-dataset model comparison to {output_file}")
    
    # By judge
    cross_dataset_judge_rows = []
    for judge in JUDGES:
        row = {"Judge": judge}
        
        for dataset_type in DATASET_TYPES:
            if dataset_type in all_results and not all_results[dataset_type]["by_judge"].empty:
                judge_data = all_results[dataset_type]["by_judge"]
                judge_row = judge_data[judge_data["Judge"] == judge]
                
                if not judge_row.empty:
                    row[f"{dataset_type}_cause"] = judge_row["Avg Cause Score"].iloc[0]
                    row[f"{dataset_type}_effect"] = judge_row["Avg Effect Score"].iloc[0]
                    row[f"{dataset_type}_total"] = judge_row["Avg Total Score"].iloc[0]
                else:
                    row[f"{dataset_type}_cause"] = np.nan
                    row[f"{dataset_type}_effect"] = np.nan
                    row[f"{dataset_type}_total"] = np.nan
            else:
                row[f"{dataset_type}_cause"] = np.nan
                row[f"{dataset_type}_effect"] = np.nan
                row[f"{dataset_type}_total"] = np.nan
        
        cross_dataset_judge_rows.append(row)
    
    if cross_dataset_judge_rows:
        # Create DataFrame for cross-dataset judge comparison
        cross_judge_df = pd.DataFrame(cross_dataset_judge_rows)
        
        # Add averages row
        avg_row = {"Judge": "AVERAGE"}
        for col in cross_judge_df.columns:
            if col != "Judge":
                avg_row[col] = cross_judge_df[col].mean()
        
        cross_judge_df = pd.concat([cross_judge_df, pd.DataFrame([avg_row])], ignore_index=True)
        
        # Save cross-dataset judge comparison
        output_file = OUTPUT_DIR / "llm-eval" / "llm_eval_cross_dataset_by_judge.csv"
        cross_judge_df.to_csv(output_file, index=False)
        print(f"Saved cross-dataset judge comparison to {output_file}")
    
    return all_results

def create_comparison_summaries(human_results, llm_results):
    """Create comparison tables between human and LLM evaluations"""
    print("\nCreating comparison summaries...")
    
    # Create comparison table for overall scores
    comparison_rows = []
    
    for dataset_type in DATASET_TYPES:
        if dataset_type in human_results and dataset_type in llm_results:
            human_overall = human_results[dataset_type].get("overall", {})
            llm_overall = llm_results[dataset_type].get("overall", {})
            
            if human_overall and llm_overall:
                row = {
                    "Dataset": dataset_type,
                    "Human Avg Cause": human_overall.get("avg_cause", np.nan),
                    "LLM Avg Cause": llm_overall.get("avg_cause", np.nan),
                    "Cause Diff": human_overall.get("avg_cause", np.nan) - llm_overall.get("avg_cause", np.nan),
                    "Human Avg Effect": human_overall.get("avg_effect", np.nan),
                    "LLM Avg Effect": llm_overall.get("avg_effect", np.nan),
                    "Effect Diff": human_overall.get("avg_effect", np.nan) - llm_overall.get("avg_effect", np.nan),
                    "Human Avg Total": human_overall.get("avg_total", np.nan),
                    "LLM Avg Total": llm_overall.get("avg_total", np.nan),
                    "Total Diff": human_overall.get("avg_total", np.nan) - llm_overall.get("avg_total", np.nan)
                }
                comparison_rows.append(row)
    
    if comparison_rows:
        # Create DataFrame for comparison
        comparison_df = pd.DataFrame(comparison_rows)
        
        # Save comparison
        output_file = OUTPUT_DIR / "human_vs_llm_comparison.csv"
        comparison_df.to_csv(output_file, index=False)
        print(f"Saved human vs LLM comparison to {output_file}")
    else:
        print("No data available for human vs LLM comparison")

def main():
    """Main function to generate summaries"""
    print("Starting evaluation summary generation...")
    
    # Process human evaluations
    human_results = summarize_human_evaluations()
    
    # Process LLM evaluations
    llm_results = summarize_llm_evaluations()
    
    # Create comparison summaries
    create_comparison_summaries(human_results, llm_results)
    
    print("\nDone! All summaries have been generated.")

if __name__ == "__main__":
    main()