#!/usr/bin/env python3
"""
Consolidate LLM evaluation results into summary tables.

This script processes judgment files with the following structure:
runs/grades/[dataset_type]/[model]-grades/score_[judge]__[model]__results.jsonl

Usage:
python consolidate_grades.py --input runs/grades --output results_summary
"""

import os
import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

# Define the dataset types
DATASET_TYPES = ["base", "masked", "shuffled", "masked_shuffled"]

def read_jsonl_scores(file_path: Path) -> Tuple[float, int]:
    """Read a JSONL file and return the average score and sample count."""
    if not file_path.exists():
        return 0.0, 0
    
    total_score = 0.0
    count = 0
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        if "avg_score" in data:
                            total_score += data["avg_score"]
                            count += 1
                        elif "cause_score" in data and "effect_score" in data:
                            # Calculate average if avg_score is not available
                            total_score += (data["cause_score"] + data["effect_score"]) / 2
                            count += 1
                    except json.JSONDecodeError:
                        print(f"Error parsing line in {file_path}")
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
    
    if count > 0:
        return total_score / count, count
    else:
        return 0.0, 0

def find_judge_model_from_filename(filename: str) -> Tuple[str, str]:
    """Extract judge and model from filename like score_judge__model__results.jsonl."""
    parts = filename.split('_', 1)[1].split('__')
    if len(parts) >= 2:
        return parts[0], parts[1]
    return "", ""

def process_dataset(grades_dir: Path, dataset_type: str) -> pd.DataFrame:
    """Process all files for a dataset type and create a summary DataFrame."""
    # Dictionary to store results
    results = {
        "Judge": [],
        "Model": [],
        "Avg Score": [],
        "Sample Count": []
    }
    
    dataset_path = grades_dir / dataset_type
    if not dataset_path.exists():
        print(f"Dataset directory not found: {dataset_path}")
        return pd.DataFrame()
    
    print(f"Processing dataset: {dataset_type}")
    
    # Iterate through all subdirectories (model-grades folders)
    for model_dir in dataset_path.iterdir():
        if not model_dir.is_dir():
            continue
        
        print(f"  Checking model directory: {model_dir.name}")
        
        # Find all score files
        for score_file in model_dir.glob("score_*__*__results.jsonl"):
            judge, model = find_judge_model_from_filename(score_file.name)
            
            if judge and model:
                avg_score, count = read_jsonl_scores(score_file)
                
                if count > 0:
                    results["Judge"].append(judge)
                    results["Model"].append(model)
                    results["Avg Score"].append(avg_score)
                    results["Sample Count"].append(count)
                    print(f"    Found: Judge={judge}, Model={model}, Avg={avg_score:.3f}, Count={count}")
    
    # Create DataFrame
    df = pd.DataFrame(results)
    return df

def create_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Create a pivot table with judges as rows and models as columns."""
    if df.empty:
        return pd.DataFrame()
    
    # Create a pivot table
    pivot_df = df.pivot(index="Judge", columns="Model", values="Avg Score")
    
    # Add a row with average score for each model
    model_avgs = []
    for model in pivot_df.columns:
        model_avgs.append(pivot_df[model].mean())
    
    avg_row = pd.Series(model_avgs, index=pivot_df.columns, name="MODEL AVERAGE")
    pivot_df = pd.concat([pivot_df, avg_row.to_frame().T])
    
    # Add a column with average score for each judge
    pivot_df["JUDGE AVERAGE"] = pivot_df.mean(axis=1)
    
    # Round to 3 decimal places for readability
    pivot_df = pivot_df.round(3)
    
    return pivot_df

def create_composite_tables(all_data: Dict[str, pd.DataFrame], output_dir: Path):
    """Create composite tables that show performance across dataset types."""
    # Combine all data with dataset type
    combined_df = pd.DataFrame()
    
    for dataset_type, df in all_data.items():
        if not df.empty:
            temp_df = df.copy()
            temp_df["Dataset"] = dataset_type
            combined_df = pd.concat([combined_df, temp_df])
    
    if combined_df.empty:
        print("No data found across all datasets")
        return
    
    # Save the combined data
    combined_df.to_csv(output_dir / "all_results.csv", index=False)
    print(f"Saved all results to {output_dir / 'all_results.csv'}")
    
    # Create a table comparing model performance across datasets
    model_compare = combined_df.groupby(["Dataset", "Model"])["Avg Score"].mean().reset_index()
    model_pivot = model_compare.pivot(index="Model", columns="Dataset", values="Avg Score")
    
    # Add average row
    avg_row = pd.Series(model_pivot.mean(), name="AVERAGE")
    model_pivot = pd.concat([model_pivot, avg_row.to_frame().T])
    
    # Add difference columns to show degradation
    if "base" in model_pivot.columns:
        for col in model_pivot.columns:
            if col != "base":
                model_pivot[f"{col}_diff"] = model_pivot[col] - model_pivot["base"]
    
    model_pivot = model_pivot.round(3)
    model_pivot.to_csv(output_dir / "model_performance_comparison.csv")
    print(f"Saved model comparison to {output_dir / 'model_performance_comparison.csv'}")
    
    # Create a table comparing judge consistency across datasets
    judge_compare = combined_df.groupby(["Dataset", "Judge"])["Avg Score"].mean().reset_index()
    judge_pivot = judge_compare.pivot(index="Judge", columns="Dataset", values="Avg Score")
    
    # Add average row
    avg_row = pd.Series(judge_pivot.mean(), name="AVERAGE")
    judge_pivot = pd.concat([judge_pivot, avg_row.to_frame().T])
    
    judge_pivot = judge_pivot.round(3)
    judge_pivot.to_csv(output_dir / "judge_consistency_comparison.csv")
    print(f"Saved judge comparison to {output_dir / 'judge_consistency_comparison.csv'}")
    
    # Create a more detailed view with judge-model combinations across datasets
    detailed_pivot = combined_df.pivot_table(
        index=["Judge", "Model"],
        columns="Dataset",
        values="Avg Score",
        aggfunc="mean"
    ).round(3)
    
    detailed_pivot.to_csv(output_dir / "detailed_cross_dataset_comparison.csv")
    print(f"Saved detailed comparison to {output_dir / 'detailed_cross_dataset_comparison.csv'}")
    
    # Create a single comprehensive table with judge as rows and models as columns,
    # stacking the different datasets
    comprehensive = pd.DataFrame()
    
    for dataset in DATASET_TYPES:
        if dataset in all_data and not all_data[dataset].empty:
            # Create a pivot table for this dataset
            dataset_pivot = all_data[dataset].pivot(
                index="Judge", 
                columns="Model", 
                values="Avg Score"
            ).round(3)
            
            # Rename columns to include dataset
            dataset_pivot = dataset_pivot.rename(
                columns={col: f"{col}_{dataset}" for col in dataset_pivot.columns}
            )
            
            # Add to the comprehensive table
            if comprehensive.empty:
                comprehensive = dataset_pivot
            else:
                # Merge on index (Judge)
                comprehensive = comprehensive.join(dataset_pivot, how="outer")
    
    if not comprehensive.empty:
        comprehensive.to_csv(output_dir / "comprehensive_summary.csv")
        print(f"Saved comprehensive summary to {output_dir / 'comprehensive_summary.csv'}")

def main():
    parser = argparse.ArgumentParser(description="Consolidate LLM evaluation results.")
    parser.add_argument("--input", type=str, default="runs/grades", help="Input directory containing grade files")
    parser.add_argument("--output", type=str, default="results_summary", help="Output directory for summary tables")
    args = parser.parse_args()
    
    grades_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"Processing grades from {grades_dir}")
    print(f"Output will be saved to {output_dir}")
    
    all_data = {}
    
    # Process each dataset type
    for dataset_type in DATASET_TYPES:
        dataset_df = process_dataset(grades_dir, dataset_type)
        all_data[dataset_type] = dataset_df
        
        # Create summary table
        if not dataset_df.empty:
            summary_table = create_summary_table(dataset_df)
            summary_file = output_dir / f"{dataset_type}_summary.csv"
            summary_table.to_csv(summary_file)
            print(f"Created summary for {dataset_type} dataset at {summary_file}")
            
            # Save raw data
            raw_file = output_dir / f"{dataset_type}_raw_data.csv"
            dataset_df.to_csv(raw_file, index=False)
            print(f"Saved raw data for {dataset_type} dataset at {raw_file}")
    
    # Create composite tables
    create_composite_tables(all_data, output_dir)
    
    print(f"All summaries have been saved to {output_dir}")

if __name__ == "__main__":
    main()