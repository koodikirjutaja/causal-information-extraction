import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create output directory if it doesn't exist
os.makedirs("charts", exist_ok=True)

# Set consistent visualization style parameters
plt.style.use('ggplot')
sns.set_palette("deep")
plt.rcParams.update({'font.size': 12, 'figure.figsize': (12, 8)})

def load_data():
    """
    Load and prepare all datasets needed for visualizations.
    Returns a dictionary containing all dataframes.
    """
    data = {}
    
    # Load human evaluation summaries
    data['human_base'] = pd.read_csv('result-summary/human-eval/human_eval_base_summary.csv')
    data['human_masked'] = pd.read_csv('result-summary/human-eval/human_eval_masked_summary.csv')
    data['human_shuffled'] = pd.read_csv('result-summary/human-eval/human_eval_shuffled_summary.csv')
    data['human_maskedshuffled'] = pd.read_csv('result-summary/human-eval/human_eval_masked-shuffled_summary.csv')
    
    # Load LLM evaluation summaries
    data['llm_base'] = pd.read_csv('result-summary/llm-eval/llm_eval_base_by_model.csv')
    data['llm_masked'] = pd.read_csv('result-summary/llm-eval/llm_eval_masked_by_model.csv')
    data['llm_shuffled'] = pd.read_csv('result-summary/llm-eval/llm_eval_shuffled_by_model.csv')
    data['llm_maskedshuffled'] = pd.read_csv('result-summary/llm-eval/llm_eval_masked-shuffled_by_model.csv')
    
    # Load cross-dataset comparison
    data['human_cross'] = pd.read_csv('result-summary/human-eval/human_eval_cross_dataset_comparison.csv')
    data['llm_cross'] = pd.read_csv('result-summary/llm-eval/llm_eval_cross_dataset_by_model.csv')
    
    # Load judge cross-dataset comparison
    try:
        data['llm_cross_judge'] = pd.read_csv('result-summary/llm-eval/llm_eval_cross_dataset_by_model.csv')
    except Exception as e:
        print(f"Warning: Could not load judge cross-dataset data: {e}")
    
    # Load human vs LLM comparison
    data['human_vs_llm'] = pd.read_csv('result-summary/human_vs_llm_comparison.csv')
    
    # Load detailed evaluations for agreement matrix
    data['llm_base_detailed'] = pd.read_csv('result-summary/llm-eval/llm_eval_base_detailed.csv')
    
    # Load GPQA scores - with fallback to hardcoded values if file loading fails
    try:
        gpqa_df = pd.read_csv('gpqa-model.csv')
        
        # Normalize column names
        column_mapping = {}
        for col in gpqa_df.columns:
            if col.lower() == 'model':
                column_mapping[col] = 'Model'
            if col.lower() == 'gpqa':
                column_mapping[col] = 'GPQA'
        
        if column_mapping:
            gpqa_df = gpqa_df.rename(columns=column_mapping)
        
        # If column names still don't match expected format, use first two columns
        if 'Model' not in gpqa_df.columns or 'GPQA' not in gpqa_df.columns:
            if len(gpqa_df.columns) >= 2:
                gpqa_df = gpqa_df.rename(columns={
                    gpqa_df.columns[0]: 'Model',
                    gpqa_df.columns[1]: 'GPQA'
                })
        
        data['gpqa'] = gpqa_df
    except Exception as e:
        # Fallback to hardcoded values if file loading fails
        gpqa_data = {
            'Model': ['O3', 'CLAUDE', 'GEMINI', 'LLAMA4', 'QWEN3', 'MISTRAL', 'DEEPSEEK', 'GROK3'],
            'GPQA': [0.833, 0.848, 0.84, 0.698, 0.658, 0.453, 0.715, 0.846]
        }
        data['gpqa'] = pd.DataFrame(gpqa_data)
    
    # Load F1 metrics data
    f1_metrics_dir = 'result-summary/f1-metrics'
    if os.path.exists(f1_metrics_dir):
        # Load combined F1 data
        combined_path = os.path.join(f1_metrics_dir, 'f1_metrics_all.csv')
        if os.path.exists(combined_path):
            data['f1_all'] = pd.read_csv(combined_path)
        
        # Load individual dataset F1 data
        for dataset_type in ['base', 'masked', 'shuffled', 'masked_shuffled']:
            dataset_path = os.path.join(f1_metrics_dir, f'f1_metrics_{dataset_type}.csv')
            if os.path.exists(dataset_path):
                data[f'f1_{dataset_type}'] = pd.read_csv(dataset_path)
    
    # Standardize model names to uppercase for consistent comparison
    for key in data:
        if 'Model' in data[key].columns:
            data[key]['Model'] = data[key]['Model'].str.upper()
    
    return data

def plot_cause_vs_effect(data):
    """
    Create a bar chart comparing cause vs effect extraction performance
    for each model based on human evaluation data.
    """
    df = data['human_base'].copy()
    
    plt.figure(figsize=(14, 8))
    models = df['Model']
    x = np.arange(len(models))
    width = 0.35
    
    # Plot cause and effect scores side by side
    plt.bar(x - width/2, df['Avg Cause Score'], width, label='Cause Score')
    plt.bar(x + width/2, df['Avg Effect Score'], width, label='Effect Score')
    
    # Add horizontal lines showing average performance
    cause_avg = df['Avg Cause Score'].mean()
    effect_avg = df['Avg Effect Score'].mean()
    plt.axhline(y=cause_avg, color='green', linestyle='--', alpha=0.5, label=f'Avg Cause: {cause_avg:.2f}')
    plt.axhline(y=effect_avg, color='red', linestyle='--', alpha=0.5, label=f'Avg Effect: {effect_avg:.2f}')
    
    # Add labels and formatting
    plt.xlabel('Models')
    plt.ylabel('Average Score')
    plt.title('Cause vs. Effect Extraction Performance by Model (Human Evaluation)', fontsize=16)
    plt.xticks(x, models, rotation=45)
    plt.ylim(0, 1.0)
    plt.legend()
    plt.tight_layout()
    
    # Add data labels on bars for better readability
    for i, v in enumerate(df['Avg Cause Score']):
        plt.text(i - width/2, v + 0.02, f'{v:.2f}', ha='center')
    for i, v in enumerate(df['Avg Effect Score']):
        plt.text(i + width/2, v + 0.02, f'{v:.2f}', ha='center')
    
    plt.savefig('charts/cause_vs_effect_comparison.svg', format='svg')
    plt.close()

def plot_cross_dataset_performance(data):
    """
    Create a 2x2 panel visualization showing model performance 
    across different dataset variants (base, masked, shuffled, masked-shuffled).
    """
    df = data['human_cross'].copy()
    
    # Create a figure with 4 subplots (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    axes = axes.flatten()
    
    # Identify dataset column prefixes
    datasets = []
    for prefix in ['base', 'masked', 'shuffled', 'masked-shuffled']:
        if f'{prefix}_cause' in df.columns and f'{prefix}_effect' in df.columns and f'{prefix}_total' in df.columns:
            datasets.append(prefix)
        else:
            # Try alternative name format (masked-shuffled vs masked_shuffled)
            alt_prefix = prefix.replace('-', '_')
            if f'{alt_prefix}_cause' in df.columns:
                datasets.append(alt_prefix)
    
    # Fallback if no datasets identified
    if not datasets:
        datasets = ['base', 'masked', 'shuffled', 'masked_shuffled']

    titles = ['Base Dataset', 'Masked Dataset', 'Shuffled Dataset', 'Masked-Shuffled Dataset']
    
    # Create a subplot for each dataset variant
    for i, (dataset, title) in enumerate(zip(datasets, titles)):
        cause_col = f'{dataset}_cause'
        effect_col = f'{dataset}_effect'
        total_col = f'{dataset}_total'
        
        # Sort by total score for this dataset
        df_sorted = df.sort_values(by=total_col, ascending=False).reset_index(drop=True)
        
        # Remove the 'AVERAGE' row if it exists
        df_sorted = df_sorted[df_sorted['Model'] != 'AVERAGE']
        
        # Create the subplot
        ax = axes[i]
        x = np.arange(len(df_sorted))
        width = 0.25
        
        # Plot bars for cause, effect, and total scores
        bars1 = ax.bar(x - width, df_sorted[cause_col], width, label='Cause')
        bars2 = ax.bar(x, df_sorted[effect_col], width, label='Effect')
        bars3 = ax.bar(x + width, df_sorted[total_col], width, label='Total')
        
        # Add data labels above each bar
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.2f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom',
                            fontsize=8)
        
        # Add reference line at average
        avg = df_sorted[total_col].mean()
        ax.axhline(y=avg, color='red', linestyle='--', alpha=0.7, label=f'Avg: {avg:.2f}')
        
        # Add labels and formatting
        ax.set_title(title, fontsize=14)
        ax.set_ylabel('Score')
        ax.set_ylim(0, 1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(df_sorted['Model'], rotation=45, ha='right')
        ax.legend(loc='upper right')
    
    # Add a common x-label
    fig.text(0.5, 0.01, 'Models', ha='center', fontsize=14)
    
    plt.suptitle('Model Performance Across Dataset Variants (Human Evaluation)', fontsize=18)
    plt.tight_layout()
    plt.subplots_adjust(top=0.93, bottom=0.1)
    
    plt.savefig('charts/cross_dataset_performance.svg', format='svg')
    plt.close()

def plot_human_llm_agreement(data):
    """
    Create a correlation heatmap showing agreement between
    different judges (human and LLM) on model evaluations.
    """
    df = data['llm_base_detailed'].copy()
    
    # Check if necessary columns exist
    if 'Judge' not in df.columns or 'Model' not in df.columns or 'Avg Total Score' not in df.columns:
        return
        
    try:
        # Pivot to create judge x model matrix of total scores
        pivot_df = df.pivot_table(
            index='Judge', 
            columns='Model', 
            values='Avg Total Score'
        )
        
        # Add human evaluation as an additional judge
        human_scores = data['human_base'][['Model', 'Avg Total Score']].set_index('Model').T
        human_scores.index = ['HUMAN']
        
        # Combine data frames
        agreement_matrix = pd.concat([pivot_df, human_scores])
        
        # Create correlation matrix (judge-to-judge agreement)
        corr_matrix = agreement_matrix.T.corr()
        
        # Plot correlation heatmap
        plt.figure(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Mask for upper triangle
        sns.heatmap(
            corr_matrix, 
            annot=True, 
            fmt=".2f", 
            cmap='viridis', 
            vmin=0, 
            vmax=1,
            mask=mask,
            square=True
        )
        plt.title('Judge Agreement Matrix (Pearson Correlation)', fontsize=16)
        plt.tight_layout()
        
        plt.savefig('charts/judge_agreement_matrix.svg', format='svg')
        plt.close()
    except Exception as e:
        print(f"Error creating judge agreement matrix: {e}")

def plot_cause_vs_effect_across_datasets(data):
    """
    Create a bar chart comparing cause vs effect extraction performance
    for each model averaged across ALL datasets (base, masked, shuffled, masked-shuffled).
    """
    # Ensure we have the cross-dataset data
    if 'human_cross' not in data:
        print("Error: Cross-dataset data not available for averaging across datasets")
        return
    
    df = data['human_cross'].copy()
    
    # Remove the AVERAGE row if it exists
    if 'AVERAGE' in df['Model'].values:
        df = df[df['Model'] != 'AVERAGE']
    
    # Calculate averages across all four datasets for each model
    dataset_prefixes = ['base', 'masked', 'shuffled', 'masked_shuffled']
    alt_prefixes = ['base', 'masked', 'shuffled', 'masked-shuffled']
    
    # Create columns for storing averages
    df['avg_cause'] = 0.0
    df['avg_effect'] = 0.0
    
    # For each model, calculate average cause and effect scores across all datasets
    for i, row in df.iterrows():
        cause_scores = []
        effect_scores = []
        
        # Try to get scores from each dataset variant
        for prefix in dataset_prefixes:
            cause_col = f'{prefix}_cause'
            effect_col = f'{prefix}_effect'
            
            # If columns don't exist, try alternative naming
            if cause_col not in df.columns or effect_col not in df.columns:
                for alt_prefix in alt_prefixes:
                    alt_cause_col = f'{alt_prefix}_cause'
                    alt_effect_col = f'{alt_prefix}_effect'
                    if alt_cause_col in df.columns and alt_effect_col in df.columns:
                        cause_col = alt_cause_col
                        effect_col = alt_effect_col
                        break
            
            # Add to scores if columns exist
            if cause_col in df.columns and effect_col in df.columns:
                cause_scores.append(row[cause_col])
                effect_scores.append(row[effect_col])
        
        # Calculate average if we found any scores
        if cause_scores:
            df.at[i, 'avg_cause'] = sum(cause_scores) / len(cause_scores)
        if effect_scores:
            df.at[i, 'avg_effect'] = sum(effect_scores) / len(effect_scores)
    
    # Sort by total average performance
    df['avg_total'] = (df['avg_cause'] + df['avg_effect']) / 2
    df = df.sort_values(by='avg_total', ascending=False).reset_index(drop=True)
    
    # Create the plot
    plt.figure(figsize=(14, 8))
    models = df['Model']
    x = np.arange(len(models))
    width = 0.35
    
    # Plot cause and effect scores side by side
    plt.bar(x - width/2, df['avg_cause'], width, label='Cause Score')
    plt.bar(x + width/2, df['avg_effect'], width, label='Effect Score')
    
    # Add horizontal lines showing average performance
    cause_avg = df['avg_cause'].mean()
    effect_avg = df['avg_effect'].mean()
    plt.axhline(y=cause_avg, color='green', linestyle='--', alpha=0.5, label=f'Avg Cause: {cause_avg:.2f}')
    plt.axhline(y=effect_avg, color='red', linestyle='--', alpha=0.5, label=f'Avg Effect: {effect_avg:.2f}')
    
    # Add labels and formatting
    plt.xlabel('Models')
    plt.ylabel('Average Score')
    plt.title('Cause vs. Effect Extraction Performance by Model (Averaged Across All Datasets)', fontsize=16)
    plt.xticks(x, models, rotation=45)
    plt.ylim(0, 1.0)
    plt.legend()
    plt.tight_layout()
    
    # Add data labels on bars for better readability
    for i, v in enumerate(df['avg_cause']):
        plt.text(i - width/2, v + 0.02, f'{v:.2f}', ha='center')
    for i, v in enumerate(df['avg_effect']):
        plt.text(i + width/2, v + 0.02, f'{v:.2f}', ha='center')
    
    plt.savefig('charts/cause_vs_effect_across_datasets.svg', format='svg')
    plt.close()
    print("✓ Created Cause vs. Effect Comparison (Averaged Across All Datasets) chart")

def plot_gpqa_correlation(data):
    """
    Create a scatter plot with regression line showing the correlation
    between GPQA benchmark scores and causal extraction performance.
    """
    df = data['human_base'].copy()
    gpqa_df = data['gpqa'].copy()
    
    # Filter out any non-model rows
    if 'AVERAGE' in df['Model'].values:
        df = df[df['Model'] != 'AVERAGE']
    
    # Ensure case-insensitive merging
    df['Model'] = df['Model'].str.upper()
    gpqa_df['Model'] = gpqa_df['Model'].str.upper()
    
    # Use the Avg Total Score for performance
    df['Performance'] = df['Avg Total Score']
    
    try:
        # Merge datasets to align GPQA scores with performance data
        merged_df = pd.merge(df, gpqa_df, on='Model')
        
        # Create scatter plot
        plt.figure(figsize=(10, 8))
        
        # Plot points
        plt.scatter(
            merged_df['GPQA'], 
            merged_df['Performance'], 
            s=100,
            label='Model Performance',
            color='blue'
        )
        
        # Add regression line with confidence interval
        sns.regplot(
            x='GPQA', 
            y='Performance', 
            data=merged_df,
            scatter=False,  # Don't plot points again
            line_kws={"color": "red", "label": "Regression Line"},
            ci=95  # 95% confidence interval
        )
        
        # Add explanatory text for the shaded area
        plt.text(0.05, 0.15, 
                "Red shaded area: 95% confidence interval\naround regression line", 
                transform=plt.gca().transAxes,
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5'))
        
        # Calculate and display correlation coefficient
        correlation = merged_df['GPQA'].corr(merged_df['Performance'])
        plt.annotate(
            f"Pearson Correlation: {correlation:.2f}",
            xy=(0.05, 0.95),
            xycoords='axes fraction',
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
        )
        
        # Sort models to better manage label positioning
        sorted_df = merged_df.sort_values('GPQA')
        
        # Track label positions to prevent overlap
        label_positions = {}
        
        # Add model labels to points with adjusted positions
        for idx, row in sorted_df.iterrows():
            model = row['Model']
            x_pos = row['GPQA']
            y_pos = row['Performance']
            
            # Default offsets
            x_offset = 5
            y_offset = 5
            
            # Check for nearby labels and adjust position to prevent overlap
            for other_model, (other_x, other_y) in label_positions.items():
                if abs(x_pos - other_x) < 0.05 and abs(y_pos - other_y) < 0.05:
                    if len(label_positions) % 2 == 0:
                        y_offset = -15  # Place below
                    else:
                        x_offset = -40  # Place to the left
            
            # Place label at adjusted position
            plt.annotate(
                model,
                (x_pos, y_pos),
                xytext=(x_offset, y_offset),
                textcoords='offset points',
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.7)
            )
            
            # Record position for overlap detection
            label_positions[model] = (x_pos, y_pos)
        
        # Add legend and labels
        plt.legend(loc='lower right')
        plt.xlabel('GPQA Score (Benchmark Performance)', fontsize=12)
        plt.ylabel('Causal Extraction Performance (Human Evaluation)', fontsize=12)
        plt.title('GPQA Benchmark vs. Causal Extraction Performance', fontsize=16)
        plt.ylim(0, 1)
        plt.xlim(0.4, 0.9)
        plt.tight_layout()
        
        plt.savefig('charts/gpqa_performance_correlation.svg', format='svg')
        plt.close()
    except Exception as e:
        print(f"Error creating GPQA correlation chart: {e}")

def plot_f1_scores(data):
    """
    Create visualizations of precision, recall, and F1 scores:
    1. Average metrics across all datasets
    2. Multi-panel comparison across dataset variants
    """
    # Check if F1 data is available
    if 'f1_all' not in data:
        return
    
    try:
        # 1. Create chart showing average metrics across all datasets
        if 'f1_all' in data:
            # Group by Model and calculate the mean of Precision, Recall, and F1
            avg_metrics = data['f1_all'].groupby('Model')[['Precision', 'Recall', 'F1']].mean().reset_index()
            
            plt.figure(figsize=(16, 10))
            
            # Sort models by F1 score
            avg_metrics = avg_metrics.sort_values(by='F1', ascending=False).reset_index(drop=True)
            
            models = avg_metrics['Model']
            x = np.arange(len(models))
            width = 0.25
            
            # Plot bars for each metric
            plt.bar(x - width, avg_metrics['Precision'], width, label='Precision', color='#1f77b4')
            plt.bar(x, avg_metrics['Recall'], width, label='Recall', color='#ff7f0e')
            plt.bar(x + width, avg_metrics['F1'], width, label='F1 Score', color='#2ca02c')
            
            # Add data labels on bars
            for i, v in enumerate(avg_metrics['Precision']):
                plt.text(i - width, v + 0.02, f'{v:.2f}', ha='center', va='bottom', fontsize=9)
            for i, v in enumerate(avg_metrics['Recall']):
                plt.text(i, v + 0.02, f'{v:.2f}', ha='center', va='bottom', fontsize=9)
            for i, v in enumerate(avg_metrics['F1']):
                plt.text(i + width, v + 0.02, f'{v:.2f}', ha='center', va='bottom', fontsize=9)
            
            # Add labels and formatting
            plt.xlabel('Models', fontsize=14)
            plt.ylabel('Score', fontsize=14)
            plt.title('Average Precision, Recall, and F1 Scores by Model (Across All Datasets)', fontsize=16)
            plt.xticks(x, models, rotation=45, ha='right')
            plt.ylim(0, 1.1)  # Leave room for data labels
            plt.legend(loc='lower right', bbox_to_anchor=(0.98, 0.15))
            plt.tight_layout()
            
            plt.savefig('charts/precision_recall_f1_all.svg', format='svg')
            plt.close()
        
        # 2. Create a multi-panel figure showing metrics for each dataset variant
        fig, axes = plt.subplots(2, 2, figsize=(24, 20))
        axes = axes.flatten()
        
        dataset_titles = {
            'base': 'Base Dataset',
            'masked': 'Masked Dataset',
            'shuffled': 'Shuffled Dataset',
            'masked_shuffled': 'Masked-Shuffled Dataset'
        }
        
        # Create a subplot for each dataset variant
        for i, dataset_type in enumerate(['base', 'masked', 'shuffled', 'masked_shuffled']):
            if f'f1_{dataset_type}' in data:
                ax = axes[i]
                
                # Sort models by F1 score
                df = data[f'f1_{dataset_type}'].sort_values(by='F1', ascending=False).reset_index(drop=True)
                
                models = df['Model']
                x = np.arange(len(models))
                width = 0.25
                
                # Plot bars for each metric
                bars1 = ax.bar(x - width, df['Precision'], width, label='Precision', color='#1f77b4')
                bars2 = ax.bar(x, df['Recall'], width, label='Recall', color='#ff7f0e')
                bars3 = ax.bar(x + width, df['F1'], width, label='F1 Score', color='#2ca02c')
                
                # Add data labels for all metrics
                for j, v in enumerate(df['Precision']):
                    ax.text(j - width, v + 0.02, f'{v:.2f}', ha='center', va='bottom', fontsize=9, rotation=0)
                for j, v in enumerate(df['Recall']):
                    ax.text(j, v + 0.02, f'{v:.2f}', ha='center', va='bottom', fontsize=9, rotation=0)
                for j, v in enumerate(df['F1']):
                    ax.text(j + width, v + 0.02, f'{v:.2f}', ha='center', va='bottom', fontsize=9, rotation=0)
                
                # Add labels and formatting
                ax.set_title(dataset_titles.get(dataset_type, dataset_type.capitalize()), fontsize=18)
                ax.set_ylabel('Score', fontsize=14)
                ax.set_ylim(0, 1.1)
                ax.set_xticks(x)
                ax.set_xticklabels(models, rotation=45, ha='right', fontsize=12)
                ax.legend(loc='lower right', bbox_to_anchor=(0.98, 0.15))
        
        # Add overall title and adjust layout
        plt.suptitle('Precision, Recall, and F1 Scores Across Dataset Variants', fontsize=22)
        plt.subplots_adjust(
            top=0.93,
            bottom=0.05,
            left=0.05,
            right=0.95,
            hspace=0.2,
            wspace=0.15
        )
        
        plt.savefig('charts/precision_recall_f1_all_datasets.svg', format='svg')
        plt.close()
        
    except Exception as e:
        print(f"Error creating F1 score charts: {e}")

def plot_judge_cause_effect_comparison(data):
    """
    Create a visualization comparing cause vs effect extraction performance
    across all datasets for each judge model.
    """
    # Load the cross-dataset by judge data if not already in data dictionary
    if 'llm_cross_judge' not in data:
        try:
            judge_df = pd.read_csv('result-summary/llm-eval/llm_eval_cross_dataset_by_judge.csv')
            data['llm_cross_judge'] = judge_df
        except Exception as e:
            print(f"Error loading judge cross-dataset data: {e}")
            return
    
    judge_df = data['llm_cross_judge'].copy()
    
    # Remove the AVERAGE row if it exists
    if 'AVERAGE' in judge_df['Judge'].values:
        judge_df = judge_df[judge_df['Judge'] != 'AVERAGE']
    
    # Create multi-panel figure (2x2 for the 4 datasets)
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    axes = axes.flatten()
    
    # Define dataset variants and their titles
    dataset_variants = ['base', 'masked', 'shuffled', 'masked-shuffled']
    titles = ['Base Dataset', 'Masked Dataset', 'Shuffled Dataset', 'Masked-Shuffled Dataset']
    
    # Process each dataset variant
    for i, (variant, title) in enumerate(zip(dataset_variants, titles)):
        ax = axes[i]
        
        # Get cause and effect columns for this variant
        cause_col = f"{variant}_cause"
        effect_col = f"{variant}_effect"
        
        # Ensure these columns exist in the dataframe
        if cause_col not in judge_df.columns or effect_col not in judge_df.columns:
            # Try alternative naming (with underscore instead of hyphen)
            alt_variant = variant.replace('-', '_')
            cause_col = f"{alt_variant}_cause"
            effect_col = f"{alt_variant}_effect"
            
            if cause_col not in judge_df.columns or effect_col not in judge_df.columns:
                print(f"Could not find data for {variant} dataset")
                continue
        
        # Sort judges by average of cause and effect scores
        judge_df['avg_score'] = (judge_df[cause_col] + judge_df[effect_col]) / 2
        df_sorted = judge_df.sort_values(by='avg_score', ascending=False).reset_index(drop=True)
        
        # Plot the data
        x = np.arange(len(df_sorted))
        width = 0.35
        
        # Create bars
        bars1 = ax.bar(x - width/2, df_sorted[cause_col], width, label='Cause Score')
        bars2 = ax.bar(x + width/2, df_sorted[effect_col], width, label='Effect Score')
        
        # Add data labels on bars
        for j, v in enumerate(df_sorted[cause_col]):
            ax.text(j - width/2, v + 0.02, f'{v:.2f}', ha='center', fontsize=9)
        for j, v in enumerate(df_sorted[effect_col]):
            ax.text(j + width/2, v + 0.02, f'{v:.2f}', ha='center', fontsize=9)
        
        # Calculate and display average lines
        cause_avg = df_sorted[cause_col].mean()
        effect_avg = df_sorted[effect_col].mean()
        ax.axhline(y=cause_avg, color='blue', linestyle='--', alpha=0.5, label=f'Avg Cause: {cause_avg:.2f}')
        ax.axhline(y=effect_avg, color='orange', linestyle='--', alpha=0.5, label=f'Avg Effect: {effect_avg:.2f}')
        
        # Add title and labels
        ax.set_title(title, fontsize=16)
        ax.set_xlabel('Judge Models')
        ax.set_ylabel('Score')
        ax.set_ylim(0, 1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(df_sorted['Judge'], rotation=45, ha='right')
        ax.legend()
    
    # Add overall title
    plt.suptitle('Cause vs. Effect Extraction Performance by Judge Models Across Datasets', fontsize=20)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    # Save the figure
    plt.savefig('charts/judge_cause_effect_comparison.svg', format='svg')
    plt.close()
    
    # Create a tabular summary visualization
    plot_judge_cause_effect_table(data)


def plot_judge_cause_effect_table(data):
    """
    Create a tabular visualization showing the difference between cause and effect scores
    for each judge across all dataset variants.
    """
    judge_df = data['llm_cross_judge'].copy()
    
    # Calculate differences between cause and effect for each dataset variant
    variants = ['base', 'masked', 'shuffled', 'masked-shuffled']
    alt_variants = ['base', 'masked', 'shuffled', 'masked_shuffled']
    
    # Create a new dataframe for the differences
    diff_data = {'Judge': judge_df['Judge']}
    
    for variant, alt_variant in zip(variants, alt_variants):
        cause_col = f"{variant}_cause"
        effect_col = f"{variant}_effect"
        
        # Check if columns exist, if not try alternative naming
        if cause_col not in judge_df.columns or effect_col not in judge_df.columns:
            cause_col = f"{alt_variant}_cause"
            effect_col = f"{alt_variant}_effect"
        
        # Calculate absolute difference and store
        if cause_col in judge_df.columns and effect_col in judge_df.columns:
            diff_data[variant] = (judge_df[effect_col] - judge_df[cause_col]).abs()
    
    diff_df = pd.DataFrame(diff_data)
    
    # Calculate average difference across all variants for sorting
    diff_df['avg_diff'] = diff_df[[v for v in variants if v in diff_df.columns]].mean(axis=1)
    diff_df = diff_df.sort_values(by='avg_diff', ascending=True)
    
    # Drop the average column after sorting
    diff_df = diff_df.drop(columns=['avg_diff'])
    
    # Create the figure and table
    fig, ax = plt.figure(figsize=(12, 8)), plt.subplot(111)
    ax.axis('off')
    ax.axis('tight')
    
    # Create table with cell coloring based on value
    table = ax.table(
        cellText=diff_df.drop(columns=['Judge']).applymap(lambda x: f'{x:.3f}').values,
        rowLabels=diff_df['Judge'],
        colLabels=[v.replace('-', '-\n') for v in variants if v in diff_df.columns],
        loc='center',
        cellLoc='center',
        colColours=['#C9D7F0']*len([v for v in variants if v in diff_df.columns]),
        rowColours=['#E3E3E3']*len(diff_df)
    )
    
    # Adjust table appearance
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Color cells based on value (lighter color for smaller differences)
    for i in range(len(diff_df)):
        for j in range(len([v for v in variants if v in diff_df.columns])):
            cell = table[i+1, j]  # +1 for header row
            value = diff_df.iloc[i, j+1]  # +1 for Judge column
            
            # Color based on value (red intensity increases with larger differences)
            # Lower values are better (less difference between cause and effect)
            # Max expected difference is around 0.4, so scale accordingly
            intensity = min(value / 0.4, 1.0)  # Scale to [0, 1]
            cell.set_facecolor((1.0, 1.0 - intensity * 0.8, 1.0 - intensity * 0.8))
    
    plt.title('Cause vs. Effect Score Absolute Difference by Judge and Dataset', fontsize=16)
    plt.figtext(0.5, 0.01, 'Lower values indicate more balanced cause-effect extraction', 
                ha='center', fontsize=10, style='italic')
    
    plt.tight_layout()
    plt.savefig('charts/judge_cause_effect_difference_table.svg', format='svg')
    plt.close()



def create_all_charts():
    """
    Main function that loads data and generates all visualization charts.
    """
    try:
        data = load_data()
        
        # Generate all charts
        plot_cause_vs_effect(data)
        plot_cause_vs_effect_across_datasets(data)
        plot_cross_dataset_performance(data)
        plot_human_llm_agreement(data)
        plot_gpqa_correlation(data)
        plot_f1_scores(data)
        plot_judge_cause_effect_comparison(data)
        
        print("All charts saved to the 'charts' directory")
    except Exception as e:
        print(f"Error in create_all_charts: {e}")

if __name__ == "__main__":
    create_all_charts()