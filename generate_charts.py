import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("charts", exist_ok=True)
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
    
    data['human_cross'] = pd.read_csv('result-summary/human-eval/human_eval_cross_dataset_comparison.csv')
    data['llm_cross'] = pd.read_csv('result-summary/llm-eval/llm_eval_cross_dataset_by_model.csv')
    
    try:
        data['llm_cross_judge'] = pd.read_csv('result-summary/llm-eval/llm_eval_cross_dataset_by_model.csv')
    except Exception as e:
        print(f"Warning: Could not load judge cross-dataset data: {e}")
    
    data['human_vs_llm'] = pd.read_csv('result-summary/human_vs_llm_comparison.csv')
    
    data['llm_base_detailed'] = pd.read_csv('result-summary/llm-eval/llm_eval_base_detailed.csv')
    
    try:
        gpqa_df = pd.read_csv('gpqa-model.csv')
        
        column_mapping = {}
        for col in gpqa_df.columns:
            if col.lower() == 'model':
                column_mapping[col] = 'Model'
            if col.lower() == 'gpqa':
                column_mapping[col] = 'GPQA'
        
        if column_mapping:
            gpqa_df = gpqa_df.rename(columns=column_mapping)
        
        if 'Model' not in gpqa_df.columns or 'GPQA' not in gpqa_df.columns:
            if len(gpqa_df.columns) >= 2:
                gpqa_df = gpqa_df.rename(columns={
                    gpqa_df.columns[0]: 'Model',
                    gpqa_df.columns[1]: 'GPQA'
                })
        
        data['gpqa'] = gpqa_df
    except Exception as e:
        gpqa_data = {
            'Model': ['O3', 'CLAUDE', 'GEMINI', 'LLAMA4', 'QWEN3', 'MISTRAL', 'DEEPSEEK', 'GROK3'],
            'GPQA': [0.833, 0.848, 0.84, 0.698, 0.658, 0.453, 0.715, 0.846]
        }
        data['gpqa'] = pd.DataFrame(gpqa_data)
    
    f1_metrics_dir = 'result-summary/f1-metrics'
    if os.path.exists(f1_metrics_dir):
        combined_path = os.path.join(f1_metrics_dir, 'f1_metrics_all.csv')
        if os.path.exists(combined_path):
            data['f1_all'] = pd.read_csv(combined_path)
        
        for dataset_type in ['base', 'masked', 'shuffled', 'masked_shuffled']:
            dataset_path = os.path.join(f1_metrics_dir, f'f1_metrics_{dataset_type}.csv')
            if os.path.exists(dataset_path):
                data[f'f1_{dataset_type}'] = pd.read_csv(dataset_path)
    
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
    
    plt.bar(x - width/2, df['Avg Cause Score'], width, label='Cause Score')
    plt.bar(x + width/2, df['Avg Effect Score'], width, label='Effect Score')
    
    cause_avg = df['Avg Cause Score'].mean()
    effect_avg = df['Avg Effect Score'].mean()
    plt.axhline(y=cause_avg, color='green', linestyle='--', alpha=0.5, label=f'Avg Cause: {cause_avg:.2f}')
    plt.axhline(y=effect_avg, color='red', linestyle='--', alpha=0.5, label=f'Avg Effect: {effect_avg:.2f}')
    
    plt.xlabel('Models')
    plt.ylabel('Average Score')
    plt.title('Cause vs. Effect Extraction Performance by Model (Human Evaluation)', fontsize=16)
    plt.xticks(x, models, rotation=45)
    plt.ylim(0, 1.0)
    plt.legend()
    plt.tight_layout()
    
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
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    axes = axes.flatten()
    
    datasets = []
    for prefix in ['base', 'masked', 'shuffled', 'masked-shuffled']:
        if f'{prefix}_cause' in df.columns and f'{prefix}_effect' in df.columns and f'{prefix}_total' in df.columns:
            datasets.append(prefix)
        else:
            alt_prefix = prefix.replace('-', '_')
            if f'{alt_prefix}_cause' in df.columns:
                datasets.append(alt_prefix)
    
    if not datasets:
        datasets = ['base', 'masked', 'shuffled', 'masked_shuffled']

    titles = ['Base Dataset', 'Masked Dataset', 'Shuffled Dataset', 'Masked-Shuffled Dataset']
    
    for i, (dataset, title) in enumerate(zip(datasets, titles)):
        cause_col = f'{dataset}_cause'
        effect_col = f'{dataset}_effect'
        total_col = f'{dataset}_total'
        
        df_sorted = df.sort_values(by=total_col, ascending=False).reset_index(drop=True)
        
        df_sorted = df_sorted[df_sorted['Model'] != 'AVERAGE']
        
        ax = axes[i]
        x = np.arange(len(df_sorted))
        width = 0.25
        
        bars1 = ax.bar(x - width, df_sorted[cause_col], width, label='Cause')
        bars2 = ax.bar(x, df_sorted[effect_col], width, label='Effect')
        bars3 = ax.bar(x + width, df_sorted[total_col], width, label='Total')
        
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.2f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom',
                            fontsize=8)
        
        avg = df_sorted[total_col].mean()
        ax.axhline(y=avg, color='red', linestyle='--', alpha=0.7, label=f'Avg: {avg:.2f}')
        
        ax.set_title(title, fontsize=14)
        ax.set_ylabel('Score')
        ax.set_ylim(0, 1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(df_sorted['Model'], rotation=45, ha='right')
        ax.legend(loc='upper right')
    
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
        pivot_df = df.pivot_table(
            index='Judge', 
            columns='Model', 
            values='Avg Total Score'
        )
        
        human_scores = data['human_base'][['Model', 'Avg Total Score']].set_index('Model').T
        human_scores.index = ['HUMAN']
        
        agreement_matrix = pd.concat([pivot_df, human_scores])
        
        corr_matrix = agreement_matrix.T.corr()
        
        plt.figure(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
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
    if 'human_cross' not in data:
        print("Error: Cross-dataset data not available for averaging across datasets")
        return
    
    df = data['human_cross'].copy()
    
    if 'AVERAGE' in df['Model'].values:
        df = df[df['Model'] != 'AVERAGE']
    
    dataset_prefixes = ['base', 'masked', 'shuffled', 'masked_shuffled']
    alt_prefixes = ['base', 'masked', 'shuffled', 'masked-shuffled']
    
    df['avg_cause'] = 0.0
    df['avg_effect'] = 0.0
    
    for i, row in df.iterrows():
        cause_scores = []
        effect_scores = []
        
        for prefix in dataset_prefixes:
            cause_col = f'{prefix}_cause'
            effect_col = f'{prefix}_effect'
            
            if cause_col not in df.columns or effect_col not in df.columns:
                for alt_prefix in alt_prefixes:
                    alt_cause_col = f'{alt_prefix}_cause'
                    alt_effect_col = f'{alt_prefix}_effect'
                    if alt_cause_col in df.columns and alt_effect_col in df.columns:
                        cause_col = alt_cause_col
                        effect_col = alt_effect_col
                        break
            
            if cause_col in df.columns and effect_col in df.columns:
                cause_scores.append(row[cause_col])
                effect_scores.append(row[effect_col])
        
        if cause_scores:
            df.at[i, 'avg_cause'] = sum(cause_scores) / len(cause_scores)
        if effect_scores:
            df.at[i, 'avg_effect'] = sum(effect_scores) / len(effect_scores)
    
    df['avg_total'] = (df['avg_cause'] + df['avg_effect']) / 2
    df = df.sort_values(by='avg_total', ascending=False).reset_index(drop=True)
    
    plt.figure(figsize=(14, 8))
    models = df['Model']
    x = np.arange(len(models))
    width = 0.35
    
    plt.bar(x - width/2, df['avg_cause'], width, label='Cause Score')
    plt.bar(x + width/2, df['avg_effect'], width, label='Effect Score')
    
    cause_avg = df['avg_cause'].mean()
    effect_avg = df['avg_effect'].mean()
    plt.axhline(y=cause_avg, color='green', linestyle='--', alpha=0.5, label=f'Avg Cause: {cause_avg:.2f}')
    plt.axhline(y=effect_avg, color='red', linestyle='--', alpha=0.5, label=f'Avg Effect: {effect_avg:.2f}')
    
    plt.xlabel('Models')
    plt.ylabel('Average Score')
    plt.title('Cause vs. Effect Extraction Performance by Model (Averaged Across All Datasets)', fontsize=16)
    plt.xticks(x, models, rotation=45)
    plt.ylim(0, 1.0)
    plt.legend()
    plt.tight_layout()
    
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
    
    if 'AVERAGE' in df['Model'].values:
        df = df[df['Model'] != 'AVERAGE']
    
    df['Model'] = df['Model'].str.upper()
    gpqa_df['Model'] = gpqa_df['Model'].str.upper()
    
    df['Performance'] = df['Avg Total Score']
    
    try:
        merged_df = pd.merge(df, gpqa_df, on='Model')
        plt.figure(figsize=(10, 8))
        plt.scatter(
            merged_df['GPQA'], 
            merged_df['Performance'], 
            s=100,
            label='Model Performance',
            color='blue'
        )
        
        sns.regplot(
            x='GPQA', 
            y='Performance', 
            data=merged_df,
            scatter=False,
            line_kws={"color": "red", "label": "Regression Line"},
            ci=95
        )
        
        plt.text(0.05, 0.15, 
                "Red shaded area: 95% confidence interval\naround regression line", 
                transform=plt.gca().transAxes,
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5'))
        
        correlation = merged_df['GPQA'].corr(merged_df['Performance'])
        plt.annotate(
            f"Pearson Correlation: {correlation:.2f}",
            xy=(0.05, 0.95),
            xycoords='axes fraction',
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
        )
        
        sorted_df = merged_df.sort_values('GPQA')
        
        label_positions = {}
        
        for idx, row in sorted_df.iterrows():
            model = row['Model']
            x_pos = row['GPQA']
            y_pos = row['Performance']
            
            x_offset = 5
            y_offset = 5
            
            for other_model, (other_x, other_y) in label_positions.items():
                if abs(x_pos - other_x) < 0.05 and abs(y_pos - other_y) < 0.05:
                    if len(label_positions) % 2 == 0:
                        y_offset = -15
                    else:
                        x_offset = -40
            
            plt.annotate(
                model,
                (x_pos, y_pos),
                xytext=(x_offset, y_offset),
                textcoords='offset points',
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.7)
            )
            
            label_positions[model] = (x_pos, y_pos)
        
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
    if 'f1_all' not in data:
        return
    
    try:
        if 'f1_all' in data:
            avg_metrics = data['f1_all'].groupby('Model')[['Precision', 'Recall', 'F1']].mean().reset_index()
            
            plt.figure(figsize=(16, 10))
            
            avg_metrics = avg_metrics.sort_values(by='F1', ascending=False).reset_index(drop=True)
            
            models = avg_metrics['Model']
            x = np.arange(len(models))
            width = 0.25
            
            plt.bar(x - width, avg_metrics['Precision'], width, label='Precision', color='#1f77b4')
            plt.bar(x, avg_metrics['Recall'], width, label='Recall', color='#ff7f0e')
            plt.bar(x + width, avg_metrics['F1'], width, label='F1 Score', color='#2ca02c')
            
            for i, v in enumerate(avg_metrics['Precision']):
                plt.text(i - width, v + 0.02, f'{v:.2f}', ha='center', va='bottom', fontsize=9)
            for i, v in enumerate(avg_metrics['Recall']):
                plt.text(i, v + 0.02, f'{v:.2f}', ha='center', va='bottom', fontsize=9)
            for i, v in enumerate(avg_metrics['F1']):
                plt.text(i + width, v + 0.02, f'{v:.2f}', ha='center', va='bottom', fontsize=9)
            
            plt.xlabel('Models', fontsize=14)
            plt.ylabel('Score', fontsize=14)
            plt.title('Average Precision, Recall, and F1 Scores by Model (Across All Datasets)', fontsize=16)
            plt.xticks(x, models, rotation=45, ha='right')
            plt.ylim(0, 1.1)  # Leave room for data labels
            plt.legend(loc='lower right', bbox_to_anchor=(0.98, 0.15))
            plt.tight_layout()
            
            plt.savefig('charts/precision_recall_f1_all.svg', format='svg')
            plt.close()
        
        fig, axes = plt.subplots(2, 2, figsize=(24, 20))
        axes = axes.flatten()
        
        dataset_titles = {
            'base': 'Base Dataset',
            'masked': 'Masked Dataset',
            'shuffled': 'Shuffled Dataset',
            'masked_shuffled': 'Masked-Shuffled Dataset'
        }
        
        for i, dataset_type in enumerate(['base', 'masked', 'shuffled', 'masked_shuffled']):
            if f'f1_{dataset_type}' in data:
                ax = axes[i]
                df = data[f'f1_{dataset_type}'].sort_values(by='F1', ascending=False).reset_index(drop=True)
                
                models = df['Model']
                x = np.arange(len(models))
                width = 0.25

                bars1 = ax.bar(x - width, df['Precision'], width, label='Precision', color='#1f77b4')
                bars2 = ax.bar(x, df['Recall'], width, label='Recall', color='#ff7f0e')
                bars3 = ax.bar(x + width, df['F1'], width, label='F1 Score', color='#2ca02c')
                
                for j, v in enumerate(df['Precision']):
                    ax.text(j - width, v + 0.02, f'{v:.2f}', ha='center', va='bottom', fontsize=9, rotation=0)
                for j, v in enumerate(df['Recall']):
                    ax.text(j, v + 0.02, f'{v:.2f}', ha='center', va='bottom', fontsize=9, rotation=0)
                for j, v in enumerate(df['F1']):
                    ax.text(j + width, v + 0.02, f'{v:.2f}', ha='center', va='bottom', fontsize=9, rotation=0)
                
                ax.set_title(dataset_titles.get(dataset_type, dataset_type.capitalize()), fontsize=18)
                ax.set_ylabel('Score', fontsize=14)
                ax.set_ylim(0, 1.1)
                ax.set_xticks(x)
                ax.set_xticklabels(models, rotation=45, ha='right', fontsize=12)
                ax.legend(loc='lower right', bbox_to_anchor=(0.98, 0.15))
        
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
    if 'llm_cross_judge' not in data:
        try:
            judge_df = pd.read_csv('result-summary/llm-eval/llm_eval_cross_dataset_by_judge.csv')
            data['llm_cross_judge'] = judge_df
        except Exception as e:
            print(f"Error loading judge cross-dataset data: {e}")
            return
    
    judge_df = data['llm_cross_judge'].copy()
    
    if 'AVERAGE' in judge_df['Judge'].values:
        judge_df = judge_df[judge_df['Judge'] != 'AVERAGE']
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    axes = axes.flatten()
    
    dataset_variants = ['base', 'masked', 'shuffled', 'masked-shuffled']
    titles = ['Base Dataset', 'Masked Dataset', 'Shuffled Dataset', 'Masked-Shuffled Dataset']
    
    for i, (variant, title) in enumerate(zip(dataset_variants, titles)):
        ax = axes[i]
        
        cause_col = f"{variant}_cause"
        effect_col = f"{variant}_effect"
        
        if cause_col not in judge_df.columns or effect_col not in judge_df.columns:
            alt_variant = variant.replace('-', '_')
            cause_col = f"{alt_variant}_cause"
            effect_col = f"{alt_variant}_effect"
            
            if cause_col not in judge_df.columns or effect_col not in judge_df.columns:
                print(f"Could not find data for {variant} dataset")
                continue
        
        judge_df['avg_score'] = (judge_df[cause_col] + judge_df[effect_col]) / 2
        df_sorted = judge_df.sort_values(by='avg_score', ascending=False).reset_index(drop=True)
        
        x = np.arange(len(df_sorted))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, df_sorted[cause_col], width, label='Cause Score')
        bars2 = ax.bar(x + width/2, df_sorted[effect_col], width, label='Effect Score')
        
        for j, v in enumerate(df_sorted[cause_col]):
            ax.text(j - width/2, v + 0.02, f'{v:.2f}', ha='center', fontsize=9)
        for j, v in enumerate(df_sorted[effect_col]):
            ax.text(j + width/2, v + 0.02, f'{v:.2f}', ha='center', fontsize=9)
        
        cause_avg = df_sorted[cause_col].mean()
        effect_avg = df_sorted[effect_col].mean()
        ax.axhline(y=cause_avg, color='blue', linestyle='--', alpha=0.5, label=f'Avg Cause: {cause_avg:.2f}')
        ax.axhline(y=effect_avg, color='orange', linestyle='--', alpha=0.5, label=f'Avg Effect: {effect_avg:.2f}')
        
        ax.set_title(title, fontsize=16)
        ax.set_xlabel('Judge Models')
        ax.set_ylabel('Score')
        ax.set_ylim(0, 1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(df_sorted['Judge'], rotation=45, ha='right')
        ax.legend()
    
    plt.suptitle('Cause vs. Effect Extraction Performance by Judge Models Across Datasets', fontsize=20)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    plt.savefig('charts/judge_cause_effect_comparison.svg', format='svg')
    plt.close()
    
    plot_judge_cause_effect_table(data)


def plot_judge_cause_effect_table(data):
    """
    Create a tabular visualization showing the difference between cause and effect scores
    for each judge across all dataset variants.
    """
    judge_df = data['llm_cross_judge'].copy()
    variants = ['base', 'masked', 'shuffled', 'masked-shuffled']
    alt_variants = ['base', 'masked', 'shuffled', 'masked_shuffled']
    
    diff_data = {'Judge': judge_df['Judge']}
    
    for variant, alt_variant in zip(variants, alt_variants):
        cause_col = f"{variant}_cause"
        effect_col = f"{variant}_effect"
        
        if cause_col not in judge_df.columns or effect_col not in judge_df.columns:
            cause_col = f"{alt_variant}_cause"
            effect_col = f"{alt_variant}_effect"
        
        if cause_col in judge_df.columns and effect_col in judge_df.columns:
            diff_data[variant] = (judge_df[effect_col] - judge_df[cause_col]).abs()
    
    diff_df = pd.DataFrame(diff_data)
    diff_df['avg_diff'] = diff_df[[v for v in variants if v in diff_df.columns]].mean(axis=1)
    diff_df = diff_df.sort_values(by='avg_diff', ascending=True)
    diff_df = diff_df.drop(columns=['avg_diff'])
    
    fig, ax = plt.figure(figsize=(12, 8)), plt.subplot(111)
    ax.axis('off')
    ax.axis('tight')
    
    table = ax.table(
        cellText=diff_df.drop(columns=['Judge']).applymap(lambda x: f'{x:.3f}').values,
        rowLabels=diff_df['Judge'],
        colLabels=[v.replace('-', '-\n') for v in variants if v in diff_df.columns],
        loc='center',
        cellLoc='center',
        colColours=['#C9D7F0']*len([v for v in variants if v in diff_df.columns]),
        rowColours=['#E3E3E3']*len(diff_df)
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    for i in range(len(diff_df)):
        for j in range(len([v for v in variants if v in diff_df.columns])):
            cell = table[i+1, j]  # +1 for header row
            value = diff_df.iloc[i, j+1]  # +1 for Judge column
            
            intensity = min(value / 0.4, 1.0)
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