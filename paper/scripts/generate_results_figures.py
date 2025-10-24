"""
Generate comprehensive figures and improved tables for results section.
Includes:
- Performance comparison charts
- Scenario heatmaps
- Transfer learning analysis
- Real data stratified analysis
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configure matplotlib
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'figure.figsize': (10, 6),
    'figure.dpi': 300,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9
})

# Paths
SYNTHETIC_CSV = r"c:\Users\allan\Documents\GitHub\online-cpd-pipeline\results\results_synthetic\10-24-2025-resultados_algoritmos_main2-synthetic.csv"
REAL_CSV = r"c:\Users\allan\Documents\GitHub\online-cpd-pipeline\results\results_real\10-24-2025-resultados_algoritmos_main3_real-real.csv"
COMPARISON_CSV = r"c:\Users\allan\Documents\GitHub\online-cpd-pipeline\results\results_comparision\10-24-2025_comparison_grid_vs_transfer.csv"
CLASSIFICATION_CSV = r"c:\Users\allan\Documents\GitHub\online-cpd-pipeline\results\results_real\10-24-2025-clasificacion_series_criminalidad.csv"

FIGURES_DIR = r"c:\Users\allan\Documents\GitHub\online-cpd-pipeline\paper\figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

def generate_top_algorithms_barplot():
    """Figure 1: Top 10 algorithms comparison (Synthetic vs Real)."""
    df_synth = pd.read_csv(SYNTHETIC_CSV)
    df_real = pd.read_csv(REAL_CSV)
    
    # Filter OK algorithms
    df_synth_ok = df_synth[df_synth['status'] == 'ok'].copy()
    df_real_ok = df_real[df_real['status'] == 'ok'].copy()
    
    # Get top 10 from synthetic
    synth_avg = df_synth_ok.groupby('algorithm_key')['test_f1_mean'].mean().sort_values(ascending=False).head(10)
    
    # Get corresponding real data performance
    real_perf = df_real_ok.set_index('algorithm_key')['test_f1_mean']
    
    # Create dataframe for plotting
    plot_data = pd.DataFrame({
        'Algorithm': synth_avg.index,
        'Synthetic': synth_avg.values,
        'Real': [real_perf.get(algo, 0) for algo in synth_avg.index]
    })
    
    # Clean algorithm names
    plot_data['Algorithm'] = plot_data['Algorithm'].str.replace('_', ' ').str.title()
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(plot_data))
    width = 0.35
    
    ax.bar(x - width/2, plot_data['Synthetic'], width, label='Synthetic Data', color='#2E86AB', alpha=0.8)
    ax.bar(x + width/2, plot_data['Real'], width, label='Real Crime Data', color='#A23B72', alpha=0.8)
    
    ax.set_xlabel('Algorithm', fontweight='bold')
    ax.set_ylabel('F1 Score', fontweight='bold')
    ax.set_title('Top 10 Algorithm Performance: Synthetic vs Real Data', fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_data['Algorithm'], rotation=45, ha='right')
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, max(plot_data[['Synthetic', 'Real']].max()) * 1.1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig_top_algorithms_comparison.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(FIGURES_DIR, 'fig_top_algorithms_comparison.png'), bbox_inches='tight')
    plt.close()
    
    print("✓ Figure 1: Top algorithms comparison")

def generate_scenario_heatmap():
    """Figure 2: Heatmap of F1 scores across scenarios."""
    df = pd.read_csv(SYNTHETIC_CSV)
    df_ok = df[df['status'] == 'ok'].copy()
    
    # Get top 15 algorithms overall
    top_algos = df_ok.groupby('algorithm_key')['test_f1_mean'].mean().sort_values(ascending=False).head(15).index
    
    # Filter to top algorithms
    df_top = df_ok[df_ok['algorithm_key'].isin(top_algos)].copy()
    
    # Create scenario labels
    df_top['scenario'] = df_top.apply(
        lambda row: f"{row['nivel_ruido'][0].upper()}{row['fuerza_cambio'][0].upper()}-{row['tipo_cambio'][:3].upper()}", 
        axis=1
    )
    
    # Pivot for heatmap
    heatmap_data = df_top.pivot_table(
        index='algorithm_key', 
        columns='scenario', 
        values='test_f1_mean',
        aggfunc='mean'
    )
    
    # Clean algorithm names
    heatmap_data.index = heatmap_data.index.str.replace('_', ' ').str.title()
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='RdYlGn', center=0.3,
                vmin=0, vmax=1.0, cbar_kws={'label': 'F1 Score'},
                linewidths=0.5, ax=ax)
    
    ax.set_xlabel('Scenario (Noise-Magnitude-Type)', fontweight='bold')
    ax.set_ylabel('Algorithm', fontweight='bold')
    ax.set_title('Algorithm Performance Across Synthetic Scenarios', fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig_scenario_heatmap.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(FIGURES_DIR, 'fig_scenario_heatmap.png'), bbox_inches='tight')
    plt.close()
    
    print("✓ Figure 2: Scenario performance heatmap")

def generate_metric_radar_chart():
    """Figure 3: Radar chart comparing top algorithms on multiple metrics."""
    df_synth = pd.read_csv(SYNTHETIC_CSV)
    df_ok = df_synth[df_synth['status'] == 'ok'].copy()
    
    # Get top 5 algorithms
    top5 = df_ok.groupby('algorithm_key')['test_f1_mean'].mean().sort_values(ascending=False).head(5).index
    
    # Aggregate metrics for top 5
    metrics_data = df_ok[df_ok['algorithm_key'].isin(top5)].groupby('algorithm_key').agg({
        'test_f1_mean': 'mean',
        'test_precision_mean': 'mean',
        'test_recall_mean': 'mean',
        'test_mmd_mean': lambda x: 1 - x.mean(),  # Invert MMD (lower is better)
        'test_mttd_mean': lambda x: 1 - (x.mean() / 10)  # Normalize MTTD
    }).clip(lower=0)
    
    # Radar chart
    categories = ['F1', 'Precision', 'Recall', 'MMD\n(inverted)', 'MTTD\n(inverted)']
    N = len(categories)
    
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']
    
    for idx, (algo, row) in enumerate(metrics_data.iterrows()):
        values = row.values.tolist()
        values += values[:1]
        
        algo_name = algo.replace('_', ' ').title()
        ax.plot(angles, values, 'o-', linewidth=2, label=algo_name, color=colors[idx])
        ax.fill(angles, values, alpha=0.15, color=colors[idx])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=10)
    ax.set_ylim(0, 1)
    ax.set_title('Multi-Metric Comparison: Top 5 Algorithms', fontweight='bold', size=14, pad=30)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig_radar_metrics.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(FIGURES_DIR, 'fig_radar_metrics.png'), bbox_inches='tight')
    plt.close()
    
    print("✓ Figure 3: Multi-metric radar chart")

def generate_transfer_learning_scatter():
    """Figure 4: Transfer learning success/failure scatter plot."""
    df = pd.read_csv(COMPARISON_CSV)
    
    # Calculate improvement
    df['improvement'] = df['f1_difference']
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Color by success/failure
    colors = df['improvement'].apply(lambda x: '#6A994E' if x >= -0.05 else ('#F18F01' if x >= -0.20 else '#C73E1D'))
    
    scatter = ax.scatter(df['synthetic_test_f1'], df['transfer_test_f1'], 
                        c=colors, s=100, alpha=0.6, edgecolors='black', linewidth=0.5)
    
    # Add diagonal line (perfect transfer)
    max_val = max(df['synthetic_test_f1'].max(), df['transfer_test_f1'].max())
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, label='Perfect Transfer')
    
    # Add labels for extreme cases
    for idx, row in df.iterrows():
        if abs(row['improvement']) > 0.10:
            algo_short = row['algorithm'].replace('_', ' ')[:15]
            ax.annotate(algo_short, (row['synthetic_test_f1'], row['transfer_test_f1']),
                       fontsize=7, alpha=0.7)
    
    ax.set_xlabel('Synthetic F1 Score', fontweight='bold')
    ax.set_ylabel('Transfer F1 Score (Real Data)', fontweight='bold')
    ax.set_title('Transfer Learning Performance: Synthetic → Real', fontweight='bold', pad=20)
    ax.grid(alpha=0.3)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#6A994E', label='Success (Δ ≥ -0.05)'),
        Patch(facecolor='#F18F01', label='Moderate (-0.20 < Δ < -0.05)'),
        Patch(facecolor='#C73E1D', label='Failure (Δ < -0.20)')
    ]
    ax.legend(handles=legend_elements, loc='upper left')
    
    # Add correlation
    corr = df[['synthetic_test_f1', 'transfer_test_f1']].corr().iloc[0, 1]
    ax.text(0.95, 0.05, f'r = {corr:.3f}', transform=ax.transAxes,
           fontsize=12, verticalalignment='bottom', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig_transfer_learning_scatter.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(FIGURES_DIR, 'fig_transfer_learning_scatter.png'), bbox_inches='tight')
    plt.close()
    
    print("✓ Figure 4: Transfer learning scatter plot")

def generate_real_data_stratified_analysis():
    """Figure 5: Real data performance by noise/magnitude categories."""
    df_real = pd.read_csv(REAL_CSV)
    df_classif = pd.read_csv(CLASSIFICATION_CSV)
    
    # Check if we have per-series results
    if 'series_id' not in df_real.columns:
        print("⚠ Warning: No per-series data available for stratified analysis")
        return
    
    # Merge with classification
    df_merged = df_real.merge(df_classif, on='series_id', how='left')
    
    # Get top 10 algorithms overall
    top10 = df_real.groupby('algorithm_key')['test_f1_mean'].mean().sort_values(ascending=False).head(10).index
    
    df_top = df_merged[df_merged['algorithm_key'].isin(top10)].copy()
    
    # Create category combinations
    df_top['category'] = df_top['nivel_ruido'] + '-' + df_top['fuerza_cambio']
    
    # Aggregate by category
    category_perf = df_top.groupby(['algorithm_key', 'category'])['test_f1_mean'].mean().unstack(fill_value=0)
    
    # Plot grouped bar chart
    fig, ax = plt.subplots(figsize=(12, 7))
    category_perf.plot(kind='bar', ax=ax, width=0.8, 
                       color=['#2E86AB', '#A23B72', '#F18F01', '#6A994E'])
    
    ax.set_xlabel('Algorithm', fontweight='bold')
    ax.set_ylabel('F1 Score', fontweight='bold')
    ax.set_title('Real Data Performance by Noise-Magnitude Categories', fontweight='bold', pad=20)
    ax.legend(title='Category', title_fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.set_xticklabels([x.replace('_', ' ').title() for x in category_perf.index], rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig_real_stratified.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(FIGURES_DIR, 'fig_real_stratified.png'), bbox_inches='tight')
    plt.close()
    
    print("✓ Figure 5: Real data stratified analysis")

def generate_scenario_difficulty_boxplot():
    """Figure 6: Box plot showing F1 distribution per scenario."""
    df = pd.read_csv(SYNTHETIC_CSV)
    df_ok = df[df['status'] == 'ok'].copy()
    
    # Create readable scenario labels
    df_ok['scenario'] = df_ok.apply(
        lambda row: f"{row['nivel_ruido'][0].upper()}{row['fuerza_cambio'][0].upper()}\n{row['tipo_cambio'][:4]}", 
        axis=1
    )
    
    # Order scenarios by median F1
    scenario_order = df_ok.groupby('scenario')['test_f1_mean'].median().sort_values(ascending=False).index
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create box plot
    bp = ax.boxplot([df_ok[df_ok['scenario'] == s]['test_f1_mean'] for s in scenario_order],
                    labels=scenario_order,
                    patch_artist=True,
                    showmeans=True,
                    meanprops=dict(marker='D', markerfacecolor='red', markersize=6))
    
    # Color boxes by difficulty
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(scenario_order)))[::-1]
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_xlabel('Scenario (Noise-Magnitude-Type)', fontweight='bold')
    ax.set_ylabel('F1 Score Distribution', fontweight='bold')
    ax.set_title('Scenario Difficulty Analysis: F1 Score Distributions', fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig_scenario_difficulty.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(FIGURES_DIR, 'fig_scenario_difficulty.png'), bbox_inches='tight')
    plt.close()
    
    print("✓ Figure 6: Scenario difficulty box plot")

def main():
    """Generate all figures."""
    print("\n" + "="*60)
    print("GENERATING COMPREHENSIVE RESULT FIGURES")
    print("="*60 + "\n")
    
    generate_top_algorithms_barplot()
    generate_scenario_heatmap()
    generate_metric_radar_chart()
    generate_transfer_learning_scatter()
    generate_real_data_stratified_analysis()
    generate_scenario_difficulty_boxplot()
    
    print("\n" + "="*60)
    print(f"✅ All figures saved to: {FIGURES_DIR}")
    print("   - PDF format (for LaTeX)")
    print("   - PNG format (for preview)")
    print("="*60 + "\n")

if __name__ == '__main__':
    main()
