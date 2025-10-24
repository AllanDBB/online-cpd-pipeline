"""
Transfer Learning Approach: Use best parameters from synthetic data on real data.

This script provides three main functionalities:
1. Transfer learning benchmark (use synthetic best params on real data)
2. Comparison with grid search results
3. Interactive pipeline to run everything

Author: Allan
Date: October 24, 2025
"""

import os
import sys
import json
import glob
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime

# Import from existing modules
from benchmark_real_data import (
    load_real_data,
    build_algorithm_specs,
    safe_algorithm_call,
    train_test_split_real_data,
    CONFIG as BASE_CONFIG,
)
from src.f1_score import f1_score_with_tolerance
from src.mmd import maximum_mean_discrepancy
from src.mttd import mean_time_to_detection


def load_best_params_from_synthetic(synthetic_results_csv: str) -> Dict[str, Dict[str, Any]]:
    """
    Load best parameters from synthetic data results.
    
    Args:
        synthetic_results_csv: Path to the synthetic results CSV file
    
    Returns:
        Dictionary mapping algorithm_key to best parameters
        Example: {'adwin_river': {'delta': 0.002, 'clock': 32, ...}, ...}
    """
    df = pd.read_csv(synthetic_results_csv)
    
    # Filter only best configurations
    best_configs = df[df['is_best'] == True].copy()
    
    print(f"📊 Loaded {len(best_configs)} best configurations from synthetic data")
    
    # Group by algorithm and scenario (if multiple best configs exist)
    best_params_map = {}
    
    for _, row in best_configs.iterrows():
        algo_key = row['algorithm_key']
        params_json = row['params_json']
        
        # Parse parameters
        try:
            params = json.loads(params_json)
        except:
            params = {}
        
        # Store the first best configuration found for each algorithm
        if algo_key not in best_params_map:
            best_params_map[algo_key] = {
                'params': params,
                'synthetic_performance': {
                    'train_f1': row.get('train_f1_mean'),
                    'test_f1': row.get('test_f1_mean'),
                    'test_precision': row.get('test_precision_mean'),
                    'test_recall': row.get('test_recall_mean'),
                    'scenario': f"{row.get('nivel_ruido')}_{row.get('fuerza_cambio')}_{row.get('tipo_cambio')}"
                }
            }
    
    print(f"✅ Extracted best parameters for {len(best_params_map)} algorithms")
    for algo_key in sorted(best_params_map.keys()):
        scenario = best_params_map[algo_key]['synthetic_performance']['scenario']
        test_f1 = best_params_map[algo_key]['synthetic_performance']['test_f1']
        print(f"  - {algo_key}: F1={test_f1:.3f} on {scenario}")
    
    return best_params_map


def evaluate_algorithm_on_real_data(
    spec,
    params: Dict[str, Any],
    datasets: List[Dict[str, Any]],
    delta_eval: int,
) -> Dict[str, Any]:
    """
    Evaluate a single algorithm configuration on real data.
    
    Args:
        spec: Algorithm specification
        params: Parameters to use (from synthetic best)
        datasets: Real data datasets
        delta_eval: Delta for evaluation
    
    Returns:
        Dictionary with evaluation metrics
    """
    per_series_metrics = []
    
    for dataset in datasets:
        serie = dataset['series']
        truth = dataset['changepoints']
        
        try:
            # Apply the detection algorithm with safety wrapper
            detected = safe_algorithm_call(
                spec.detect_fn, serie, params, spec.key, timeout_seconds=30
            ) if spec.detect_fn else []
            
            # Calculate metrics
            f1_result = f1_score_with_tolerance(truth, detected, delta_eval)
            
            # MMD needs series length
            series_length = len(serie)
            mmd_val = maximum_mean_discrepancy(truth, detected, series_length) if len(detected) > 0 else np.inf
            
            # MTTD needs delta parameter
            mttd_result = mean_time_to_detection(truth, detected, delta_eval)
            mttd_val = mttd_result if mttd_result is not None else np.inf
            
            per_series_metrics.append({
                'series_id': dataset.get('series_id'),
                'filename': dataset.get('filename'),
                'annotator': dataset.get('annotator'),
                'f1': f1_result['f1'],
                'precision': f1_result['precision'],
                'recall': f1_result['recall'],
                'tp': f1_result['TP'],
                'fp': f1_result['FP'],
                'fn': f1_result['FN'],
                'mmd': mmd_val,
                'mttd': mttd_val,
                'detections': len(detected),
                'ground_truth_cps': len(truth),
                'status': 'ok'
            })
            
        except Exception as e:
            per_series_metrics.append({
                'series_id': dataset.get('series_id'),
                'filename': dataset.get('filename'),
                'annotator': dataset.get('annotator'),
                'f1': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'tp': 0,
                'fp': 0,
                'fn': 0,
                'mmd': np.inf,
                'mttd': np.inf,
                'detections': 0,
                'ground_truth_cps': len(truth),
                'status': f'error: {str(e)[:100]}'
            })
    
    # Calculate summary statistics
    valid_metrics = [m for m in per_series_metrics if m['status'] == 'ok']
    
    if not valid_metrics:
        return {
            'per_series_metrics': per_series_metrics,
            'summary': {
                'series_count': len(datasets),
                'valid_series': 0,
                'f1_mean': 0.0,
                'precision_mean': 0.0,
                'recall_mean': 0.0,
                'mmd_mean': np.inf,
                'mttd_mean': np.inf,
            }
        }
    
    summary = {
        'series_count': len(datasets),
        'valid_series': len(valid_metrics),
        'f1_mean': np.mean([m['f1'] for m in valid_metrics]),
        'f1_std': np.std([m['f1'] for m in valid_metrics]),
        'precision_mean': np.mean([m['precision'] for m in valid_metrics]),
        'recall_mean': np.mean([m['recall'] for m in valid_metrics]),
        'tp_mean': np.mean([m['tp'] for m in valid_metrics]),
        'fp_mean': np.mean([m['fp'] for m in valid_metrics]),
        'fn_mean': np.mean([m['fn'] for m in valid_metrics]),
        'mmd_mean': np.mean([m['mmd'] for m in valid_metrics if np.isfinite(m['mmd'])]),
        'mttd_mean': np.mean([m['mttd'] for m in valid_metrics if np.isfinite(m['mttd'])]),
        'detections_mean': np.mean([m['detections'] for m in valid_metrics]),
    }
    
    return {
        'per_series_metrics': per_series_metrics,
        'summary': summary
    }


def run_transfer_learning_benchmark(
    synthetic_results_csv: str,
    config: Dict[str, Any]
):
    """
    Main function to run transfer learning benchmark.
    
    Args:
        synthetic_results_csv: Path to synthetic results CSV
        config: Configuration dictionary
    """
    print("="*80)
    print(" "*15 + "TRANSFER LEARNING BENCHMARK")
    print(" "*10 + "Best Params from Synthetic → Real Data")
    print("="*80)
    print()
    
    # Load best parameters from synthetic
    best_params_map = load_best_params_from_synthetic(synthetic_results_csv)
    
    # Load real data
    data_path = os.path.join(os.path.dirname(__file__), config["data_real_path"])
    datasets = load_real_data(data_path, use_only_martin=config.get("use_only_martin_labels", True))
    
    if not datasets:
        print("❌ No real data found. Exiting.")
        return
    
    # Split into train/test
    split_data = train_test_split_real_data(datasets, test_size=0.5, seed=config["seed"])
    train_datasets = split_data['train']
    test_datasets = split_data['test']
    
    print(f"\n📊 Data split:")
    print(f"  Training: {len(train_datasets)} series")
    print(f"  Testing: {len(test_datasets)} series")
    print()
    
    # Build algorithm specs
    specs = build_algorithm_specs(config)
    
    # Results storage
    results = []
    
    # Evaluate each algorithm
    for spec in specs:
        if not spec.implemented:
            print(f"⏭️  Skipping {spec.key} (not implemented)")
            continue
        
        if spec.key not in best_params_map:
            print(f"⚠️  No best params found for {spec.key} in synthetic results, skipping...")
            continue
        
        print(f"\n{'='*80}")
        print(f"🔧 Algorithm: {spec.key}")
        print(f"   Library: {spec.library} | Method: {spec.method}")
        print(f"{'='*80}")
        
        best_params_info = best_params_map[spec.key]
        params = best_params_info['params']
        synthetic_perf = best_params_info['synthetic_performance']
        
        print(f"📦 Using parameters from synthetic data:")
        print(f"   Params: {params}")
        print(f"   Synthetic scenario: {synthetic_perf['scenario']}")
        print(f"   Synthetic test F1: {synthetic_perf['test_f1']:.3f}")
        print()
        
        # Evaluate on train data
        print(f"🏋️  Evaluating on training data ({len(train_datasets)} series)...")
        train_eval = evaluate_algorithm_on_real_data(
            spec, params, train_datasets, config["delta_eval"]
        )
        
        # Evaluate on test data
        print(f"🧪 Evaluating on test data ({len(test_datasets)} series)...")
        test_eval = evaluate_algorithm_on_real_data(
            spec, params, test_datasets, config["delta_eval"]
        )
        
        # Print results
        print(f"\n📈 Results:")
        print(f"   Train F1: {train_eval['summary']['f1_mean']:.4f}")
        print(f"   Test F1:  {test_eval['summary']['f1_mean']:.4f}")
        print(f"   Test Precision: {test_eval['summary']['precision_mean']:.4f}")
        print(f"   Test Recall: {test_eval['summary']['recall_mean']:.4f}")
        
        # Store results
        result = {
            'algorithm_key': spec.key,
            'algorithm_library': spec.library,
            'algorithm_method': spec.method,
            'approach': 'transfer_learning',
            'params_source': 'synthetic_best',
            'params_json': json.dumps(params, sort_keys=True),
            'synthetic_scenario': synthetic_perf['scenario'],
            'synthetic_test_f1': synthetic_perf['test_f1'],
            'train_f1_mean': train_eval['summary']['f1_mean'],
            'train_precision_mean': train_eval['summary']['precision_mean'],
            'train_recall_mean': train_eval['summary']['recall_mean'],
            'train_mmd_mean': train_eval['summary']['mmd_mean'],
            'train_mttd_mean': train_eval['summary']['mttd_mean'],
            'train_series_count': train_eval['summary']['series_count'],
            'train_valid_series': train_eval['summary']['valid_series'],
            'test_f1_mean': test_eval['summary']['f1_mean'],
            'test_f1_std': test_eval['summary']['f1_std'],
            'test_precision_mean': test_eval['summary']['precision_mean'],
            'test_recall_mean': test_eval['summary']['recall_mean'],
            'test_mmd_mean': test_eval['summary']['mmd_mean'],
            'test_mttd_mean': test_eval['summary']['mttd_mean'],
            'test_detections_mean': test_eval['summary']['detections_mean'],
            'test_tp_mean': test_eval['summary']['tp_mean'],
            'test_fp_mean': test_eval['summary']['fp_mean'],
            'test_fn_mean': test_eval['summary']['fn_mean'],
            'test_series_count': test_eval['summary']['series_count'],
            'test_valid_series': test_eval['summary']['valid_series'],
        }
        results.append(result)
        
        # Save detailed results
        timestamp = datetime.now().strftime("%m-%d-%Y")
        detailed_results_file = f"{timestamp}_transfer_learning_{spec.key}_detailed.json"
        detailed_data = {
            'algorithm': spec.key,
            'params': params,
            'synthetic_performance': synthetic_perf,
            'train_evaluation': train_eval,
            'test_evaluation': test_eval,
        }
        with open(detailed_results_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_data, f, indent=2, ensure_ascii=False, default=str)
        print(f"   💾 Detailed results saved to: {detailed_results_file}")
    
    # Save summary results
    df = pd.DataFrame(results)
    timestamp = datetime.now().strftime("%m-%d-%Y")
    summary_csv = f"{timestamp}_transfer_learning_real_data_results.csv"
    df.to_csv(summary_csv, index=False)
    
    print("\n" + "="*80)
    print("✅ TRANSFER LEARNING BENCHMARK COMPLETED")
    print("="*80)
    print(f"📊 Summary results saved to: {summary_csv}")
    print(f"🔢 Total algorithms evaluated: {len(results)}")
    
    # Print ranking
    if results:
        df_sorted = df.sort_values('test_f1_mean', ascending=False)
        print(f"\n🏆 TOP 5 ALGORITHMS (by Test F1):")
        for idx, (_, row) in enumerate(df_sorted.head(5).iterrows(), 1):
            print(f"   {idx}. {row['algorithm_key']}: F1={row['test_f1_mean']:.4f} "
                  f"(Synthetic F1={row['synthetic_test_f1']:.4f})")
    
    print("="*80)


# ============================================================================
# COMPARISON FUNCTIONS
# ============================================================================

def find_latest_file(pattern: str) -> Optional[str]:
    """Find the latest file matching a pattern in multiple locations."""
    # Search in multiple locations
    search_paths = [
        pattern,  # Current directory
        os.path.join("results", "results_real", os.path.basename(pattern)),  # results/results_real/
        os.path.join("results", os.path.basename(pattern)),  # results/
    ]
    
    all_files = []
    for search_path in search_paths:
        files = glob.glob(search_path)
        all_files.extend(files)
    
    if not all_files:
        return None
    
    # Sort by modification time and return the latest
    all_files.sort(key=os.path.getmtime, reverse=True)
    return all_files[0]


def compare_approaches() -> Optional[pd.DataFrame]:
    """
    Compare grid search vs transfer learning results.
    
    Returns:
        DataFrame with comparison or None if files not found
    """
    print("="*80)
    print(" "*20 + "COMPARISON ANALYSIS")
    print(" "*10 + "Grid Search vs Transfer Learning")
    print("="*80)
    print()
    
    # Find latest results files
    grid_search_file = find_latest_file("*-resultados_algoritmos_main3_real-real.csv")
    transfer_learning_file = find_latest_file("*_transfer_learning_real_data_results.csv")
    
    if not grid_search_file:
        print("⚠️  No grid search results found. Run benchmark_real_data.py first.")
        return None
    
    if not transfer_learning_file:
        print("⚠️  No transfer learning results found. Run transfer learning benchmark first.")
        return None
    
    # Load data
    print(f"📂 Loading grid search results from: {grid_search_file}")
    df_grid = pd.read_csv(grid_search_file)
    
    print(f"📂 Loading transfer learning results from: {transfer_learning_file}")
    df_transfer = pd.read_csv(transfer_learning_file)
    
    # Filter to best configurations for grid search
    if 'is_best' in df_grid.columns:
        df_grid_best = df_grid[df_grid['is_best'] == True].copy()
    else:
        # Assume all rows are best if no is_best column
        df_grid_best = df_grid.copy()
    
    print(f"\n✅ Loaded:")
    print(f"   Grid search (best configs): {len(df_grid_best)} algorithms")
    print(f"   Transfer learning: {len(df_transfer)} algorithms")
    print()
    
    # Find common algorithms
    common_algos = set(df_grid_best['algorithm_key']) & set(df_transfer['algorithm_key'])
    print(f"🔗 Common algorithms: {len(common_algos)}")
    print()
    
    if not common_algos:
        print("❌ No common algorithms found between approaches!")
        return None
    
    # Compare results
    comparison_data = []
    
    for algo in sorted(common_algos):
        grid_row = df_grid_best[df_grid_best['algorithm_key'] == algo].iloc[0]
        transfer_row = df_transfer[df_transfer['algorithm_key'] == algo].iloc[0]
        
        grid_f1 = grid_row.get('test_f1_mean', 0)
        transfer_f1 = transfer_row.get('test_f1_mean', 0)
        
        comparison = {
            'algorithm': algo,
            'grid_test_f1': grid_f1,
            'transfer_test_f1': transfer_f1,
            'grid_test_precision': grid_row.get('test_precision_mean', 0),
            'transfer_test_precision': transfer_row.get('test_precision_mean', 0),
            'grid_test_recall': grid_row.get('test_recall_mean', 0),
            'transfer_test_recall': transfer_row.get('test_recall_mean', 0),
            'grid_params': grid_row.get('params_json', '{}'),
            'transfer_params': transfer_row.get('params_json', '{}'),
            'synthetic_test_f1': transfer_row.get('synthetic_test_f1', 0),
            'f1_difference': transfer_f1 - grid_f1,
            'f1_improvement_pct': ((transfer_f1 - grid_f1) / max(grid_f1, 0.001)) * 100,
        }
        comparison_data.append(comparison)
    
    df_comparison = pd.DataFrame(comparison_data)
    
    # Sort by F1 difference
    df_comparison = df_comparison.sort_values('f1_difference', ascending=False)
    
    # Print summary statistics
    print("="*80)
    print("📊 SUMMARY STATISTICS")
    print("="*80)
    
    print(f"\n🎯 Average Test F1 Score:")
    print(f"   Grid Search:       {df_comparison['grid_test_f1'].mean():.4f} (±{df_comparison['grid_test_f1'].std():.4f})")
    print(f"   Transfer Learning: {df_comparison['transfer_test_f1'].mean():.4f} (±{df_comparison['transfer_test_f1'].std():.4f})")
    print(f"   Difference:        {df_comparison['f1_difference'].mean():.4f}")
    
    print(f"\n📈 Performance Comparison:")
    transfer_better = (df_comparison['f1_difference'] > 0).sum()
    grid_better = (df_comparison['f1_difference'] < 0).sum()
    tied = (df_comparison['f1_difference'] == 0).sum()
    
    total = len(common_algos)
    print(f"   Transfer Learning better: {transfer_better} algorithms ({transfer_better/total*100:.1f}%)")
    print(f"   Grid Search better:       {grid_better} algorithms ({grid_better/total*100:.1f}%)")
    print(f"   Tied:                     {tied} algorithms ({tied/total*100:.1f}%)")
    
    # Best and worst improvements
    if len(df_comparison) > 0:
        print(f"\n🏆 TOP 5 - Transfer Learning WINS (biggest improvements):")
        for idx, (_, row) in enumerate(df_comparison.head(min(5, len(df_comparison))).iterrows(), 1):
            print(f"   {idx}. {row['algorithm']}")
            print(f"      Transfer F1: {row['transfer_test_f1']:.4f} | Grid F1: {row['grid_test_f1']:.4f}")
            print(f"      Improvement: {row['f1_difference']:+.4f} ({row['f1_improvement_pct']:+.1f}%)")
        
        if len(df_comparison) > 5:
            print(f"\n❌ TOP 5 - Grid Search WINS (transfer learning did worse):")
            for idx, (_, row) in enumerate(df_comparison.tail(min(5, len(df_comparison))).iterrows(), 1):
                print(f"   {idx}. {row['algorithm']}")
                print(f"      Transfer F1: {row['transfer_test_f1']:.4f} | Grid F1: {row['grid_test_f1']:.4f}")
                print(f"      Difference: {row['f1_difference']:+.4f} ({row['f1_improvement_pct']:+.1f}%)")
    
    # Correlation between synthetic and real performance
    print(f"\n🔗 SYNTHETIC vs REAL CORRELATION:")
    synthetic_f1 = df_comparison['synthetic_test_f1'].values
    real_transfer_f1 = df_comparison['transfer_test_f1'].values
    real_grid_f1 = df_comparison['grid_test_f1'].values
    
    if len(synthetic_f1) > 1:
        corr_transfer = np.corrcoef(synthetic_f1, real_transfer_f1)[0, 1]
        corr_grid = np.corrcoef(synthetic_f1, real_grid_f1)[0, 1]
        
        print(f"   Synthetic F1 vs Real Transfer F1: {corr_transfer:.3f}")
        print(f"   Synthetic F1 vs Real Grid F1:     {corr_grid:.3f}")
        
        if corr_transfer > 0.5:
            print(f"   ✅ Strong positive correlation! Synthetic data is predictive.")
        elif corr_transfer > 0.3:
            print(f"   ⚠️  Moderate correlation. Some transfer, but domain shift exists.")
        else:
            print(f"   ❌ Weak correlation. Significant domain shift between synthetic and real.")
    
    # Save comparison results
    timestamp = datetime.now().strftime("%m-%d-%Y")
    comparison_file = f"{timestamp}_comparison_grid_vs_transfer.csv"
    df_comparison.to_csv(comparison_file, index=False)
    
    print(f"\n💾 Comparison saved to: {comparison_file}")
    
    # Generate report
    generate_comparison_report(df_comparison)
    
    print("="*80)
    
    return df_comparison


def generate_comparison_report(df_comparison: pd.DataFrame):
    """Generate a detailed comparison report."""
    timestamp = datetime.now().strftime("%m-%d-%Y")
    report_file = f"{timestamp}_comparison_report.txt"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(" "*20 + "COMPARISON REPORT\n")
        f.write(" "*15 + "Grid Search vs Transfer Learning\n")
        f.write("="*80 + "\n\n")
        
        f.write("EXECUTIVE SUMMARY\n")
        f.write("-"*80 + "\n\n")
        
        # Overall statistics
        f.write(f"Total algorithms compared: {len(df_comparison)}\n\n")
        
        f.write("Average Test F1 Score:\n")
        f.write(f"  Grid Search:       {df_comparison['grid_test_f1'].mean():.4f}\n")
        f.write(f"  Transfer Learning: {df_comparison['transfer_test_f1'].mean():.4f}\n")
        f.write(f"  Difference:        {df_comparison['f1_difference'].mean():.4f}\n\n")
        
        # Winner count
        transfer_better = (df_comparison['f1_difference'] > 0).sum()
        grid_better = (df_comparison['f1_difference'] < 0).sum()
        
        f.write(f"Performance Comparison:\n")
        f.write(f"  Transfer Learning better: {transfer_better} algorithms\n")
        f.write(f"  Grid Search better:       {grid_better} algorithms\n\n")
        
        # Recommendation
        f.write("RECOMMENDATION\n")
        f.write("-"*80 + "\n\n")
        
        if df_comparison['f1_difference'].mean() > 0:
            f.write("✅ Transfer Learning approach shows promise!\n")
            f.write("   Consider using synthetic best parameters as starting point.\n")
        else:
            f.write("⚠️  Grid Search on real data performs better on average.\n")
            f.write("   Domain shift between synthetic and real data may be significant.\n")
        
        f.write("\n\nDETAILED RESULTS BY ALGORITHM\n")
        f.write("="*80 + "\n\n")
        
        for _, row in df_comparison.iterrows():
            f.write(f"Algorithm: {row['algorithm']}\n")
            f.write(f"  Grid Search F1:       {row['grid_test_f1']:.4f}\n")
            f.write(f"  Transfer Learning F1: {row['transfer_test_f1']:.4f}\n")
            f.write(f"  Difference:           {row['f1_difference']:+.4f} ({row['f1_improvement_pct']:+.1f}%)\n")
            f.write(f"  Synthetic F1:         {row['synthetic_test_f1']:.4f}\n")
            f.write(f"\n")
    
    print(f"📄 Detailed report saved to: {report_file}")


# ============================================================================
# INTERACTIVE PIPELINE
# ============================================================================

def print_banner(text: str):
    """Print a formatted banner."""
    print("\n" + "="*80)
    print(" " * ((80 - len(text)) // 2) + text)
    print("="*80 + "\n")


def check_file_exists(filepath: str, description: str) -> bool:
    """Check if a required file exists."""
    if os.path.exists(filepath):
        print(f"✅ {description} found: {filepath}")
        return True
    else:
        print(f"❌ {description} NOT found: {filepath}")
        return False


def run_interactive_pipeline():
    """Run the complete pipeline interactively."""
    print_banner("TRANSFER LEARNING PIPELINE")
    print(f"🕐 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Step 0: Check prerequisites
    print_banner("STEP 0: Checking Prerequisites")
    
    synthetic_results = os.path.join(
        os.path.dirname(__file__),
        "results",
        "results_synthetic",
        "10-24-2025-resultados_algoritmos_main2-synthetic.csv"
    )
    
    has_synthetic = check_file_exists(
        synthetic_results,
        "Synthetic results"
    )
    
    if not has_synthetic:
        print("\n❌ ERROR: Synthetic results not found!")
        print("   You need to run benchmark_synthetic_data.py first.")
        print(f"   Expected location: {synthetic_results}")
        return
    
    # Step 1: Run transfer learning
    print()
    print_banner("STEP 1: Transfer Learning Benchmark")
    response = input("Run transfer learning benchmark? (y/n): ").strip().lower()
    
    if response == 'y':
        config = BASE_CONFIG.copy()
        try:
            run_transfer_learning_benchmark(synthetic_results, config)
        except Exception as e:
            print(f"\n❌ Transfer learning failed: {e}")
            print("   Check errors above.")
    else:
        print("\n⏭️  Skipping transfer learning benchmark.")
    
    # Step 2: Check for grid search results
    print()
    print_banner("STEP 2: Checking for Grid Search Results")
    
    grid_search_files = glob.glob("*-resultados_algoritmos_main3_real-real.csv")
    
    if grid_search_files:
        print(f"✅ Found {len(grid_search_files)} grid search result file(s)")
        for f in grid_search_files:
            print(f"   - {f}")
    else:
        print("❌ No grid search results found")
        print("   Note: Grid search must be run separately (benchmark_real_data.py)")
        print("   It can take a LONG time, so it's not included in this pipeline.")
    
    # Step 3: Run comparison
    print()
    print_banner("STEP 3: Comparison Analysis")
    response = input("Run comparison analysis? (y/n): ").strip().lower()
    
    if response == 'y':
        try:
            df_comparison = compare_approaches()
            if df_comparison is not None:
                print("\n🎉 COMPARISON COMPLETED SUCCESSFULLY!")
            else:
                print("\n⚠️  Comparison could not be completed (missing files).")
        except Exception as e:
            print(f"\n❌ Comparison failed: {e}")
    else:
        print("\n⏭️  Skipping comparison analysis.")
    
    # Summary
    print()
    print_banner("PIPELINE SUMMARY")
    print(f"🕐 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("📂 Check the following output files:")
    print("   - *_transfer_learning_real_data_results.csv")
    print("   - *_transfer_learning_*_detailed.json")
    print("   - *_comparison_grid_vs_transfer.csv")
    print("   - *_comparison_report.txt")
    print()


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point with mode selection."""
    # Parse command line arguments
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
    else:
        mode = None
    
    # Configuration
    config = BASE_CONFIG.copy()
    
    # Path to synthetic results  
    synthetic_results_csv = os.path.join(
        os.path.dirname(__file__),
        "results",
        "results_synthetic",
        "10-24-2025-resultados_algoritmos_main2-synthetic.csv"
    )
    
    # Show menu if no mode specified
    if mode is None:
        print("="*80)
        print(" "*20 + "TRANSFER LEARNING TOOL")
        print("="*80)
        print("\nAvailable modes:")
        print("  1. benchmark    - Run transfer learning benchmark only")
        print("  2. compare      - Compare grid search vs transfer learning")
        print("  3. interactive  - Interactive pipeline (recommended)")
        print()
        choice = input("Select mode (1-3) or 'q' to quit: ").strip()
        
        if choice == '1':
            mode = 'benchmark'
        elif choice == '2':
            mode = 'compare'
        elif choice == '3':
            mode = 'interactive'
        elif choice.lower() == 'q':
            print("Exiting...")
            return
        else:
            print("Invalid choice. Exiting...")
            return
    
    # Execute based on mode
    if mode == 'benchmark':
        if not os.path.exists(synthetic_results_csv):
            print(f"❌ Error: Synthetic results not found at {synthetic_results_csv}")
            print("   Please run benchmark_synthetic_data.py first to generate results.")
            return
        run_transfer_learning_benchmark(synthetic_results_csv, config)
    
    elif mode == 'compare':
        compare_approaches()
    
    elif mode == 'interactive':
        run_interactive_pipeline()
    
    else:
        print(f"❌ Unknown mode: {mode}")
        print("   Valid modes: benchmark, compare, interactive")
        return


if __name__ == "__main__":
    main()
