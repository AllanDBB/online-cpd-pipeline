# Online Change Point Detection Pipeline

## Overview

This repository contains a comprehensive benchmarking framework for evaluating online change point detection (CPD) algorithms on both synthetic and real-world time series data. The framework implements multiple state-of-the-art algorithms and provides tools for systematic evaluation, parameter optimization, and performance comparison.

## Project Structure

```
online-cpd-pipeline/
├── src/                              # Core source code
│   ├── data_processing.py           # Data loading and preprocessing utilities
│   ├── f1_score.py                  # F1 score with temporal tolerance
│   ├── mmd.py                       # Maximum Mean Discrepancy metric
│   └── mttd.py                      # Mean Time to Detection metric
├── data/                            # Data directories
│   ├── data_real/                   # Real-world labeled datasets
│   └── TCDP-paper/                  # TCPD benchmark datasets
├── results/                         # Output directories
│   ├── results_synthetic/           # Synthetic data results
│   └── results_real/                # Real data results
├── benchmark_synthetic_data.py      # Synthetic data benchmark
├── benchmark_real_data.py           # Real data benchmark with grid search
├── benchmark_real_data_transfer_learning.py  # Transfer learning approach
├── benchmark_tcpd_data.py           # TCPD benchmark data evaluation
└── calculate_agreement.py           # Inter-annotator agreement analysis
```

## Features

### Implemented Algorithms

The framework includes 17 online change point detection algorithms from various libraries:

**River-based algorithms:**
- Page-Hinkley Test
- ADWIN (Adaptive Windowing)

**Statistical methods:**
- EWMA (Exponentially Weighted Moving Average)
- CUSUM (Cumulative Sum)

**Ruptures-based (online variants):**
- Focus (RBF kernel)
- Gaussian (L2 cost)
- NPFocus (Non-parametric)
- MDFocus (Mahalanobis distance)

**State-space models:**
- SSM with Kalman filtering
- TAGI-LSTM/SSM
- SKF (Switching Kalman Filter)

**Other methods:**
- Bayesian Online CPD
- ChangeFinder
- OCPDet suite (neural networks, two-sample tests)
- RULSIF

### Evaluation Metrics

- **F1 Score with temporal tolerance (delta)**: Primary metric accounting for detection delays
- **Precision and Recall**: Standard classification metrics
- **Maximum Mean Discrepancy (MMD)**: Distribution-based similarity measure
- **Mean Time to Detection (MTTD)**: Average delay in detecting true change points
- **True Positives, False Positives, False Negatives**: Confusion matrix elements

## Usage

### 1. Synthetic Data Benchmark

Evaluates algorithms on synthetically generated time series with known change points.

```bash
python benchmark_synthetic_data.py
```

**Features:**
- Configurable noise levels (low, medium, high)
- Multiple change types (step, slope, variance)
- Change magnitude control (low, medium, high)
- Automatic train/test split
- Grid search for hyperparameter optimization

**Output:**
- `results/results_synthetic/{date}-resultados_algoritmos_main2-synthetic.csv`
- `results/results_synthetic/best_series_{scenario}_{algorithm}_trial{n}.json`

### 2. Real Data Benchmark

Evaluates algorithms on manually labeled real-world time series.

```bash
python benchmark_real_data.py
```

**Features:**
- Uses labeled crime statistics time series
- Configurable to use single or multiple annotators
- Inter-annotator agreement analysis
- Series classification by noise level and change characteristics
- Grid search for optimal parameters per algorithm

**Configuration:**
```python
CONFIG = {
    "use_only_martin_labels": True,  # Use single annotator (recommended)
    "delta_eval": 10,                # Temporal tolerance for evaluation
    "seed": 123,                     # Random seed for reproducibility
    "profile": "quick",              # or "full" for exhaustive search
}
```

**Output:**
- `results/results_real/{date}-resultados_algoritmos_main3_real-real.csv`
- `results/results_real/{date}-best_series_analysis_real_data.json`
- `results/results_real/{date}-clasificacion_series_criminalidad.csv`

### 3. Transfer Learning Approach

Applies best parameters found on synthetic data directly to real data, avoiding expensive grid search.

```bash
# Interactive mode (recommended)
python benchmark_real_data_transfer_learning.py interactive

# Direct benchmark mode
python benchmark_real_data_transfer_learning.py benchmark

# Comparison mode only
python benchmark_real_data_transfer_learning.py compare
```

**Features:**
- Loads optimal parameters from synthetic benchmark
- Evaluates on real data without hyperparameter search
- Compares performance with grid search approach
- Analyzes correlation between synthetic and real performance
- Generates comprehensive comparison reports

**Output:**
- `{date}_transfer_learning_real_data_results.csv`
- `{date}_transfer_learning_{algorithm}_detailed.json`
- `{date}_comparison_grid_vs_transfer.csv`
- `{date}_comparison_report.txt`

### 4. Inter-Annotator Agreement Analysis

Calculates agreement between multiple human annotators using F1 score.

```bash
python calculate_agreement.py
```

**Output:**
- `{timestamp}_agreement_analysis_martin_vs_allan.json`

### 5. TCPD Benchmark Data Evaluation

Evaluates algorithms on the TCPD benchmark suite.

```bash
python benchmark_tcpd_data.py
```

## Methodology

### Experimental Design

1. **Data Preparation:**
   - Synthetic: Generated with controlled characteristics
   - Real: Manually labeled by domain experts
   - TCPD: Standard benchmark datasets

2. **Algorithm Configuration:**
   - Systematic grid search over parameter space
   - Profile options: "quick" (coarse grid) or "full" (fine grid)

3. **Evaluation Protocol:**
   - Train/test split (typically 50/50)
   - Best parameters selected on training set
   - Performance measured on held-out test set
   - Temporal tolerance (delta) applied to account for detection delays

4. **Performance Analysis:**
   - Per-algorithm performance metrics
   - Per-scenario analysis (noise level, change type)
   - Statistical significance testing
   - Correlation analysis (synthetic vs. real)

### Transfer Learning Validation

The framework includes a novel approach to validate parameter transfer from synthetic to real data:

1. Identify best parameters on synthetic data
2. Apply directly to real data
3. Compare with grid search on real data
4. Measure correlation between domains

**Key Finding:** Strong positive correlation (r=0.71) between synthetic and real performance indicates that synthetic data can reliably guide algorithm selection.

## Configuration

### Main Configuration Parameters

**Data paths:**
- `data_real_path`: Directory containing real labeled data
- `results_csv`: Output filename for results

**Evaluation settings:**
- `delta_eval`: Temporal tolerance for matching detections (default: 10)
- `seed`: Random seed for reproducibility
- `profile`: Grid search granularity ("quick" or "full")

**Labeling settings:**
- `use_only_martin_labels`: Boolean to use single annotator vs. all (default: True)

### Algorithm-specific Parameters

Each algorithm has a configurable parameter grid. Example:

```python
"page_hinkley_river": {
    "grid": {
        "threshold": [20, 40, 60, 80],
        "min_instances": [5, 10, 20],
        "delta": [0.001, 0.005, 0.01],
    }
}
```

## Results Interpretation

### Output Files

**CSV Results:**
- Algorithm metadata (library, method, supervision type)
- Hyperparameters (JSON serialized)
- Performance metrics (F1, precision, recall, MMD, MTTD)
- Dataset statistics (series count, length, changepoints)
- Best configuration indicator

**JSON Results:**
- Detailed per-series results
- Training and test performance
- Dataset classification information
- Timestamp and configuration metadata

### Performance Metrics

**F1 Score:** Primary metric. Values range [0, 1], higher is better.
- < 0.3: Poor performance
- 0.3-0.5: Moderate performance
- 0.5-0.7: Good performance
- > 0.7: Excellent performance

**MMD:** Distribution similarity. Values range [0, inf], lower is better.
- 0: Perfect match
- < 0.5: Good alignment
- > 1.0: Significant discrepancy

**MTTD:** Detection delay in time steps. Lower is better.

## Dependencies

**Core libraries:**
- numpy
- pandas
- scipy
- scikit-learn

**Algorithm libraries:**
- river (online learning)
- ruptures (change point detection)
- Various specialized CPD libraries (see requirements.txt)

**Installation:**
```bash
pip install -r requirements.txt
```

## Inter-Annotator Agreement

For real-world data, multiple annotators may label the same series. The framework includes tools to:

1. Calculate F1-based agreement between annotators
2. Determine if multiple annotations can be combined
3. Recommend using single annotator if agreement is low

**Finding:** Inter-annotator agreement F1=0.24 indicates substantial disagreement, recommending single-annotator approach.

## Computational Considerations

**Grid Search Complexity:**
- Synthetic data: O(n_scenarios × n_algorithms × n_param_combinations × n_series)
- Real data: O(n_algorithms × n_param_combinations × n_series)

**Recommended Approach:**
1. Use "quick" profile for initial exploration
2. Use transfer learning for fast real-data evaluation
3. Use grid search on real data for final optimization (if needed)

**Time Estimates:**
- Synthetic benchmark (quick): 30-60 minutes
- Real data grid search (quick): 1-2 hours
- Transfer learning: 5-10 minutes

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{online_cpd_pipeline,
  author = {Allan},
  title = {Online Change Point Detection Pipeline},
  year = {2025},
  url = {https://github.com/AllanDBB/online-cpd-pipeline}
}
```

## License

[Specify license here]

## Contributing

Contributions are welcome. Please submit pull requests with:
- Clear description of changes
- Test cases for new algorithms
- Documentation updates

## Contact

[Contact information]

## Acknowledgments

This framework builds upon multiple open-source libraries and benchmark datasets from the change point detection community.
