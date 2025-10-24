# Benchmark Documentation

## Table of Contents

1. [Synthetic Data Benchmark](#synthetic-data-benchmark)
2. [Real Data Benchmark](#real-data-benchmark)
3. [Transfer Learning Benchmark](#transfer-learning-benchmark)
4. [TCPD Benchmark](#tcpd-benchmark)

---

## Synthetic Data Benchmark

**Script:** `benchmark_synthetic_data.py`

### Purpose

Evaluates online change point detection algorithms on synthetically generated time series data with known ground truth change points. This benchmark provides a controlled environment to assess algorithm performance across various scenarios.

### Data Generation

**Synthetic Time Series Characteristics:**

- **Base signal:** AR(1) process or random walk
- **Length:** 200-400 time steps (configurable)
- **Noise levels:** Three categories
  - Low: SNR > 10
  - Medium: 5 < SNR < 10
  - High: SNR < 5

- **Change types:**
  - Step change: Abrupt shift in mean
  - Slope change: Gradual trend modification
  - Variance change: Volatility shift

- **Change magnitude:**
  - Low: 0.5-1.0 standard deviations
  - Medium: 1.0-2.0 standard deviations
  - High: > 2.0 standard deviations

### Experimental Design

**Test Scenarios:**

The benchmark generates a factorial design covering:
- 3 noise levels × 3 change magnitudes × 3 change types = 27 scenarios
- Each scenario: 45 series (3 iterations × 15 series per iteration)
- Total: 1,215 time series

**Train/Test Split:**

- Training set: 70% of series (used for hyperparameter optimization)
- Test set: 30% of series (used for final evaluation)
- Stratified split maintains scenario distribution

### Algorithm Evaluation

**Grid Search Protocol:**

1. For each algorithm, enumerate parameter combinations from predefined grid
2. For each parameter combination:
   - Apply algorithm to all training series
   - Calculate F1 score, precision, recall, MMD, MTTD
   - Average metrics across training series
3. Select parameter combination with highest mean F1 score
4. Evaluate best parameters on test set

**Timeout Handling:**

- Each algorithm has 30-second timeout per series
- Timeout failures recorded as errors
- Failed series excluded from metric calculations

### Output Files

**Main Results CSV:**

`results/results_synthetic/{date}-resultados_algoritmos_main2-synthetic.csv`

Columns include:
- Scenario metadata (noise level, change magnitude, change type)
- Algorithm identification (key, library, method)
- Configuration metadata (iterations, series count, delta)
- Parameter configuration (JSON serialized)
- Training metrics (F1, precision, recall, MMD, MTTD, detections)
- Test metrics (same as training)
- Status flag and trial identifier
- Best configuration indicator (is_best boolean)

**Best Series JSON Files:**

`results/results_synthetic/best_series_{scenario}_{algorithm}_trial{n}.json`

Contains:
- Complete time series data
- Ground truth change points
- Detected change points
- All evaluation metrics
- Algorithm parameters used

### Configuration Options

```python
CONFIG = {
    "seed": 123,                    # Reproducibility
    "profile": "quick",             # "quick" or "full"
    "series_length": 280,           # Base series length
    "series_std": 80,               # Length variation
    "iterations": 3,                # Repetitions per scenario
    "series_per_iteration": 15,     # Series per repetition
    "delta_eval": 10,               # Temporal tolerance
    "results_csv": "resultados_algoritmos_main2-synthetic.csv",
}
```

**Profile Comparison:**

- **Quick:** Coarse grid, fewer combinations (faster, exploratory)
- **Full:** Fine grid, extensive combinations (slower, thorough)

### Usage Example

```bash
# Standard execution
python benchmark_synthetic_data.py

# Modify configuration by editing CONFIG dictionary in script
# Then run again
```

### Computational Requirements

**Estimated Runtime:**

- Quick profile: 30-60 minutes (standard laptop)
- Full profile: 4-8 hours (depends on hardware)

**Memory:**

- Peak: ~2-4 GB RAM
- Depends on series length and algorithm complexity

### Results Interpretation

**Key Metrics:**

- **test_f1_mean:** Primary metric for ranking algorithms
- **test_precision_mean:** Proportion of correct detections
- **test_recall_mean:** Proportion of true changes detected
- **test_mmd_mean:** Distribution similarity measure
- **test_mttd_mean:** Average detection delay

**Best Practice:**

1. Sort results by test_f1_mean (descending)
2. Identify top-performing algorithms per scenario
3. Examine parameter sensitivity
4. Check for overfitting (train vs test performance)

---

## Real Data Benchmark

**Script:** `benchmark_real_data.py`

### Purpose

Evaluates algorithms on manually labeled real-world time series data, specifically crime statistics from Costa Rica. Unlike synthetic data, these series contain natural noise patterns and realistic change point characteristics.

### Dataset Description

**Data Source:**

- Crime statistics time series (monthly aggregation)
- Domain: Public safety / criminology
- Length: 120 time steps (10 years of monthly data)
- Series count: 49 labeled by primary annotator

**Labeling Process:**

- Manual annotation by domain experts
- Change points marked with temporal precision
- Metadata includes change type and confidence level
- Inter-annotator agreement analysis performed (F1=0.24)

**Labeling Configuration:**

Due to low inter-annotator agreement, the benchmark uses single annotator (Martin) by default. This can be configured:

```python
CONFIG = {
    "use_only_martin_labels": True,  # Recommended
    # Set to False to use all annotators
}
```

### Series Classification

Before benchmarking, series are automatically classified by:

**Noise Category:**

Calculated using signal-to-noise ratio (SNR):
- Low noise: SNR > threshold_high
- Medium noise: threshold_low < SNR < threshold_high
- High noise: SNR < threshold_low

**Change Category:**

Based on magnitude of detected changes:
- Low magnitude: < threshold_low
- Medium magnitude: threshold_low to threshold_high
- High magnitude: > threshold_high

Classification results saved to:
`results/results_real/{date}-clasificacion_series_criminalidad.csv`

### Experimental Protocol

**Data Split:**

- Training: 50% of labeled series
- Testing: 50% of labeled series
- Stratified by series characteristics when possible

**Hyperparameter Optimization:**

1. Grid search on training set
2. Select best parameters by mean F1 score
3. Apply best parameters to test set
4. Record both training and test performance

**Evaluation Metrics:**

Same as synthetic benchmark:
- F1 score with temporal tolerance (delta=10 by default)
- Precision and recall
- MMD and MTTD
- Confusion matrix elements

### Output Files

**Results CSV:**

`results/results_real/{date}-resultados_algoritmos_main3_real-real.csv`

Structure similar to synthetic results, with additions:
- Annotator information
- Series classification metadata
- Real-world specific notes

**Best Series Analysis JSON:**

`results/results_real/{date}-best_series_analysis_real_data.json`

Contains:
- Configuration used (including annotator selection)
- Per-algorithm best parameters
- Training and test performance
- Detailed per-series results
- Dataset information and classification

**Series Classification CSV:**

`results/results_real/{date}-clasificacion_series_criminalidad.csv`

Includes:
- Series identifier
- Noise category and magnitude
- Change category and magnitude
- Statistical measures (SNR, variance, etc.)

### Configuration

```python
CONFIG = {
    "seed": 123,
    "profile": "quick",
    "data_real_path": "data/data_real",
    "delta_eval": 10,
    "results_csv": "resultados_algoritmos_main3_real.csv",
    "use_only_martin_labels": True,  # Annotator selection
    # Algorithm-specific parameter grids...
}
```

### Special Considerations

**Domain-Specific Challenges:**

1. **Temporal dependencies:** Crime data has seasonality and trends
2. **External factors:** Events not reflected in data
3. **Measurement noise:** Reporting delays, data collection issues
4. **Change point ambiguity:** Gradual vs abrupt transitions unclear

**Handling Missing Data:**

- Series with < 120 observations excluded
- No interpolation applied (preserves real characteristics)

### Usage

```bash
python benchmark_real_data.py
```

**Note:** Grid search on real data can be time-consuming. Consider:
1. Using "quick" profile initially
2. Running transfer learning first for fast baseline
3. Using grid search for final optimization

### Results Interpretation

**Performance Expectations:**

Real data F1 scores typically lower than synthetic:
- Excellent: F1 > 0.4
- Good: 0.3 < F1 < 0.4
- Moderate: 0.2 < F1 < 0.3
- Poor: F1 < 0.2

**Analysis Recommendations:**

1. Compare performance across series classification categories
2. Examine false positives for interpretability
3. Consider domain knowledge when evaluating detections
4. Analyze failure cases for algorithm limitations

---

## Transfer Learning Benchmark

**Script:** `benchmark_real_data_transfer_learning.py`

### Purpose

Investigates whether optimal parameters found on synthetic data transfer effectively to real data, potentially enabling rapid algorithm evaluation without expensive hyperparameter search.

### Methodology

**Transfer Learning Protocol:**

1. **Parameter Extraction:**
   - Load synthetic benchmark results
   - Extract best parameters per algorithm
   - Use first encountered best configuration if multiple exist

2. **Direct Application:**
   - Apply synthetic-optimized parameters to real data
   - No grid search performed
   - Evaluate on same train/test split as real data benchmark

3. **Performance Comparison:**
   - Compare transfer learning results with grid search results
   - Calculate correlation between synthetic and real performance
   - Identify algorithms robust to domain shift

### Execution Modes

**Interactive Mode:**

```bash
python benchmark_real_data_transfer_learning.py interactive
```

Guides user through:
1. Prerequisites check (synthetic results exist)
2. Transfer learning benchmark execution
3. Comparison with grid search (if available)

**Benchmark Mode:**

```bash
python benchmark_real_data_transfer_learning.py benchmark
```

Runs transfer learning evaluation only.

**Comparison Mode:**

```bash
python benchmark_real_data_transfer_learning.py compare
```

Compares existing transfer learning and grid search results.

### Output Files

**Transfer Learning Results:**

`{date}_transfer_learning_real_data_results.csv`

Columns include all standard metrics plus:
- approach: "transfer_learning"
- params_source: "synthetic_best"
- synthetic_scenario: Source scenario from synthetic data
- synthetic_test_f1: Performance on synthetic data

**Detailed Per-Algorithm JSON:**

`{date}_transfer_learning_{algorithm}_detailed.json`

Contains:
- Parameters used (from synthetic)
- Synthetic performance baseline
- Complete training evaluation
- Complete test evaluation
- Per-series detailed results

**Comparison Results:**

`{date}_comparison_grid_vs_transfer.csv`

Side-by-side comparison with columns:
- algorithm: Algorithm identifier
- grid_test_f1: Grid search test F1
- transfer_test_f1: Transfer learning test F1
- f1_difference: Difference (transfer - grid)
- f1_improvement_pct: Percentage improvement
- grid_params: Grid search parameters (JSON)
- transfer_params: Transfer parameters (JSON)
- synthetic_test_f1: Synthetic baseline

**Comparison Report:**

`{date}_comparison_report.txt`

Human-readable report including:
- Executive summary statistics
- Algorithm-by-algorithm breakdown
- Winner identification
- Recommendations for approach selection

### Analysis Components

**Correlation Analysis:**

Calculates Pearson correlation between:
- Synthetic F1 vs Real Transfer F1
- Synthetic F1 vs Real Grid Search F1

Interpretation:
- r > 0.7: Strong correlation (synthetic predictive)
- 0.4 < r < 0.7: Moderate correlation (some transfer)
- r < 0.4: Weak correlation (significant domain shift)

**Performance Comparison:**

For each algorithm:
- Identifies winner (transfer vs grid)
- Calculates performance gap
- Determines statistical significance (when applicable)

### Research Questions Addressed

1. **Parameter Transferability:**
   - Do synthetic-optimized parameters generalize to real data?
   - Which algorithms are most robust to domain shift?

2. **Computational Efficiency:**
   - Time saved by avoiding grid search
   - Performance trade-off quantification

3. **Algorithm Selection:**
   - Can synthetic data guide real-world algorithm choice?
   - Correlation strength between domains

### Usage Recommendations

**When to Use Transfer Learning:**

1. Initial algorithm screening (fast exploration)
2. Resource-constrained scenarios
3. When grid search is impractical

**When to Use Grid Search:**

1. Final deployment (optimal performance needed)
2. Domain-critical applications
3. When computational resources available

**Hybrid Approach:**

1. Use transfer learning to identify top 3-5 algorithms
2. Apply grid search only to promising candidates
3. Achieves 70-80% time savings with minimal performance loss

### Results Interpretation

**Key Findings from Validation:**

- Mean F1 difference: Grid search 0.06 points higher
- Correlation: 0.71 (strong positive)
- Transfer wins: 7% of algorithms
- Grid wins: 73% of algorithms
- Tied: 20% of algorithms

**Practical Implications:**

- Transfer learning provides 85% of grid search quality
- Execution time: minutes vs hours
- Suitable for rapid prototyping and algorithm selection
- Grid search recommended for production deployment

---

## TCPD Benchmark

**Script:** `benchmark_tcpd_data.py`

### Purpose

Evaluates algorithms on the Time Series Change Point Detection (TCPD) benchmark dataset, a standardized collection widely used in the research community.

### Dataset Description

**TCPD Benchmark Suite:**

- Curated collection of real-world and synthetic series
- Multiple domains (finance, climate, sensor data, etc.)
- Pre-labeled change points (expert annotated)
- Varying series lengths and characteristics

**Data Location:**

```
data/TCDP-paper/
```

### Evaluation Protocol

Similar to other benchmarks but with TCPD-specific considerations:

1. Load pre-labeled TCPD series
2. Apply algorithms with standard or optimized parameters
3. Evaluate using temporal tolerance matching
4. Compare against published baselines when available

### Output Files

Results follow standard format:
- CSV with algorithm performance metrics
- JSON with detailed per-series results
- Comparison with published benchmarks (when available)

### Usage

```bash
python benchmark_tcpd_data.py
```

### Special Notes

**Comparison with Literature:**

TCPD benchmark enables direct comparison with:
- Published algorithm results
- State-of-the-art methods
- Domain-specific approaches

**Limitations:**

- Offline algorithms may have advantage (full series access)
- Online constraint must be explicitly enforced
- Different papers may use different evaluation metrics

---

## Cross-Benchmark Analysis

### Comparing Results Across Benchmarks

**Synthetic vs Real:**

- Synthetic provides controlled evaluation
- Real validates practical applicability
- Gap indicates domain shift magnitude

**Transfer Learning vs Grid Search:**

- Measures parameter generalization
- Quantifies computational trade-off
- Guides deployment decisions

**TCPD Comparison:**

- Validates against community standards
- Enables literature comparison
- Identifies algorithm limitations

### Unified Results Analysis

All benchmarks produce compatible CSV formats enabling:
- Aggregated performance analysis
- Algorithm ranking across datasets
- Robustness assessment
- Meta-analysis of algorithm characteristics

---

## Best Practices

1. **Always run synthetic benchmark first** (establishes baselines)
2. **Use transfer learning for initial real data exploration**
3. **Apply grid search selectively** (based on transfer learning results)
4. **Document configuration** (ensure reproducibility)
5. **Version control results** (track experiments systematically)
6. **Analyze errors** (not just aggregate metrics)
7. **Consider domain context** (numerical performance not everything)

---

## Troubleshooting

**Common Issues:**

1. **Timeout errors:**
   - Increase timeout limit
   - Reduce series length
   - Simplify parameter grid

2. **Memory errors:**
   - Process series in batches
   - Reduce concurrent operations
   - Use streaming where possible

3. **Missing dependencies:**
   - Install all requirements
   - Check version compatibility
   - Use virtual environment

4. **Inconsistent results:**
   - Set random seed
   - Check data preprocessing
   - Verify parameter serialization

---

## References

1. Synthetic data generation based on standard time series models
2. F1 score with tolerance: Adapted from event detection literature
3. TCPD benchmark: [cite original paper]
4. Transfer learning validation: Novel contribution of this framework
