# Online CPD Pipeline

This repository hosts a synthetic change point detection (CPD) benchmarking suite.
`main_v2.py` orchestrates the end-to-end pipeline: synthetic data generation, grid-search
hyper-parameter tuning across multiple detectors, metric aggregation, and CSV logging.

## Synthetic Data Generation
- Series are produced with `src/allan_synthetic.py` following Allan_synthetic rules.
- Controls exposed at the top of `main_v2.py` under `CONFIG`:
  - Noise level: `alto` or `bajo`.
  - Change strength: `alto` or `bajo`.
  - Change type: `escalon` (step) or `pendiente` (trend).
- Every run enumerates the 2 × 2 × 2 = 8 combinations. For each combination the
  generator draws:
  - Series length from `series_length_choices`.
  - Number of change points from `n_changes_range`.
  - Random seeds derived from the global `seed` to keep runs reproducible.

## Profiles: Quick vs Extensive
`CONFIG["profile"]` selects how much data and how many hyper-parameter combinations to evaluate.

| Profile | Iterations × series per combination | Typical runtime | Notes |
|---------|-------------------------------------|-----------------|-------|
| `quick` *(default)* | 3 × 15 = 45 series | ~20 minutes (reference) | Suitable for debugging or smoke tests. |
| `extensive` | 8 × 50 = 400 series | ~1 hour (target) | Adds longer series, more change points, and wider grids. |

Switch profiles by editing the value of `CONFIG["profile"]`. The helper `load_config()`
merges the chosen preset with the base configuration and also expands the per-algorithm
grids for the `extensive` runs.

Key knobs to customise per run:
- `n_iterations`: resampling loops per combo.
- `series_per_combo`: number of series generated per loop.
- `series_length_choices`: possible sequence lengths.
- `n_changes_range`: min/max number of change points.
- `algorithm_configs`: hyper-parameter search space per detector.

## Implemented Detectors
All detectors expose a `detect_changepoints_*` function inside `src/algorithms/`.
They are wrapped so that each accepts a 1D `numpy.ndarray` and returns a list of indices.

| Key | Library/Origin | Module | Notes |
|-----|----------------|--------|-------|
| `page_hinkley_river` | `river` | `src/algorithms/page_hinkley.py` | Streaming Page-Hinkley (baseline online). |
| `ewma_numpy` | custom | `src/algorithms/ewma.py` | Lightweight EWMA thresholding. |
| `changepoint_online_focus` | `ruptures` | `changepoint_online.py` | PELT + RBF cost (Focus). |
| `changepoint_online_gaussian` | `ruptures` | `changepoint_online.py` | PELT + L2 cost (Gaussian). |
| `changepoint_online_np_focus` | `ruptures` | `changepoint_online.py` | Window-based non-parametric focus. |
| `changepoint_online_md_focus` | `ruptures` | `changepoint_online.py` | Binary segmentation (multivariate-ready). |
| `ocpdet_cumsum` | custom | `ocpdet.py` | Classic CUSUM detector. |
| `ocpdet_ewma` | custom | `ocpdet.py` | EWMA wrapper with adaptive params. |
| `ocpdet_two_sample_tests` | custom | `ocpdet.py` | Sliding Kolmogorov–Smirnov test. |
| `ocpdet_neural_networks` | `scikit-learn` | `ocpdet.py` | MLP autoregression residuals. |
| `ssm_canary` | `filterpy` | `state_space.py` | Kalman filter with fixed process noise. |
| `tagi_lstm_ssm` | `filterpy` | `state_space.py` | Kalman variant with adaptive covariance. |
| `skf_kalman_canary` | `filterpy` | `state_space.py` | Kalman filter tuned for higher noise. |
| `bayesian_online_cpd_cpfinder` | `bayesian_changepoint_detection` | `bayesian_online.py` | BOCPD with Student-T likelihood. |
| `changefinder_sdar` | `changefinder` | `changefinder_detector.py` | SDAR anomaly scoring with z-threshold. |
| `rulsif_roerich` | `roerich` | `rulsif_detector.py` | Online RuLSIF neural density ratio. |

## Metrics and Outputs
For every combination × algorithm × hyper-parameter set the pipeline computes:
- **F1 score** (`f1_mean`): tolerance-based matching using `src/f1_score.py`.
- **Precision / Recall**: per-series mean.
- **MMD** (`mmd_mean`): Gaussian-kernel maximum mean discrepancy, see `src/mmd.py`.
- **MTTD** (`mttd_mean`): mean time-to-detection in samples.
- Counts of detections (`tp_mean`, `fp_mean`, `fn_mean`, `detections_mean`).

Results are appended to `resultados_algoritmos_main2.csv` with the following fields:
- Labels: `nivel_ruido`, `fuerza_cambio`, `tipo_cambio`.
- Algorithm descriptors and the JSON encoded hyper-parameters under `params_json`.
- `trial_id`: index of the grid-search combination.
- `is_best`: marks the highest-F1 configuration (ties broken by lower MMD).

## Running the Pipeline
1. Adjust `CONFIG` (profile or manual tweaks).
2. Ensure dependencies are installed (see `requirements` summary below).
3. Execute `python main_v2.py`.
4. Inspect console output for the top configurations and analyse the CSV.

### Dependencies
Key third-party packages used in the new pipeline:
- `numpy`, `pandas`
- `scikit-learn`
- `river`
- `ruptures`
- `changefinder`
- `roerich`
- `bayesian-changepoint-detection`
- `filterpy`

Install them via pip (exact versions depend on your environment):

```bash
pip install numpy pandas scikit-learn river ruptures changefinder roerich \
    bayesian-changepoint-detection filterpy
```

## Design Notes
- `main_v2.py` keeps detector definitions and grid spaces declarative, so adding a new
  algorithm only requires implementing a wrapper in `src/algorithms/` and registering it
  in `ALGORITHM_TEMPLATES`.
- The evaluation loop stores every trial (not only the winners) to support later
  hyper-parameter analysis.
- Profiles allow you to keep a “small” run for iteration and a heavy-weight run for
  final benchmark sweeps without editing every knob manually.
