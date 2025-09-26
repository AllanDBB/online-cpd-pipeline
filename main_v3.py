"""Main 3 pipeline for testing change-point algorithms on real labeled data."""
from __future__ import annotations

import itertools
import json
import os
import copy
import glob
import signal
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple
from contextlib import contextmanager

import numpy as np
import pandas as pd

from src.f1_score import f1_score_with_tolerance
from src.mttd import mean_time_to_detection
from src.mmd import maximum_mean_discrepancy
from src.algorithms.page_hinkley import detect_changepoints_page_hinkley
from src.algorithms.ewma import detect_changepoints_ewma
from src.algorithms.changepoint_online import (
    detect_changepoints_focus,
    detect_changepoints_gaussian,
    detect_changepoints_md_focus,
    detect_changepoints_np_focus,
)
from src.algorithms.ocpdet import (
    detect_changepoints_cumsum,
    detect_changepoints_ewma_wrapper,
    detect_changepoints_neural,
    detect_changepoints_two_sample,
)
from src.algorithms.state_space import (
    detect_changepoints_skf_canary,
    detect_changepoints_ssm_canary,
    detect_changepoints_tagi_canary,
)
from src.algorithms.bayesian_online import detect_changepoints_cpfinder
from src.algorithms.changefinder_detector import detect_changepoints_changefinder
from src.algorithms.rulsif_detector import detect_changepoints_rulsif

DetectorFn = Callable[..., List[int]]


@dataclass
class AlgorithmSpec:
    key: str
    library: str
    method: str
    detect_fn: DetectorFn | None
    param_grid: Dict[str, Sequence[Any]]
    supervision: str
    is_univariate: bool
    notes: str = ""
    implemented: bool = True

    def iter_param_grid(self) -> Iterable[Dict[str, Any]]:
        if not self.param_grid:
            yield {}
            return
        keys = list(self.param_grid.keys())
        value_lists = [self.param_grid[k] for k in keys]
        for combo in itertools.product(*value_lists):
            params = {key: value for key, value in zip(keys, combo)}
            cleaned = {k: v for k, v in params.items() if v is not None}
            yield cleaned


CONFIG: Dict[str, Any] = {
    "seed": 123,
    "profile": "quick",
    "data_real_path": "data/data_real",
    "delta_eval": 10,
    "results_csv": "resultados_algoritmos_main3_real.csv",
    "algorithm_configs": {
        "page_hinkley_river": {
            "grid": {
                "threshold": [20, 40, 60, 80],
                "min_instances": [5, 10, 20],
                "delta": [0.001, 0.005, 0.01],
            }
        },
        "ewma_numpy": {
            "grid": {
                "alpha": [0.05, 0.1, 0.2, 0.3],
                "threshold": [2.0, 2.5, 3.0],
                "min_instances": [5, 10, 15],
            }
        },
        "changepoint_online_focus": {
            "grid": {
                "penalty": [10.0, 20.0, 30.0],
                "min_size": [10, 15, 25],
                "jump": [2, 3, 5],
            }
        },
        "changepoint_online_gaussian": {
            "grid": {
                "penalty": [10.0, 20.0, 30.0],
                "min_size": [15, 20, 25],
                "jump": [2, 3, 5],
            }
        },
        "changepoint_online_np_focus": {
            "grid": {
                "penalty": [10.0, 20.0, 30.0],
                "width": [20, 30, 50],
                "jump": [2, 3],
            }
        },
        "changepoint_online_md_focus": {
            "grid": {
                "penalty": [10.0, 20.0, 30.0],
                "min_size": [15, 20, 30],
                "jump": [2, 3],
            }
        },
        "ocpdet_cumsum": {
            "grid": {
                "threshold": [4.0, 6.0, 8.0, 10.0],
                "drift": [0.0, 0.1],
                "min_distance": [15, 20, 30],
            }
        },
        "ocpdet_ewma": {
            "grid": {
                "alpha": [0.05, 0.1, 0.2],
                "threshold": [2.0, 2.5, 3.0],
                "min_instances": [5, 10],
            }
        },
        "ocpdet_two_sample_tests": {
            "grid": {
                "window_size": [20, 30, 40],
                "step": [3, 5],
                "alpha": [0.01, 0.05],
                "min_distance": [15, 20, 30],
            }
        },
        "ocpdet_neural_networks": {
            "grid": {
                "window_size": [15, 20],
                "step": [1, 2],
                "hidden_layer_sizes": [(15,), (20,), (25, 15)],
                "threshold": [2.0, 2.5],
                "min_distance": [20, 30],
            }
        },
        "ssm_canary": {
            "grid": {
                "process_noise": [1e-3, 5e-3, 1e-2],
                "measurement_noise": [0.3, 0.5, 0.8],
                "threshold": [2.0, 2.5, 3.0],
                "min_distance": [15, 25],
            }
        },
        "tagi_lstm_ssm": {
            "grid": {
                "process_noise": [1e-4, 2e-4, 5e-4],
                "measurement_noise": [0.2, 0.4, 0.6],
                "threshold": [2.0, 2.5, 3.0],
                "min_distance": [15, 25],
                "adaptation": [1e-5, 5e-5],
            }
        },
        "skf_kalman_canary": {
            "grid": {
                "process_noise": [5e-3, 1e-2, 2e-2],
                "measurement_noise": [0.5, 0.8, 1.0],
                "threshold": [2.5, 3.0, 3.5],
                "min_distance": [20, 30],
            }
        },
        "bayesian_online_cpd_cpfinder": {
            "grid": {
                "hazard_lambda": [100.0, 150.0, 300.0],
                "alpha": [0.1, 0.2, 0.3],
                "beta": [0.01, 0.05],
                "kappa": [1.0, 2.0],
                "probability_threshold": [0.5, 0.7],
                "min_distance": [15, 25],
            }
        },
        "changefinder_sdar": {
            "grid": {
                "r": [0.3],
                "order": [1],
                "smooth": [5],
                "threshold": [2.0],
                "min_distance": [20],
            }
        },
        "rulsif_roerich": {
            "grid": {
                "window_size": [5],
                "lag_size": [40],
                "step": [1],
                "n_epochs": [1],
                "threshold": [0.08],
                "min_distance": [20],
                "alpha": [0.05],
            }
        },
    },
}

ALGORITHM_TEMPLATES: List[Dict[str, Any]] = [
    {
        "key": "page_hinkley_river",
        "library": "river",
        "method": "PageHinkley",
        "detect_fn": detect_changepoints_page_hinkley,
        "implemented": True,
        "supervision": "no_supervisado",
        "is_univariate": True,
        "notes": "Page-Hinkley de river.",
    },
    {
        "key": "ewma_numpy",
        "library": "EWMA",
        "method": "ewma",
        "detect_fn": detect_changepoints_ewma,
        "implemented": True,
        "supervision": "no_supervisado",
        "is_univariate": True,
        "notes": "EWMA ligero NumPy.",
    },
    {
        "key": "changepoint_online_focus",
        "library": "changepoint_online",
        "method": "Focus",
        "detect_fn": detect_changepoints_focus,
        "implemented": True,
        "supervision": "no_supervisado",
        "is_univariate": True,
        "notes": "Ruptures PELT con costo RBF.",
    },
    {
        "key": "changepoint_online_gaussian",
        "library": "changepoint_online",
        "method": "Gaussian",
        "detect_fn": detect_changepoints_gaussian,
        "implemented": True,
        "supervision": "no_supervisado",
        "is_univariate": True,
        "notes": "Ruptures PELT con costo L2.",
    },
    {
        "key": "changepoint_online_np_focus",
        "library": "changepoint_online",
        "method": "NPFocus",
        "detect_fn": detect_changepoints_np_focus,
        "implemented": True,
        "supervision": "no_supervisado",
        "is_univariate": True,
        "notes": "Detector no parametrico tipo ventana.",
    },
    {
        "key": "changepoint_online_md_focus",
        "library": "changepoint_online",
        "method": "MDFocus",
        "detect_fn": detect_changepoints_md_focus,
        "implemented": True,
        "supervision": "no_supervisado",
        "is_univariate": False,
        "notes": "Binary segmentation RBF (aprox multivariado).",
    },
    {
        "key": "ocpdet_cumsum",
        "library": "OCPDet",
        "method": "cumsum",
        "detect_fn": detect_changepoints_cumsum,
        "implemented": True,
        "supervision": "no_supervisado",
        "is_univariate": True,
        "notes": "CUSUM clasico.",
    },
    {
        "key": "ocpdet_ewma",
        "library": "OCPDet",
        "method": "ewma",
        "detect_fn": detect_changepoints_ewma_wrapper,
        "implemented": True,
        "supervision": "no_supervisado",
        "is_univariate": True,
        "notes": "EWMA adaptativo.",
    },
    {
        "key": "ocpdet_two_sample_tests",
        "library": "OCPDet",
        "method": "two sample tests",
        "detect_fn": detect_changepoints_two_sample,
        "implemented": True,
        "supervision": "no_supervisado",
        "is_univariate": True,
        "notes": "Ventanas con KS test.",
    },
    {
        "key": "ocpdet_neural_networks",
        "library": "OCPDet",
        "method": "neural networks",
        "detect_fn": detect_changepoints_neural,
        "implemented": True,
        "supervision": "no_supervisado",
        "is_univariate": True,
        "notes": "Auto-regresion MLP residual.",
    },
    {
        "key": "ssm_canary",
        "library": "SSM",
        "method": "canary",
        "detect_fn": detect_changepoints_ssm_canary,
        "implemented": True,
        "supervision": "semi_supervisado",
        "is_univariate": True,
        "notes": "Filtro de Kalman basico.",
    },
    {
        "key": "tagi_lstm_ssm",
        "library": "TAGI-LSTM/SSM",
        "method": "canary",
        "detect_fn": detect_changepoints_tagi_canary,
        "implemented": True,
        "supervision": "semi_supervisado",
        "is_univariate": True,
        "notes": "Filtro Kalman adaptativo (TAGI).",
    },
    {
        "key": "skf_kalman_canary",
        "library": "SKF",
        "method": "canary",
        "detect_fn": detect_changepoints_skf_canary,
        "implemented": True,
        "supervision": "semi_supervisado",
        "is_univariate": True,
        "notes": "Filtro Kalman robusto.",
    },
    {
        "key": "bayesian_online_cpd_cpfinder",
        "library": "Bayesian Online Change Point Detection",
        "method": "cpfinder",
        "detect_fn": detect_changepoints_cpfinder,
        "implemented": True,
        "supervision": "no_supervisado",
        "is_univariate": True,
        "notes": "BCPD con Student-T.",
    },
    {
        "key": "changefinder_sdar",
        "library": "ChangeFinder",
        "method": "changefinder",
        "detect_fn": detect_changepoints_changefinder,
        "implemented": False,  # Temporarily disabled due to numerical issues
        "supervision": "no_supervisado",
        "is_univariate": True,
        "notes": "ChangeFinder SDAR - DISABLED due to numerical warnings.",
    },
    {
        "key": "rulsif_roerich",
        "library": "RuLSIF",
        "method": "Roerich",
        "detect_fn": detect_changepoints_rulsif,
        "implemented": False,  # Temporarily disabled due to hanging
        "supervision": "no_supervisado",
        "is_univariate": False,
        "notes": "Roerich Online NN RuLSIF - DISABLED due to hanging issues.",
    },
]


def build_algorithm_specs(config: Dict[str, Any]) -> List[AlgorithmSpec]:
    """Build algorithm specifications with parameter grids."""
    overrides = config.get("algorithm_configs", {})
    specs: List[AlgorithmSpec] = []
    for template in ALGORITHM_TEMPLATES:
        key = template["key"]
        override = overrides.get(key, {})
        param_grid = override.get("grid", {})
        notes = override.get("notes", template.get("notes", ""))
        detect_fn = template.get("detect_fn")
        implemented = template.get("implemented", True) and detect_fn is not None
        specs.append(
            AlgorithmSpec(
                key=key,
                library=template["library"],
                method=template["method"],
                detect_fn=detect_fn if implemented else None,  
                param_grid=param_grid,
                supervision=template.get("supervision", ""),
                is_univariate=template.get("is_univariate", True),
                notes=notes,
                implemented=implemented,
            )
        )
    return specs


def _safe_mean(values: Iterable[Any]) -> float | None:
    """Calculate mean safely, handling None values."""
    filtered = [float(v) for v in values if v is not None]
    if not filtered:
        return None
    return float(np.mean(filtered))


def _series_effective_length(series: np.ndarray) -> int:
    """Calculate effective length of series (non-NaN values)."""
    arr = np.asarray(series, dtype=float)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    valid = ~np.isnan(arr)
    return int(np.count_nonzero(valid))


@contextmanager
def timeout_context(seconds: int):
    """Context manager for timeout handling."""
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Algorithm timed out after {seconds} seconds")
    
    # Set the signal handler
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        # Restore the old signal handler
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def safe_algorithm_call(detect_fn: Callable, serie: np.ndarray, params: Dict[str, Any], 
                       algorithm_key: str, timeout_seconds: int = 60) -> List[int]:
    """Safely call detection algorithm with timeout and error handling."""
    try:
        # Suppress warnings for problematic algorithms
        with warnings.catch_warnings():
            if algorithm_key in ['changefinder_sdar', 'rulsif_roerich']:
                warnings.simplefilter("ignore")
            
            # For Windows, we can't use signal-based timeout, so we'll use a simpler approach
            detected = detect_fn(serie, **params)
            return detected if detected is not None else []
            
    except TimeoutError:
        print(f"  WARNING: {algorithm_key} timed out after {timeout_seconds} seconds")
        return []
    except Exception as e:
        print(f"  WARNING: {algorithm_key} failed with error: {str(e)[:100]}...")
        return []


def train_test_split_real_data(datasets: List[Dict[str, Any]], test_size: float = 0.2, seed: int = 42) -> Dict[str, List[Dict[str, Any]]]:
    """Split real datasets into train and test sets."""
    np.random.seed(seed)
    
    # Shuffle datasets
    shuffled_datasets = datasets.copy()
    np.random.shuffle(shuffled_datasets)
    
    # Calculate split point
    n_test = max(1, int(len(shuffled_datasets) * test_size))
    n_train = len(shuffled_datasets) - n_test
    
    train_data = shuffled_datasets[:n_train]
    test_data = shuffled_datasets[n_train:]
    
    print(f"Train/Test split: {len(train_data)} train, {len(test_data)} test series")
    
    return {
        'train': train_data,
        'test': test_data
    }


def load_real_data(data_path: str) -> List[Dict[str, Any]]:
    """Load real labeled data from CSV files."""
    csv_files = glob.glob(os.path.join(data_path, "*.csv"))
    datasets = []
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            
            # Extract series data
            series = df['Value'].values.astype(float)
            
            # Extract changepoints (where Is_ChangePoint is TRUE)
            changepoints = df[df['Is_ChangePoint'] == True]['Index'].tolist()
            
            # Extract metadata from filename
            filename = os.path.basename(csv_file)
            if 'allan_' in filename:
                annotator = 'allan'
            elif 'Mart_n_Sol_s_Salazar_' in filename:
                annotator = 'martin'
            else:
                annotator = 'unknown'
            
            # Extract series ID from filename
            import re
            series_match = re.search(r'-s(\d+)_', filename)
            series_id = int(series_match.group(1)) if series_match else 0
            
            datasets.append({
                'filename': filename,
                'annotator': annotator,
                'series_id': series_id,
                'series': series,
                'changepoints': changepoints,
                'length': len(series),
                'n_changepoints': len(changepoints),
                'metadata': {
                    'changepoint_types': df[df['Is_ChangePoint'] == True]['ChangePoint_Type'].tolist(),
                    'changepoint_confidence': df[df['Is_ChangePoint'] == True]['ChangePoint_Confidence'].tolist(),
                    'changepoint_notes': df[df['Is_ChangePoint'] == True]['ChangePoint_Notes'].tolist(),
                }
            })
            
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")
            continue
    
    print(f"Loaded {len(datasets)} real data series from {data_path}")
    print(f"Annotators found: {set(d['annotator'] for d in datasets)}")
    print(f"Series lengths range: {min(d['length'] for d in datasets)} - {max(d['length'] for d in datasets)}")
    print(f"Changepoints per series range: {min(d['n_changepoints'] for d in datasets)} - {max(d['n_changepoints'] for d in datasets)}")
    
    return datasets


def find_best_params_on_train_data(
    spec: AlgorithmSpec,
    train_datasets: List[Dict[str, Any]],
    delta_eval: int,
) -> Dict[str, Any]:
    """Find best parameters using grid search on training data."""
    records: List[Dict[str, Any]] = []
    best_index: int | None = None
    best_score = -np.inf
    best_mmd = np.inf

    for idx, params in enumerate(spec.iter_param_grid()):
        per_series_metrics: List[Dict[str, Any]] = []

        for dataset in train_datasets:
            serie = dataset['series']
            truth = dataset['changepoints']
            
            try:
                # Apply the detection algorithm with safety wrapper
                detected = safe_algorithm_call(
                    spec.detect_fn, serie, params, spec.key, timeout_seconds=30
                ) if spec.detect_fn else []

                # Calculate metrics
                scores = f1_score_with_tolerance(truth, detected, delta_eval)
                mmd_value = maximum_mean_discrepancy(truth, detected, len(serie))
                mttd_value = mean_time_to_detection(truth, detected, delta_eval)

                per_series_metrics.append({
                    "filename": dataset['filename'],
                    "annotator": dataset['annotator'],
                    "series_id": dataset['series_id'],
                    "f1": scores["f1"],
                    "precision": scores["precision"],
                    "recall": scores["recall"],
                    "tp": scores["TP"],
                    "fp": scores["FP"],
                    "fn": scores["FN"],
                    "mmd": mmd_value,
                    "mttd": mttd_value,
                    "detections": len(detected),
                    "true_changepoints": len(truth),
                    "series_length": len(serie),
                })
                
            except Exception as e:
                print(f"Error processing {dataset['filename']} with {spec.key}: {e}")
                continue

        if not per_series_metrics:
            continue

        # Calculate summary statistics
        summary = {
            "f1_mean": _safe_mean(m["f1"] for m in per_series_metrics),
            "precision_mean": _safe_mean(m["precision"] for m in per_series_metrics),
            "recall_mean": _safe_mean(m["recall"] for m in per_series_metrics),
            "mmd_mean": _safe_mean(m["mmd"] for m in per_series_metrics),
            "mttd_mean": _safe_mean(m["mttd"] for m in per_series_metrics),
            "detections_mean": _safe_mean(m["detections"] for m in per_series_metrics),
            "tp_mean": _safe_mean(m["tp"] for m in per_series_metrics),
            "fp_mean": _safe_mean(m["fp"] for m in per_series_metrics),
            "fn_mean": _safe_mean(m["fn"] for m in per_series_metrics),
            "series_count": len(per_series_metrics),
        }

        records.append({
            "params": params, 
            "summary": summary,
            "per_series": per_series_metrics
        })

        # Update best configuration based on F1 score and MMD
        score = summary["f1_mean"] if summary["f1_mean"] is not None else -np.inf
        mmd_score = summary["mmd_mean"] if summary["mmd_mean"] is not None else np.inf

        if score > best_score or (score == best_score and mmd_score < best_mmd):
            best_score = score
            best_mmd = mmd_score
            best_index = idx

    return {
        "records": records,
        "best_index": best_index,
    }


def evaluate_best_params_on_test_data(
    spec: AlgorithmSpec,
    test_datasets: List[Dict[str, Any]],
    best_params: Dict[str, Any],
    delta_eval: int,
) -> Dict[str, Any]:
    """Evaluate best parameters on test data."""
    per_series_metrics: List[Dict[str, Any]] = []

    for dataset in test_datasets:
        serie = dataset['series']
        truth = dataset['changepoints']
        
        try:
            # Apply the detection algorithm with best params and safety wrapper
            detected = safe_algorithm_call(
                spec.detect_fn, serie, best_params, spec.key, timeout_seconds=30
            ) if spec.detect_fn else []

            # Calculate metrics
            scores = f1_score_with_tolerance(truth, detected, delta_eval)
            mmd_value = maximum_mean_discrepancy(truth, detected, len(serie))
            mttd_value = mean_time_to_detection(truth, detected, delta_eval)

            per_series_metrics.append({
                "filename": dataset['filename'],
                "annotator": dataset['annotator'],
                "series_id": dataset['series_id'],
                "f1": scores["f1"],
                "precision": scores["precision"],
                "recall": scores["recall"],
                "tp": scores["TP"],
                "fp": scores["FP"],
                "fn": scores["FN"],
                "mmd": mmd_value,
                "mttd": mttd_value,
                "detections": len(detected),
                "true_changepoints": len(truth),
                "series_length": len(serie),
            })
            
        except Exception as e:
            print(f"Error processing {dataset['filename']} with {spec.key}: {e}")
            continue

    if not per_series_metrics:
        return {"summary": None, "per_series": []}

    # Calculate summary statistics
    summary = {
        "f1_mean": _safe_mean(m["f1"] for m in per_series_metrics),
        "precision_mean": _safe_mean(m["precision"] for m in per_series_metrics),
        "recall_mean": _safe_mean(m["recall"] for m in per_series_metrics),
        "mmd_mean": _safe_mean(m["mmd"] for m in per_series_metrics),
        "mttd_mean": _safe_mean(m["mttd"] for m in per_series_metrics),
        "detections_mean": _safe_mean(m["detections"] for m in per_series_metrics),
        "tp_mean": _safe_mean(m["tp"] for m in per_series_metrics),
        "fp_mean": _safe_mean(m["fp"] for m in per_series_metrics),
        "fn_mean": _safe_mean(m["fn"] for m in per_series_metrics),
        "series_count": len(per_series_metrics),
    }

    return {
        "summary": summary,
        "per_series": per_series_metrics
    }


def main() -> None:
    """Main execution function."""
    config = CONFIG
    specs = build_algorithm_specs(config)
    
    # Load real data
    data_path = os.path.join(os.path.dirname(__file__), config["data_real_path"])
    datasets = load_real_data(data_path)
    
    if not datasets:
        print("No real data found. Exiting.")
        return

    # Split data into train/test (80/20)
    split_data = train_test_split_real_data(datasets, test_size=0.5, seed=config["seed"])
    train_datasets = split_data['train']
    test_datasets = split_data['test']

    results: List[Dict[str, Any]] = []
    best_series_data: List[Dict[str, Any]] = []  # Array para guardar series is_best

    # Group datasets by annotator for separate analysis
    train_by_annotator = {}
    test_by_annotator = {}
    for dataset in train_datasets:
        annotator = dataset['annotator']
        if annotator not in train_by_annotator:
            train_by_annotator[annotator] = []
        train_by_annotator[annotator].append(dataset)
    
    for dataset in test_datasets:
        annotator = dataset['annotator']
        if annotator not in test_by_annotator:
            test_by_annotator[annotator] = []
        test_by_annotator[annotator].append(dataset)

    print(f"Processing {len(specs)} algorithms on real data...")
    print(f"Train set: {len(train_datasets)} series, Test set: {len(test_datasets)} series")
    
    for i, spec in enumerate(specs, 1):
        print(f"\n[{i}/{len(specs)}] Evaluating algorithm: {spec.key}")
        print(f"  Library: {spec.library}, Method: {spec.method}")
        print(f"  Parameter combinations: {len(list(spec.iter_param_grid()))}")
        
        # Base row information
        base_row = {
            "annotator": "all",
            "algorithm_key": spec.key,
            "algorithm_library": spec.library,
            "algorithm_method": spec.method,
            "supervision": spec.supervision,
            "is_univariate": spec.is_univariate,
            "delta_eval": config["delta_eval"],
            "train_series": len(train_datasets),
            "test_series": len(test_datasets),
            "notes": spec.notes,
        }

        if not spec.implemented:
            results.append(
                base_row
                | {
                    "status": "not_implemented",
                    "data_split": "train_test",
                    "best_params_json": "",
                    "train_f1_mean": None,
                    "test_f1_mean": None,
                    "test_precision_mean": None,
                    "test_recall_mean": None,
                    "test_mmd_mean": None,
                    "test_mttd_mean": None,
                    "test_detections_mean": None,
                    "test_tp_mean": None,
                    "test_fp_mean": None,
                    "test_fn_mean": None,
                }
            )
            continue

        # 1. Grid search on TRAIN data to find best parameters
        train_evaluation = find_best_params_on_train_data(spec, train_datasets, config["delta_eval"])
        train_records = train_evaluation["records"]
        best_index = train_evaluation["best_index"]
        
        if best_index is None or not train_records:
            print(f"No valid results for {spec.key}, skipping...")
            continue
            
        # Get best parameters from training
        best_train_record = train_records[best_index]
        best_params = best_train_record["params"]
        train_summary = best_train_record["summary"]
        
        # 2. Evaluate best parameters on TEST data
        test_evaluation = evaluate_best_params_on_test_data(spec, test_datasets, best_params, config["delta_eval"])
        test_summary = test_evaluation["summary"]
        
        if test_summary is None:
            print(f"No valid test results for {spec.key}, skipping...")
            continue

        # Save result with train/test methodology
        results.append(
            base_row
            | {
                "status": "ok",
                "data_split": "train_test",
                "best_params_json": json.dumps(best_params, sort_keys=True, default=float),
                "train_f1_mean": train_summary.get("f1_mean"),
                "test_f1_mean": test_summary.get("f1_mean"),
                "test_precision_mean": test_summary.get("precision_mean"),
                "test_recall_mean": test_summary.get("recall_mean"),
                "test_mmd_mean": test_summary.get("mmd_mean"),
                "test_mttd_mean": test_summary.get("mttd_mean"),
                "test_detections_mean": test_summary.get("detections_mean"),
                "test_tp_mean": test_summary.get("tp_mean"),
                "test_fp_mean": test_summary.get("fp_mean"),
                "test_fn_mean": test_summary.get("fn_mean"),
                "test_series_count": test_summary.get("series_count"),
            }
        )
        
        # Guardar datos de series is_best para análisis estadístico (solo mejores configuraciones)
        best_series_entry = {
            "combo_key": f"real_data_all_{spec.key}",
            "annotator": "all",
            "algorithm_key": spec.key,
            "algorithm_library": spec.library,
            "algorithm_method": spec.method,
            "best_params": best_params,
            "best_params_json": json.dumps(best_params, sort_keys=True, default=float),
            "train_performance": {
                "f1_mean": train_summary.get("f1_mean"),
                "precision_mean": train_summary.get("precision_mean"),
                "recall_mean": train_summary.get("recall_mean"),
                "mmd_mean": train_summary.get("mmd_mean"),
                "mttd_mean": train_summary.get("mttd_mean"),
                "series_count": train_summary.get("series_count"),
            },
            "test_performance": {
                "f1_mean": test_summary.get("f1_mean"),
                "precision_mean": test_summary.get("precision_mean"),
                "recall_mean": test_summary.get("recall_mean"),
                "mmd_mean": test_summary.get("mmd_mean"),
                "mttd_mean": test_summary.get("mttd_mean"),
                "detections_mean": test_summary.get("detections_mean"),
                "tp_mean": test_summary.get("tp_mean"),
                "fp_mean": test_summary.get("fp_mean"),
                "fn_mean": test_summary.get("fn_mean"),
                "series_count": test_summary.get("series_count"),
            },
            "dataset_info": {
                "train_series": len(train_datasets),
                "test_series": len(test_datasets),
                "total_train_length": sum(d['length'] for d in train_datasets),
                "total_test_length": sum(d['length'] for d in test_datasets),
                "total_train_changepoints": sum(d['n_changepoints'] for d in train_datasets),
                "total_test_changepoints": sum(d['n_changepoints'] for d in test_datasets),
            },
            "detailed_test_results": test_evaluation.get("per_series", []),
            "timestamp": pd.Timestamp.now().isoformat(),
        }
        best_series_data.append(best_series_entry)

        # TODO: Agregar evaluación por anotador si es necesario
        # (requiere suficientes muestras en train/test para cada anotador)


    # Save results
    df = pd.DataFrame(results)
    output_path = os.path.join(os.path.dirname(__file__), config["results_csv"])
    df.to_csv(output_path, index=False)

    # Guardar array de series is_best para análisis estadístico
    best_series_json_path = os.path.join(os.path.dirname(__file__), "best_series_analysis_real_data.json")
    with open(best_series_json_path, 'w', encoding='utf-8') as f:
        json.dump(best_series_data, f, ensure_ascii=False, indent=2, default=str)

    print(f"Main 3 completado. Resultados guardados en: {output_path}")
    print(f"Datos de series is_best guardados para análisis estadístico en: {best_series_json_path}")
    print(f"Total de configuraciones is_best encontradas: {len(best_series_data)}")

    # Show top results (based on test performance)
    implemented_df = df[df["status"] == "ok"]
    if not implemented_df.empty:
        implemented_df = implemented_df.sort_values(by="test_f1_mean", ascending=False)
        summary_cols = [
            "annotator",
            "algorithm_key",
            "train_f1_mean",
            "test_f1_mean",
            "test_precision_mean", 
            "test_recall_mean",
            "test_mmd_mean",
            "best_params_json",
        ]
        print("\nTop resultados por F1 en TEST (metodología train/test):")
        print(implemented_df[summary_cols].head(15).to_string(index=False))
        
        # Also show train vs test performance comparison
        print("\n=== Comparación Train vs Test Performance ===")
        for _, row in implemented_df.head(10).iterrows():
            train_f1 = row.get('train_f1_mean', 0) or 0
            test_f1 = row.get('test_f1_mean', 0) or 0
            diff = train_f1 - test_f1
            print(f"{row['algorithm_key']:25} | Train F1: {train_f1:.3f} | Test F1: {test_f1:.3f} | Diff: {diff:+.3f}")
    else:
        print("No se obtuvieron resultados para los algoritmos evaluados.")


if __name__ == "__main__":
    main()