"""
Benchmark pipeline for evaluating change-point algorithms on TCPD-paper datasets.

This script evaluates all implemented CPD algorithms on the Time Series Change Point 
Database (TCPD) datasets. Since these datasets do not have ground truth annotations,
the evaluation focuses on:
- Number of changepoints detected
- Detection consistency across parameter settings
- Algorithm runtime performance
- Visual inspection outputs

Author: Adapted for TCPD-paper benchmark
Date: October 2025
"""

from __future__ import annotations

import itertools
import json
import os
import glob
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, List, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from src.algorithms.page_hinkley import detect_changepoints_page_hinkley
from src.algorithms.adwin import detect_changepoints_adwin
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
    "tcpd_data_path": "data/TCDP-paper",
    "results_csv": "resultados_tcpd_benchmark.csv",
    "timeout_seconds": 120,  # Timeout per algorithm per dataset
    "algorithm_configs": {
        "page_hinkley_river": {
            "grid": {
                "threshold": [20, 50],
                "min_instances": [10],
                "delta": [0.005],
            }
        },
        "adwin_river": {
            "grid": {
                "delta": [0.002, 0.01],
                "clock": [32],
                "max_buckets": [5],
                "min_window_length": [10],
                "grace_period": [20],
            }
        },
        "ewma_numpy": {
            "grid": {
                "alpha": [0.1, 0.2],
                "threshold": [2.5],
                "min_instances": [10],
            }
        },
        "changepoint_online_focus": {
            "grid": {
                "penalty": [10.0, 20.0],
                "min_size": [15],
                "jump": [3],
            }
        },
        "changepoint_online_gaussian": {
            "grid": {
                "penalty": [10.0, 20.0],
                "min_size": [20],
                "jump": [3],
            }
        },
        "changepoint_online_np_focus": {
            "grid": {
                "penalty": [20.0],
                "width": [30],
                "jump": [3],
            }
        },
        "changepoint_online_md_focus": {
            "grid": {
                "penalty": [20.0],
                "min_size": [20],
                "jump": [3],
            }
        },
        "ocpdet_cumsum": {
            "grid": {
                "threshold": [6.0, 10.0],
                "drift": [0.0],
                "min_distance": [20],
            }
        },
        "ocpdet_ewma": {
            "grid": {
                "alpha": [0.1, 0.2],
                "threshold": [2.5],
                "min_instances": [10],
            }
        },
        "ocpdet_two_sample_tests": {
            "grid": {
                "window_size": [30],
                "step": [5],
                "alpha": [0.05],
                "min_distance": [20],
            }
        },
        "ocpdet_neural_networks": {
            "grid": {
                "window_size": [20],
                "step": [1],
                "hidden_layer_sizes": [(20,)],
                "threshold": [2.0],
                "min_distance": [30],
            }
        },
        "ssm_canary": {
            "grid": {
                "process_noise": [1e-3],
                "measurement_noise": [0.6],
                "threshold": [2.5],
                "min_distance": [25],
            }
        },
        "tagi_lstm_ssm": {
            "grid": {
                "process_noise": [2e-4],
                "measurement_noise": [0.3],
                "threshold": [2.5],
                "min_distance": [25],
                "adaptation": [5e-5],
            }
        },
        "skf_kalman_canary": {
            "grid": {
                "process_noise": [5e-3],
                "measurement_noise": [0.7],
                "threshold": [3.0],
                "min_distance": [30],
            }
        },
        "bayesian_online_cpd_cpfinder": {
            "grid": {
                "hazard_lambda": [150.0],
                "alpha": [0.2],
                "beta": [0.01],
                "kappa": [1.0],
                "probability_threshold": [0.6],
                "min_distance": [25],
            }
        },
        "changefinder_sdar": {
            "grid": {
                "r": [0.3],
                "order": [1],
                "smooth": [7],
                "threshold": [2.5],
                "min_distance": [25],
            }
        },
        "rulsif_roerich": {
            "grid": {
                "window_size": [10],
                "lag_size": [60],
                "step": [2],
                "n_epochs": [1],
                "threshold": [0.10],
                "min_distance": [30],
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
        "key": "adwin_river",
        "library": "river",
        "method": "ADWIN",
        "detect_fn": detect_changepoints_adwin,
        "implemented": True,
        "supervision": "no_supervisado",
        "is_univariate": True,
        "notes": "ADWIN (Adaptive Windowing) de river.",
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
        "implemented": True,
        "supervision": "no_supervisado",
        "is_univariate": True,
        "notes": "ChangeFinder SDAR.",
    },
    {
        "key": "rulsif_roerich",
        "library": "RuLSIF",
        "method": "Roerich",
        "detect_fn": detect_changepoints_rulsif,
        "implemented": True,
        "supervision": "no_supervisado",
        "is_univariate": False,
        "notes": "Roerich Online NN RuLSIF.",
    },
]


def build_algorithm_specs(config: Dict[str, Any]) -> List[AlgorithmSpec]:
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


def load_tcpd_datasets(data_path: str) -> List[Dict[str, Any]]:
    """Load all TCPD JSON datasets from the specified path."""
    datasets = []
    json_files = glob.glob(os.path.join(data_path, "**", "*.json"), recursive=True)
    
    print(f"Found {len(json_files)} JSON files in {data_path}")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Extract the time series data
            if "series" in data and len(data["series"]) > 0:
                # Get the first series (assuming univariate for now)
                series_data = data["series"][0]
                if "raw" in series_data:
                    series = np.array(series_data["raw"], dtype=float)
                else:
                    continue
                
                datasets.append({
                    "name": data.get("name", Path(json_file).stem),
                    "longname": data.get("longname", ""),
                    "n_obs": data.get("n_obs", len(series)),
                    "n_dim": data.get("n_dim", 1),
                    "series": series,
                    "file_path": json_file
                })
                
        except Exception as e:
            print(f"Warning: Could not load {json_file}: {e}")
            continue
    
    print(f"Successfully loaded {len(datasets)} datasets")
    return datasets


def safe_algorithm_call(detect_fn: Callable, series: np.ndarray, params: Dict[str, Any], 
                       algorithm_key: str, timeout_seconds: int = 120) -> tuple[List[int], float, str]:
    """
    Safely call detection algorithm with error handling.
    
    Returns:
        tuple: (detected_changepoints, runtime_seconds, status_message)
    """
    import time
    
    try:
        # Suppress warnings for problematic algorithms
        with warnings.catch_warnings():
            if algorithm_key in ['changefinder_sdar', 'rulsif_roerich']:
                warnings.simplefilter("ignore")
            
            start_time = time.time()
            detected = detect_fn(series, **params)
            runtime = time.time() - start_time
            
            if detected is None:
                return [], runtime, "returned_none"
            
            return detected, runtime, "success"
            
    except Exception as e:
        error_msg = str(e)[:100]
        print(f"  WARNING: {algorithm_key} failed: {error_msg}")
        return [], 0.0, f"error: {error_msg}"


def main() -> None:
    config = CONFIG
    specs = build_algorithm_specs(config)
    datasets = load_tcpd_datasets(config["tcpd_data_path"])
    
    if not datasets:
        print("ERROR: No datasets loaded. Please check the data path.")
        return
    
    results: List[Dict[str, Any]] = []
    
    print(f"\nStarting TCPD benchmark evaluation...")
    print(f"Datasets: {len(datasets)}")
    print(f"Algorithms: {len([s for s in specs if s.implemented])}")
    print("=" * 60)
    
    for idx, dataset in enumerate(datasets, 1):
        dataset_name = dataset["name"]
        series = dataset["series"]
        series_length = len(series)
        
        print(f"\n[{idx}/{len(datasets)}] Processing: {dataset_name} (n={series_length})")
        
        for spec in specs:
            if not spec.implemented:
                continue
            
            print(f"  Algorithm: {spec.key}")
            
            for trial_id, params in enumerate(spec.iter_param_grid()):
                detected, runtime, status = safe_algorithm_call(
                    spec.detect_fn,
                    series,
                    params,
                    spec.key,
                    config["timeout_seconds"]
                )
                
                results.append({
                    "dataset_name": dataset_name,
                    "dataset_longname": dataset["longname"],
                    "series_length": series_length,
                    "n_dimensions": dataset["n_dim"],
                    "algorithm_key": spec.key,
                    "algorithm_library": spec.library,
                    "algorithm_method": spec.method,
                    "supervision": spec.supervision,
                    "is_univariate": spec.is_univariate,
                    "trial_id": trial_id,
                    "params_json": json.dumps(params, sort_keys=True, default=float),
                    "n_changepoints_detected": len(detected),
                    "changepoints_detected": json.dumps(detected) if detected else "[]",
                    "runtime_seconds": runtime,
                    "status": status,
                    "notes": spec.notes,
                })
    
    # Save results with timestamp
    df = pd.DataFrame(results)
    
    # Generate timestamped filename: month-day-year-filename
    timestamp = datetime.now().strftime("%m-%d-%Y")
    base_name = os.path.splitext(config["results_csv"])[0]
    extension = os.path.splitext(config["results_csv"])[1]
    timestamped_filename = f"{timestamp}-{base_name}{extension}"
    output_path = os.path.join(os.path.dirname(__file__), timestamped_filename)
    
    df.to_csv(output_path, index=False)
    
    print("\n" + "=" * 60)
    print(f"TCPD Benchmark completed. Results saved to: {output_path}")
    print(f"Total evaluations: {len(results)}")
    print(f"Successful runs: {len(df[df['status'] == 'success'])}")
    print(f"Failed runs: {len(df[df['status'] != 'success'])}")
    
    # Summary statistics
    if not df.empty:
        print("\nSummary by algorithm:")
        summary = df[df['status'] == 'success'].groupby('algorithm_key').agg({
            'n_changepoints_detected': ['mean', 'std'],
            'runtime_seconds': ['mean', 'std']
        }).round(3)
        print(summary.to_string())


if __name__ == "__main__":
    main()
