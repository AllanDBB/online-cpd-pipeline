"""Benchmark pipeline for comparing change-point algorithms on synthetic data."""
from __future__ import annotations

import itertools
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

from src.synthetic_generator import generar_serie_sintetica
from src.f1_score import f1_score_with_tolerance
from src.mttd import mean_time_to_detection
from src.mmd import maximum_mean_discrepancy
from src.split_train_test import split_train_test_synthetic
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
    "n_iterations": 3,
    "series_per_combo": 15,
    "series_length_choices": [200, 300, 400],
    "n_changes_range": (1, 4),
    "noise_levels": {
        "alto": (3.0, 6.0),
        "bajo": (0.0, 0.4),
    },
    "change_strengths": {
        "alto": (3.0, 6.0),
        "bajo": (0.5, 1.5),
    },
    "change_types": ["escalon", "pendiente"],
    "delta_eval": 10,
    "results_csv": "resultados_algoritmos_main2.csv",
    "algorithm_configs": {
        "page_hinkley_river": {
            "grid": {
                "threshold": [20, 40],
                "min_instances": [5, 20],
                "delta": [0.001, 0.005],
            }
        },
        "adwin_river": {
            "grid": {
                "delta": [0.002, 0.005, 0.01],
                "clock": [32, 64],
                "max_buckets": [5],
                "min_window_length": [5, 10],
                "grace_period": [10, 20],
            }
        },
        "ewma_numpy": {
            "grid": {
                "alpha": [0.05, 0.1, 0.2],
                "threshold": [2.0, 2.5],
                "min_instances": [5, 10],
            }
        },
        "changepoint_online_focus": {
            "grid": {
                "penalty": [20.0, 30.0],
                "min_size": [15, 25],
                "jump": [3],
            }
        },
        "changepoint_online_gaussian": {
            "grid": {
                "penalty": [20.0, 30.0],
                "min_size": [20],
                "jump": [3, 5],
            }
        },
        "changepoint_online_np_focus": {
            "grid": {
                "penalty": [20.0, 30.0],
                "width": [30, 50],
                "jump": [3],
            }
        },
        "changepoint_online_md_focus": {
            "grid": {
                "penalty": [20.0, 30.0],
                "min_size": [20, 30],
                "jump": [3],
            }
        },
        "ocpdet_cumsum": {
            "grid": {
                "threshold": [6.0, 10.0],
                "drift": [0.0],
                "min_distance": [20, 30],
            }
        },
        "ocpdet_ewma": {
            "grid": {
                "alpha": [0.05, 0.1, 0.2],
                "threshold": [2.0, 3.0],
                "min_instances": [5],
            }
        },
        "ocpdet_two_sample_tests": {
            "grid": {
                "window_size": [30, 40],
                "step": [5],
                "alpha": [0.01, 0.05],
                "min_distance": [20, 30],
            }
        },
        "ocpdet_neural_networks": {
            "grid": {
                "window_size": [20],
                "step": [1],
                "hidden_layer_sizes": [(20,), (30, 15)],
                "threshold": [2.0, 2.5],
                "min_distance": [30],
            }
        },
        "ssm_canary": {
            "grid": {
                "process_noise": [1e-3, 5e-3],
                "measurement_noise": [0.4, 0.8],
                "threshold": [2.5, 3.0],
                "min_distance": [25],
            }
        },
        "tagi_lstm_ssm": {
            "grid": {
                "process_noise": [2e-4, 5e-4],
                "measurement_noise": [0.2, 0.4],
                "threshold": [2.5, 3.0],
                "min_distance": [25],
                "adaptation": [5e-5],
            }
        },
        "skf_kalman_canary": {
            "grid": {
                "process_noise": [5e-3, 1e-2],
                "measurement_noise": [0.6, 0.9],
                "threshold": [3.0, 3.5],
                "min_distance": [30],
            }
        },
        "bayesian_online_cpd_cpfinder": {
            "grid": {
                "hazard_lambda": [150.0, 300.0],
                "alpha": [0.1, 0.3],
                "beta": [0.01],
                "kappa": [1.0],
                "probability_threshold": [0.5, 0.7],
                "min_distance": [25],
            }
        },
        "changefinder_sdar": {
            "grid": {
                "r": [0.3, 0.5],
                "order": [1],
                "smooth": [5, 9],
                "threshold": [2.5],
                "min_distance": [25],
            }
        },
        "rulsif_roerich": {
            "grid": {
                "window_size": [5, 10],
                "lag_size": [60],
                "step": [2],
                "n_epochs": [1],
                "threshold": [0.08, 0.12],
                "min_distance": [30],
                "alpha": [0.05, 0.1],
            }
        },
    },
}


ALGORITHM_TEMPLATES: List[Dict[str, Any]] = [
    {
        "key": "changepoint_online_focus",
        "library": "changepoint_online",
        "method": "Focus",
        "detect_fn": detect_changepoints_focus,
        "implemented": True,
        "supervision": "no_supervisado",
        "is_univariate": True,
        "notes": "Ruptures PELT con costo RBF.",
        "default_grid": {
            "penalty": [20.0, 30.0],
            "min_size": [15, 25],
            "jump": [3],
        },
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
        "default_grid": {
            "penalty": [20.0, 30.0],
            "min_size": [20],
            "jump": [3, 5],
        },
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
        "default_grid": {
            "penalty": [20.0, 30.0],
            "width": [30, 50],
            "jump": [3],
        },
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
        "default_grid": {
            "penalty": [20.0, 30.0],
            "min_size": [20, 30],
            "jump": [3],
        },
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
        "default_grid": {
            "threshold": [6.0, 10.0],
            "drift": [0.0],
            "min_distance": [20, 30],
        },
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
        "default_grid": {
            "alpha": [0.05, 0.1, 0.2],
            "threshold": [2.0, 3.0],
            "min_instances": [5],
        },
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
        "default_grid": {
            "window_size": [30, 40],
            "step": [5],
            "alpha": [0.01, 0.05],
            "min_distance": [20, 30],
        },
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
        "default_grid": {
            "window_size": [20],
            "step": [1],
            "hidden_layer_sizes": [(20,), (30, 15)],
            "threshold": [2.0, 2.5],
            "min_distance": [30],
        },
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
        "default_grid": {
            "process_noise": [1e-3, 5e-3],
            "measurement_noise": [0.4, 0.8],
            "threshold": [2.5, 3.0],
            "min_distance": [25],
        },
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
        "default_grid": {
            "process_noise": [2e-4, 5e-4],
            "measurement_noise": [0.2, 0.4],
            "threshold": [2.5, 3.0],
            "min_distance": [25],
            "adaptation": [5e-5],
        },
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
        "default_grid": {
            "process_noise": [5e-3, 1e-2],
            "measurement_noise": [0.6, 0.9],
            "threshold": [3.0, 3.5],
            "min_distance": [30],
        },
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
        "default_grid": {
            "hazard_lambda": [150.0, 300.0],
            "alpha": [0.1, 0.3],
            "beta": [0.01],
            "kappa": [1.0],
            "probability_threshold": [0.5, 0.7],
            "min_distance": [25],
        },
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
        "default_grid": {
            "r": [0.3, 0.5],
            "order": [1],
            "smooth": [5, 9],
            "threshold": [2.5],
            "min_distance": [25],
        },
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
        "default_grid": {
            "window_size": [5, 10],
            "lag_size": [60],
            "step": [2],
            "n_epochs": [1],
            "threshold": [0.08, 0.12],
            "min_distance": [30],
            "alpha": [0.05, 0.1],
        },
    },
    {
        "key": "page_hinkley_river",
        "library": "river",
        "method": "PageHinkley",
        "detect_fn": detect_changepoints_page_hinkley,
        "implemented": True,
        "supervision": "no_supervisado",
        "is_univariate": True,
        "notes": "Page-Hinkley de river.",
        "default_grid": {
            "threshold": [20, 40],
            "min_instances": [5, 20],
            "delta": [0.001, 0.005],
        },
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
        "default_grid": {
            "delta": [0.002, 0.005, 0.01],
            "clock": [32, 64],
            "max_buckets": [5],
            "min_window_length": [5, 10],
            "grace_period": [10, 20],
        },
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
        "default_grid": {
            "alpha": [0.05, 0.1, 0.2],
            "threshold": [2.0, 2.5],
            "min_instances": [5, 10],
        },
    },
]


# --- Helper utilities for dataset generation and evaluation ---


def build_algorithm_specs(config: Dict[str, Any]) -> List[AlgorithmSpec]:
    overrides = config.get("algorithm_configs", {})
    specs: List[AlgorithmSpec] = []
    for template in ALGORITHM_TEMPLATES:
        key = template["key"]
        override = overrides.get(key, {})
        param_grid = override.get("grid", template.get("default_grid", {}))
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
    filtered = [float(v) for v in values if v is not None]
    if not filtered:
        return None
    return float(np.mean(filtered))


def _series_effective_length(series: np.ndarray) -> int:
    arr = np.asarray(series, dtype=float)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    valid = ~np.isnan(arr)
    return int(np.count_nonzero(valid))


def generate_batch(
    noise_key: str,
    strength_key: str,
    change_type: str,
    config: Dict[str, Any],
    rng: np.random.Generator,
) -> Dict[str, Any]:
    noise_range = config["noise_levels"][noise_key]
    strength_range = config["change_strengths"][strength_key]
    length_choices = config["series_length_choices"]
    change_min, change_max = config["n_changes_range"]
    series_per_combo = config["series_per_combo"]

    series_list: List[np.ndarray] = []
    changepoints_list: List[List[int]] = []
    lengths: List[int] = []

    for _ in range(series_per_combo):
        length = int(rng.choice(length_choices))
        n_changes = int(rng.integers(change_min, change_max + 1))
        noise_level = float(rng.uniform(*noise_range))
        strength = float(rng.uniform(*strength_range))

        serie_seed = None
        if config.get("seed") is not None:
            serie_seed = int(rng.integers(0, 2 ** 31 - 1))

        serie, changepoints = generar_serie_sintetica(
            longitud=length,
            nivel_ruido=noise_level,
            num_cambios=n_changes,
            fuerza_cambio=strength,
            tipo_cambio=change_type,
            seed=serie_seed,
        )
        series_list.append(serie.astype(float))
        changepoints_list.append([int(c) for c in changepoints])
        lengths.append(length)

    return {
        "series": series_list,
        "changepoints": changepoints_list,
        "lengths": lengths,
    }


def build_datasets(config: Dict[str, Any]) -> Dict[tuple[str, str, str], Dict[str, Any]]:
    combos = list(
        itertools.product(
            sorted(config["noise_levels"].keys()),
            sorted(config["change_strengths"].keys()),
            config["change_types"],
        )
    )
    rng = np.random.default_rng(config.get("seed"))
    datasets: Dict[tuple[str, str, str], Dict[str, Any]] = {}

    for noise_key, strength_key, change_type in combos:
        aggregated_series: List[np.ndarray] = []
        aggregated_cps: List[List[int]] = []
        aggregated_lengths: List[int] = []
        for _ in range(config["n_iterations"]):
            batch = generate_batch(noise_key, strength_key, change_type, config, rng)
            aggregated_series.extend(batch["series"])
            aggregated_cps.extend(batch["changepoints"])
            aggregated_lengths.extend(batch["lengths"])
        datasets[(noise_key, strength_key, change_type)] = {
            "series": aggregated_series,
            "changepoints": aggregated_cps,
            "lengths": aggregated_lengths,
        }
    return datasets


def evaluate_algorithm_on_dataset(
    spec: AlgorithmSpec,
    train_data: Dict[str, Any],
    test_data: Dict[str, Any],
    delta_eval: int,
) -> Dict[str, Any]:
    """
    Evaluate algorithm using train/test methodology:
    1. Find best parameters on train set
    2. Evaluate those parameters on test set
    """
    records: List[Dict[str, Any]] = []
    best_index: int | None = None
    best_train_score = -np.inf
    best_train_mmd = np.inf

    # Phase 1: Grid search on TRAIN data to find best parameters
    for idx, params in enumerate(spec.iter_param_grid()):
        train_metrics: List[Dict[str, Any]] = []
        test_metrics: List[Dict[str, Any]] = []

        # Evaluate on TRAIN set
        for serie, truth in zip(train_data["series"], train_data["changepoints"]):
            serie_arr = np.asarray(serie, dtype=float)
            effective_length = max(_series_effective_length(serie_arr), 1)
            detected = spec.detect_fn(serie_arr, **params) if spec.detect_fn else []

            scores = f1_score_with_tolerance(truth, detected, delta_eval)
            mmd_value = maximum_mean_discrepancy(truth, detected, effective_length)
            mttd_value = mean_time_to_detection(truth, detected, delta_eval)

            train_metrics.append(
                {
                    "f1": scores["f1"],
                    "precision": scores["precision"],
                    "recall": scores["recall"],
                    "tp": scores["TP"],
                    "fp": scores["FP"],
                    "fn": scores["FN"],
                    "mmd": mmd_value,
                    "mttd": mttd_value,
                    "detections": len(detected),
                }
            )
        
        # Evaluate on TEST set with same parameters
        for serie, truth in zip(test_data["series"], test_data["changepoints"]):
            serie_arr = np.asarray(serie, dtype=float)
            effective_length = max(_series_effective_length(serie_arr), 1)
            detected = spec.detect_fn(serie_arr, **params) if spec.detect_fn else []

            scores = f1_score_with_tolerance(truth, detected, delta_eval)
            mmd_value = maximum_mean_discrepancy(truth, detected, effective_length)
            mttd_value = mean_time_to_detection(truth, detected, delta_eval)

            test_metrics.append(
                {
                    "f1": scores["f1"],
                    "precision": scores["precision"],
                    "recall": scores["recall"],
                    "tp": scores["TP"],
                    "fp": scores["FP"],
                    "fn": scores["FN"],
                    "mmd": mmd_value,
                    "mttd": mttd_value,
                    "detections": len(detected),
                }
            )

        train_summary = {
            "f1_mean": _safe_mean(m["f1"] for m in train_metrics),
            "precision_mean": _safe_mean(m["precision"] for m in train_metrics),
            "recall_mean": _safe_mean(m["recall"] for m in train_metrics),
            "mmd_mean": _safe_mean(m["mmd"] for m in train_metrics),
            "mttd_mean": _safe_mean(m["mttd"] for m in train_metrics),
            "detections_mean": _safe_mean(m["detections"] for m in train_metrics),
            "tp_mean": _safe_mean(m["tp"] for m in train_metrics),
            "fp_mean": _safe_mean(m["fp"] for m in train_metrics),
            "fn_mean": _safe_mean(m["fn"] for m in train_metrics),
            "series_count": len(train_metrics),
        }
        
        test_summary = {
            "f1_mean": _safe_mean(m["f1"] for m in test_metrics),
            "precision_mean": _safe_mean(m["precision"] for m in test_metrics),
            "recall_mean": _safe_mean(m["recall"] for m in test_metrics),
            "mmd_mean": _safe_mean(m["mmd"] for m in test_metrics),
            "mttd_mean": _safe_mean(m["mttd"] for m in test_metrics),
            "detections_mean": _safe_mean(m["detections"] for m in test_metrics),
            "tp_mean": _safe_mean(m["tp"] for m in test_metrics),
            "fp_mean": _safe_mean(m["fp"] for m in test_metrics),
            "fn_mean": _safe_mean(m["fn"] for m in test_metrics),
            "series_count": len(test_metrics),
        }

        records.append({
            "params": params, 
            "train_summary": train_summary,
            "test_summary": test_summary
        })

        # Select best parameters based on TRAIN performance
        train_score = train_summary["f1_mean"] if train_summary["f1_mean"] is not None else -np.inf
        train_mmd = train_summary["mmd_mean"] if train_summary["mmd_mean"] is not None else np.inf

        if train_score > best_train_score or (train_score == best_train_score and train_mmd < best_train_mmd):
            best_train_score = train_score
            best_train_mmd = train_mmd
            best_index = idx

    return {
        "records": records,
        "best_index": best_index,
    }


def main() -> None:
    config = CONFIG
    specs = build_algorithm_specs(config)
    datasets = build_datasets(config)

    results: List[Dict[str, Any]] = []
    
    # Crear directorio para guardar las series por separado
    series_dir = os.path.join(os.path.dirname(__file__), "series_data")
    os.makedirs(series_dir, exist_ok=True)
    
    # Diccionario para evitar duplicar series idénticas
    series_cache: Dict[str, str] = {}

    for (noise_key, strength_key, change_type), data in datasets.items():
        lengths = data["lengths"]
        avg_length = _safe_mean(lengths)
        std_length = float(np.std(lengths)) if lengths else None

        # Split data into train/test (70/30)
        split_data = split_train_test_synthetic(data, test_size=0.3, seed=config["seed"])
        train_data = split_data["train"]
        test_data = split_data["test"]
        
        print(f"\nProcessing: noise={noise_key}, strength={strength_key}, type={change_type}")
        print(f"  Train: {len(train_data['series'])} series, Test: {len(test_data['series'])} series")

        # Crear identificador único para esta combinación de datos
        data_id = f"{noise_key}_{strength_key}_{change_type}"
        
        # Guardar las series solo una vez por combinación
        if data_id not in series_cache:
            series_file = f"series_{data_id}.json"
            series_path = os.path.join(series_dir, series_file)
            
            series_data = {
                "data": [serie.tolist() for serie in data["series"]],
                "changepoints": data["changepoints"],
                "lengths": data["lengths"]
            }
            
            with open(series_path, 'w') as f:
                json.dump(series_data, f, indent=2)
            
            series_cache[data_id] = series_file

        for spec in specs:
            base_row = {
                "nivel_ruido": noise_key,
                "fuerza_cambio": strength_key,
                "tipo_cambio": change_type,
                "algorithm_key": spec.key,
                "algorithm_library": spec.library,
                "algorithm_method": spec.method,
                "supervision": spec.supervision,
                "is_univariate": spec.is_univariate,
                "delta_eval": config["delta_eval"],
                "series_total": len(data["series"]),
                "series_train": len(train_data["series"]),
                "series_test": len(test_data["series"]),
                "series_avg_length": avg_length,
                "series_std_length": std_length,
                "notes": spec.notes,
                "series_file": series_cache[data_id],  # Solo referencia al archivo
            }

            if not spec.implemented:
                results.append(
                    base_row
                    | {
                        "status": "not_implemented",
                        "trial_id": None,
                        "is_best": False,
                        "params_json": "",
                        "train_f1_mean": None,
                        "train_precision_mean": None,
                        "train_recall_mean": None,
                        "train_mmd_mean": None,
                        "train_mttd_mean": None,
                        "test_f1_mean": None,
                        "test_precision_mean": None,
                        "test_recall_mean": None,
                        "test_mmd_mean": None,
                        "test_mttd_mean": None,
                    }
                )
                continue

            evaluation = evaluate_algorithm_on_dataset(spec, train_data, test_data, config["delta_eval"])
            records = evaluation["records"]
            best_index = evaluation["best_index"]

            for trial_id, record in enumerate(records):
                train_summary = record.get("train_summary") or {}
                test_summary = record.get("test_summary") or {}
                params = record.get("params") or {}
                is_best = trial_id == best_index
                
                # Para configuraciones is_best, guardar las series train/test usadas
                best_series_file = None
                if is_best:
                    best_series_id = f"{data_id}_{spec.key}_trial{trial_id}"
                    best_series_filename = f"best_series_{best_series_id}.json"
                    best_series_path = os.path.join(series_dir, best_series_filename)
                    
                    best_series_data = {
                        "metadata": {
                            "nivel_ruido": noise_key,
                            "fuerza_cambio": strength_key,
                            "tipo_cambio": change_type,
                            "algorithm_key": spec.key,
                            "algorithm_library": spec.library,
                            "algorithm_method": spec.method,
                            "params": params,
                            "train_f1_mean": train_summary.get("f1_mean"),
                            "test_f1_mean": test_summary.get("f1_mean"),
                        },
                        "train": {
                            "series": [serie.tolist() for serie in train_data["series"]],
                            "changepoints": train_data["changepoints"],
                            "lengths": train_data["lengths"]
                        },
                        "test": {
                            "series": [serie.tolist() for serie in test_data["series"]],
                            "changepoints": test_data["changepoints"],
                            "lengths": test_data["lengths"]
                        }
                    }
                    
                    with open(best_series_path, 'w') as f:
                        json.dump(best_series_data, f, indent=2)
                    
                    best_series_file = best_series_filename
                
                results.append(
                    base_row
                    | {
                        "status": "ok",
                        "trial_id": trial_id,
                        "is_best": is_best,
                        "params_json": json.dumps(params, sort_keys=True, default=float),
                        "train_f1_mean": train_summary.get("f1_mean"),
                        "train_precision_mean": train_summary.get("precision_mean"),
                        "train_recall_mean": train_summary.get("recall_mean"),
                        "train_mmd_mean": train_summary.get("mmd_mean"),
                        "train_mttd_mean": train_summary.get("mttd_mean"),
                        "train_detections_mean": train_summary.get("detections_mean"),
                        "train_tp_mean": train_summary.get("tp_mean"),
                        "train_fp_mean": train_summary.get("fp_mean"),
                        "train_fn_mean": train_summary.get("fn_mean"),
                        "test_f1_mean": test_summary.get("f1_mean"),
                        "test_precision_mean": test_summary.get("precision_mean"),
                        "test_recall_mean": test_summary.get("recall_mean"),
                        "test_mmd_mean": test_summary.get("mmd_mean"),
                        "test_mttd_mean": test_summary.get("mttd_mean"),
                        "test_detections_mean": test_summary.get("detections_mean"),
                        "test_tp_mean": test_summary.get("tp_mean"),
                        "test_fp_mean": test_summary.get("fp_mean"),
                        "test_fn_mean": test_summary.get("fn_mean"),
                        "best_series_file": best_series_file,  # Nueva columna
                    }
                )

    df = pd.DataFrame(results)
    
    # Generate timestamped filename: month-day-year-filename
    timestamp = datetime.now().strftime("%m-%d-%Y")
    base_name = os.path.splitext(config["results_csv"])[0]
    extension = os.path.splitext(config["results_csv"])[1]
    timestamped_filename = f"{timestamp}-{base_name}-synthetic{extension}"
    output_path = os.path.join(os.path.dirname(__file__), timestamped_filename)
    
    df.to_csv(output_path, index=False)

    print(f"\nBenchmark on synthetic data completed. Results saved to: {output_path}")
    print(f"Total evaluations: {len(df)}")
    print(f"Using Train/Test methodology (70/30 split)")
    
    # Count best configurations with saved series
    best_count = len(df[df["is_best"] == True])
    print(f"Best configurations found: {best_count}")
    print(f"Series for best configurations saved in: {series_dir}/")

    implemented_df = df[(df["status"] == "ok") & (df["is_best"])]
    if not implemented_df.empty:
        implemented_df = implemented_df.sort_values(by="test_f1_mean", ascending=False)
        summary_cols = [
            "nivel_ruido",
            "fuerza_cambio",
            "tipo_cambio",
            "algorithm_key",
            "train_f1_mean",
            "test_f1_mean",
            "test_mmd_mean",
            "params_json",
        ]
        print("\nTop results by TEST F1 score (best configuration per combination):")
        print(implemented_df[summary_cols].head(10).to_string(index=False))
        
        # Show train vs test comparison
        print("\n=== Train vs Test Performance Comparison ===")
        for idx, row in implemented_df.head(10).iterrows():
            train_f1 = row.get("train_f1_mean", 0) or 0
            test_f1 = row.get("test_f1_mean", 0) or 0
            diff = train_f1 - test_f1
            print(f"{row['algorithm_key']:30} | Train F1: {train_f1:.3f} | Test F1: {test_f1:.3f} | Diff: {diff:+.3f}")
    else:
        print("No se obtuvieron resultados para los algoritmos evaluados.")


if __name__ == "__main__":
    main()
