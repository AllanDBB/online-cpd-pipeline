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
    dataset: Dict[str, Any],
    delta_eval: int,
) -> Dict[str, Any]:
    records: List[Dict[str, Any]] = []
    best_index: int | None = None
    best_score = -np.inf
    best_mmd = np.inf

    for idx, params in enumerate(spec.iter_param_grid()):
        per_series_metrics: List[Dict[str, Any]] = []

        for serie, truth in zip(dataset["series"], dataset["changepoints"]):
            serie_arr = np.asarray(serie, dtype=float)
            effective_length = max(_series_effective_length(serie_arr), 1)
            detected = spec.detect_fn(serie_arr, **params) if spec.detect_fn else []

            scores = f1_score_with_tolerance(truth, detected, delta_eval)
            mmd_value = maximum_mean_discrepancy(truth, detected, effective_length)
            mttd_value = mean_time_to_detection(truth, detected, delta_eval)

            per_series_metrics.append(
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

        records.append({"params": params, "summary": summary})

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
                "iterations": config["n_iterations"],
                "series_per_iteration": config["series_per_combo"],
                "delta_eval": config["delta_eval"],
                "series_evaluated": len(data["series"]),
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
                        "f1_mean": None,
                        "precision_mean": None,
                        "recall_mean": None,
                        "mmd_mean": None,
                        "mttd_mean": None,
                        "detections_mean": None,
                        "tp_mean": None,
                        "fp_mean": None,
                        "fn_mean": None,
                    }
                )
                continue

            evaluation = evaluate_algorithm_on_dataset(spec, data, config["delta_eval"])
            records = evaluation["records"]
            best_index = evaluation["best_index"]

            for trial_id, record in enumerate(records):
                summary = record["summary"] or {}
                params = record["params"] or {}
                results.append(
                    base_row
                    | {
                        "status": "ok",
                        "trial_id": trial_id,
                        "is_best": trial_id == best_index,
                        "params_json": json.dumps(params, sort_keys=True, default=float),
                        "f1_mean": summary.get("f1_mean"),
                        "precision_mean": summary.get("precision_mean"),
                        "recall_mean": summary.get("recall_mean"),
                        "mmd_mean": summary.get("mmd_mean"),
                        "mttd_mean": summary.get("mttd_mean"),
                        "detections_mean": summary.get("detections_mean"),
                        "tp_mean": summary.get("tp_mean"),
                        "fp_mean": summary.get("fp_mean"),
                        "fn_mean": summary.get("fn_mean"),
                        "series_count": summary.get("series_count"),
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

    print(f"Benchmark on synthetic data completed. Results saved to: {output_path}")

    implemented_df = df[(df["status"] == "ok") & (df["is_best"])]
    if not implemented_df.empty:
        implemented_df = implemented_df.sort_values(by="f1_mean", ascending=False)
        summary_cols = [
            "nivel_ruido",
            "fuerza_cambio",
            "tipo_cambio",
            "algorithm_key",
            "f1_mean",
            "mmd_mean",
            "params_json",
        ]
        print("Top resultados por F1 (mejor configuracion por combinacion):")
        print(implemented_df[summary_cols].head(10).to_string(index=False))
    else:
        print("No se obtuvieron resultados para los algoritmos evaluados.")


if __name__ == "__main__":
    main()
