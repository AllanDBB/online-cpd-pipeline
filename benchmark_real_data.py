"""Benchmark pipeline for testing change-point algorithms on real labeled data."""
from __future__ import annotations

import itertools
import json
import os
import copy
import glob
import signal
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple
from contextlib import contextmanager

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

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


# ============================================================================
# FUNCIONES DE CLASIFICACIÓN DE SERIES (del notebook generateSynthetic.ipynb)
# ============================================================================

def estimar_ruido(serie, metric="NSR"):
    """
    Estima el nivel de ruido de una serie usando SNR o NSR.
    Ajusta automáticamente el window_length según la longitud de la serie.
    Usa polyorder=2 como valor fijo y genérico.
    
    Parámetros:
    -----------
    serie : array-like
        Serie de tiempo (list o numpy array).
    metric : str
        'SNR' para signal-to-noise ratio
        'NSR' para noise-to-signal ratio
    
    Retorna:
    --------
    valor : float
        Estimación del ruido en escala SNR o NSR.
    """
    serie = np.array(serie)
    n = len(serie)

    # Elegir window_length adaptativo (impar, al menos 5)
    window_length = max(5, n // 20)  # regla: 5% del tamaño de la serie
    if window_length % 2 == 0:  
        window_length += 1  # debe ser impar
    
    polyorder = 2  # valor fijo recomendado
    
    # Señal suavizada
    señal = savgol_filter(serie, window_length=window_length, polyorder=polyorder)
    ruido = serie - señal
    
    # Potencias (varianzas)
    var_signal = np.var(señal)
    var_noise = np.var(ruido)
    
    if metric.upper() == "SNR":
        if var_noise == 0:
            return np.inf
        return var_signal / var_noise
    
    elif metric.upper() == "NSR":
        if var_signal == 0:
            return np.inf
        return var_noise / var_signal
    
    else:
        raise ValueError("metric debe ser 'SNR' o 'NSR'")


def estimar_cambio(serie, puntos_cambio):
    """
    Estima la magnitud de cambio de una serie a partir de puntos de cambio dados.

    Parámetros:
    -----------
    serie : array-like
        Serie de tiempo (list o numpy array).
    puntos_cambio : list of int
        Lista con los índices donde ocurren los cambios.

    Retorna:
    --------
    magnitud_min : float
        La menor magnitud de cambio detectada.
    magnitudes : list
        Lista con las magnitudes de cada cambio.
    """
    serie = np.array(serie)
    puntos = sorted([0] + puntos_cambio + [len(serie)])  # incluir inicio y final
    
    magnitudes = []
    for i in range(1, len(puntos)-1):  # comparamos segmentos consecutivos
        seg_anterior = serie[puntos[i-1]:puntos[i]]
        seg_actual = serie[puntos[i]:puntos[i+1]]
        
        if len(seg_anterior) > 0 and len(seg_actual) > 0:
            magnitud = abs(np.mean(seg_actual) - np.mean(seg_anterior))
            magnitudes.append(magnitud)
    
    magnitud_min = min(magnitudes) if magnitudes else 0
    return magnitud_min, magnitudes


def clasificar_series_reales(datasets: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Clasifica series de criminalidad (datos reales) basándose en ruido y magnitud de cambio.
    
    Retorna categorías descriptivas en lugar de números.
    
    Parámetros:
    -----------
    datasets : list of dict
        Lista de diccionarios con información de series reales.
        Cada dict debe tener: 'series', 'changepoints', 'filename', 'metadata'
    
    Retorna:
    --------
    resultados : pd.DataFrame
        DataFrame con columnas: filename, serie_id, ruido, cambio, categoria_ruido, categoria_cambio, tipo_cambio
    """
    resultados = []
    
    for dataset in datasets:
        serie = dataset['series']
        puntos = dataset['changepoints']
        filename = dataset.get('filename', 'unknown')
        serie_id = dataset.get('series_id', 0)
        metadata = dataset.get('metadata', {})
        
        # Extraer tipos de changepoint del metadata
        changepoint_types = metadata.get('changepoint_types', [])
        # Determinar tipo de cambio predominante (escalón o pendiente)
        tipo_cambio = 'sin_cambio'
        if changepoint_types:
            # Contar tipos
            tipo_cambio_list = [str(ct).lower() for ct in changepoint_types if ct and str(ct).strip()]
            if tipo_cambio_list:
                # Si hay escalón, marcarlo como escalón, sino pendiente
                if any('escalon' in t or 'step' in t or 'abrupt' in t for t in tipo_cambio_list):
                    tipo_cambio = 'escalon'
                elif any('pendiente' in t or 'trend' in t or 'gradual' in t for t in tipo_cambio_list):
                    tipo_cambio = 'pendiente'
                else:
                    # Si no se puede determinar, usar el primero
                    tipo_cambio = tipo_cambio_list[0] if tipo_cambio_list else 'desconocido'
        
        # --- estandarización ---
        mu, sigma = np.mean(serie), np.std(serie)
        serie_std = (serie - mu) / sigma if sigma > 0 else serie - mu
        
        # --- ruido (NSR) ---
        try:
            ruido = estimar_ruido(serie_std, metric="NSR")
        except Exception as e:
            print(f"  WARNING: No se pudo estimar ruido para {filename}: {e}")
            ruido = np.nan
        
        # --- cambios ---
        try:
            cambio_min, magnitudes = estimar_cambio(serie_std, puntos)
        except Exception as e:
            print(f"  WARNING: No se pudo estimar cambio para {filename}: {e}")
            cambio_min = np.nan
        
        resultados.append({
            "filename": filename,
            "serie_id": serie_id,
            "annotator": dataset.get('annotator', 'unknown'),
            "ruido": ruido,
            "cambio": cambio_min,
            "tipo_cambio": tipo_cambio,
            "length": len(serie),
            "n_changepoints": len(puntos),
        })
    
    df_resultados = pd.DataFrame(resultados)
    
    # Clasificar usando umbrales (medianas) solo de valores válidos
    df_valid = df_resultados.dropna(subset=['ruido', 'cambio'])
    
    if len(df_valid) > 0:
        umbral_ruido = df_valid['ruido'].median()
        umbral_cambio = df_valid['cambio'].median()
        
        # Asignar categorías descriptivas
        def asignar_categoria_ruido(row):
            if pd.isna(row['ruido']):
                return None
            return 'alto' if row['ruido'] > umbral_ruido else 'bajo'
        
        def asignar_categoria_cambio(row):
            if pd.isna(row['cambio']):
                return None
            return 'alto' if row['cambio'] > umbral_cambio else 'bajo'
        
        df_resultados['categoria_ruido'] = df_resultados.apply(asignar_categoria_ruido, axis=1)
        df_resultados['categoria_cambio'] = df_resultados.apply(asignar_categoria_cambio, axis=1)
        
        # Agregar info de umbrales usados
        df_resultados.attrs['umbral_ruido'] = umbral_ruido
        df_resultados.attrs['umbral_cambio'] = umbral_cambio
    else:
        df_resultados['categoria_ruido'] = None
        df_resultados['categoria_cambio'] = None
        df_resultados.attrs['umbral_ruido'] = None
        df_resultados.attrs['umbral_cambio'] = None
    
    return df_resultados


# ============================================================================
# FIN DE FUNCIONES DE CLASIFICACIÓN
# ============================================================================


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
    # Labeler configuration
    # Set to True to use only Martin's labels (recommended due to low agreement F1=0.24)
    # Set to False to use all labels (Martin + Allan)
    "use_only_martin_labels": True,
    "algorithm_configs": {
        "page_hinkley_river": {
            "grid": {
                "threshold": [20, 40, 60, 80],
                "min_instances": [5, 10, 20],
                "delta": [0.001, 0.005, 0.01],
            }
        },
        "adwin_river": {
            "grid": {
                "delta": [0.002, 0.005, 0.01, 0.02],
                "clock": [32, 64, 128],
                "max_buckets": [5],
                "min_window_length": [5, 10, 15],
                "grace_period": [10, 20, 30],
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


def load_real_data(data_path: str, use_only_martin: bool = True) -> List[Dict[str, Any]]:
    """
    Load real labeled data from CSV files.
    
    Args:
        data_path: Path to the directory containing CSV files
        use_only_martin: If True, load only Martin's labels. If False, load all labels.
                        Default True due to low inter-annotator agreement (F1=0.24)
    
    Returns:
        List of datasets with series and changepoint labels
    """
    csv_files = glob.glob(os.path.join(data_path, "*.csv"))
    datasets = []
    
    # Track statistics
    martin_count = 0
    allan_count = 0
    skipped_count = 0
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            
            # Extract metadata from filename
            filename = os.path.basename(csv_file)
            if 'allan_' in filename:
                annotator = 'allan'
            elif 'Mart_n_Sol_s_Salazar_' in filename:
                annotator = 'martin'
            else:
                annotator = 'unknown'
            
            # Filter based on configuration
            if use_only_martin and annotator != 'martin':
                skipped_count += 1
                continue
            
            # Extract series data
            series = df['Value'].values.astype(float)
            
            # Extract changepoints (where Is_ChangePoint is TRUE)
            changepoints = df[df['Is_ChangePoint'] == True]['Index'].tolist()
            
            # Extract series ID from filename
            import re
            series_match = re.search(r'-s(\d+)_', filename)
            series_id = int(series_match.group(1)) if series_match else 0
            
            # Track counts
            if annotator == 'martin':
                martin_count += 1
            elif annotator == 'allan':
                allan_count += 1
            
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
    
    # Print summary
    print("="*80)
    print("REAL DATA LOADING SUMMARY")
    print("="*80)
    print(f"Configuration: {'ONLY MARTIN LABELS' if use_only_martin else 'ALL LABELS (Martin + Allan)'}")
    if use_only_martin:
        print(f"⚠️  Using only Martin's labels due to low inter-annotator agreement (F1=0.24)")
    print(f"\nLoaded {len(datasets)} real data series from {data_path}")
    print(f"  - Martin's series: {martin_count}")
    print(f"  - Allan's series: {allan_count}")
    if skipped_count > 0:
        print(f"  - Skipped (filtered out): {skipped_count}")
    print(f"\nAnnotators in final dataset: {set(d['annotator'] for d in datasets)}")
    if datasets:
        print(f"Series lengths range: {min(d['length'] for d in datasets)} - {max(d['length'] for d in datasets)}")
        print(f"Changepoints per series range: {min(d['n_changepoints'] for d in datasets)} - {max(d['n_changepoints'] for d in datasets)}")
    print("="*80)
    print()
    
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
    datasets = load_real_data(data_path, use_only_martin=config.get("use_only_martin_labels", True))
    
    if not datasets:
        print("No real data found. Exiting.")
        return

    # Clasificar todas las series antes del split
    print("\n=== Clasificando series de criminalidad ===")
    clasificacion_df = clasificar_series_reales(datasets)
    
    # Mostrar estadísticas de clasificación
    print(f"Total de series clasificadas: {len(clasificacion_df)}")
    print(f"Series válidas: {clasificacion_df['categoria_ruido'].notna().sum()}")
    
    # Distribución por tipo de cambio
    tipo_cambio_counts = clasificacion_df['tipo_cambio'].value_counts()
    print(f"\n📊 Distribución por tipo de cambio:")
    for tipo_cambio, count in tipo_cambio_counts.items():
        print(f"  {tipo_cambio}: {count} series")
    
    # Distribución por categoría de ruido
    ruido_counts = clasificacion_df['categoria_ruido'].value_counts()
    print(f"\n🔊 Distribución por nivel de ruido:")
    for cat, count in ruido_counts.items():
        if pd.notna(cat):
            print(f"  Ruido {cat}: {count} series")
    
    # Distribución por categoría de cambio
    cambio_counts = clasificacion_df['categoria_cambio'].value_counts()
    print(f"\n📈 Distribución por magnitud de cambio:")
    for cat, count in cambio_counts.items():
        if pd.notna(cat):
            print(f"  Cambio {cat}: {count} series")
    
    # Tabla cruzada de categorías
    if clasificacion_df['categoria_ruido'].notna().any() and clasificacion_df['categoria_cambio'].notna().any():
        print(f"\n📋 Tabla cruzada (Ruido x Cambio):")
        tabla_cruzada = pd.crosstab(
            clasificacion_df['categoria_ruido'], 
            clasificacion_df['categoria_cambio'],
            margins=True
        )
        print(tabla_cruzada)
    
    if clasificacion_df.attrs.get('umbral_ruido') is not None:
        print(f"\n🎯 Umbrales utilizados:")
        print(f"  Ruido (NSR): {clasificacion_df.attrs['umbral_ruido']:.4f}")
        print(f"  Cambio: {clasificacion_df.attrs['umbral_cambio']:.4f}")
    
    print(f"\n📊 Estadísticas por combinación:")
    # Agrupar por combinación de categorías
    if 'categoria_ruido' in clasificacion_df.columns and 'categoria_cambio' in clasificacion_df.columns:
        for (cat_ruido, cat_cambio), group in clasificacion_df.groupby(['categoria_ruido', 'categoria_cambio']):
            if pd.notna(cat_ruido) and pd.notna(cat_cambio):
                print(f"\n  Ruido {cat_ruido} + Cambio {cat_cambio}: {len(group)} series")
                print(f"    Ruido promedio (NSR): {group['ruido'].mean():.4f}")
                print(f"    Cambio promedio: {group['cambio'].mean():.4f}")
                print(f"    Longitud promedio: {group['length'].mean():.1f}")
                print(f"    Changepoints promedio: {group['n_changepoints'].mean():.1f}")
                # Mostrar distribución de tipo_cambio en este grupo
                tipo_cambio_en_grupo = group['tipo_cambio'].value_counts()
                print(f"    Tipos de cambio: {dict(tipo_cambio_en_grupo)}")
    
    # Guardar clasificación en CSV
    timestamp = datetime.now().strftime("%m-%d-%Y")
    clasificacion_filename = f"{timestamp}-clasificacion_series_criminalidad.csv"
    clasificacion_path = os.path.join(os.path.dirname(__file__), clasificacion_filename)
    clasificacion_df.to_csv(clasificacion_path, index=False)
    print(f"\nClasificación guardada en: {clasificacion_filename}")

    # Split data into train/test (70/30)
    split_data = train_test_split_real_data(datasets, test_size=0.3, seed=config["seed"])
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
        
        # Agregar info de clasificación a resultados detallados por serie
        test_results_with_classification = []
        for test_result in test_evaluation.get("per_series", []):
            filename = test_result.get("filename", "")
            # Buscar clasificación de esta serie
            clasificacion_serie = clasificacion_df[clasificacion_df['filename'] == filename]
            if not clasificacion_serie.empty:
                row = clasificacion_serie.iloc[0]
                test_result['clasificacion_ruido'] = float(row['ruido']) if pd.notna(row['ruido']) else None
                test_result['clasificacion_cambio'] = float(row['cambio']) if pd.notna(row['cambio']) else None
                test_result['clasificacion_categoria_ruido'] = str(row['categoria_ruido']) if pd.notna(row['categoria_ruido']) else None
                test_result['clasificacion_categoria_cambio'] = str(row['categoria_cambio']) if pd.notna(row['categoria_cambio']) else None
                test_result['clasificacion_tipo_cambio'] = str(row['tipo_cambio']) if pd.notna(row['tipo_cambio']) else None
            test_results_with_classification.append(test_result)
        
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
            "clasificacion_info": {
                "total_series": len(clasificacion_df),
                "series_clasificadas": int(clasificacion_df['categoria_ruido'].notna().sum()),
                "distribucion_tipo_cambio": clasificacion_df['tipo_cambio'].value_counts().to_dict(),
                "distribucion_categoria_ruido": clasificacion_df['categoria_ruido'].value_counts().to_dict(),
                "distribucion_categoria_cambio": clasificacion_df['categoria_cambio'].value_counts().to_dict(),
                "umbral_ruido": clasificacion_df.attrs.get('umbral_ruido'),
                "umbral_cambio": clasificacion_df.attrs.get('umbral_cambio'),
            },
            "detailed_test_results": test_results_with_classification,
            "timestamp": pd.Timestamp.now().isoformat(),
        }
        best_series_data.append(best_series_entry)

        # TODO: Agregar evaluación por anotador si es necesario
        # (requiere suficientes muestras en train/test para cada anotador)


    # Save results with timestamp
    df = pd.DataFrame(results)
    
    # Generate timestamped filename: month-day-year-filename
    timestamp = datetime.now().strftime("%m-%d-%Y")
    base_name = os.path.splitext(config["results_csv"])[0]
    extension = os.path.splitext(config["results_csv"])[1]
    timestamped_filename = f"{timestamp}-{base_name}-real{extension}"
    output_path = os.path.join(os.path.dirname(__file__), timestamped_filename)
    
    df.to_csv(output_path, index=False)

    # Save best series data for statistical analysis with timestamp
    best_series_json_filename = f"{timestamp}-best_series_analysis_real_data.json"
    best_series_json_path = os.path.join(os.path.dirname(__file__), best_series_json_filename)
    
    # Add configuration metadata to the JSON output
    output_json = {
        "configuration": {
            "use_only_martin_labels": config.get("use_only_martin_labels", True),
            "inter_annotator_agreement_f1": 0.24,  # From agreement analysis
            "note": "Using only Martin's labels due to low inter-annotator agreement" if config.get("use_only_martin_labels", True) else "Using all labels (Martin + Allan)",
            "delta_eval": config.get("delta_eval"),
            "seed": config.get("seed"),
            "profile": config.get("profile"),
            "timestamp": timestamp,
        },
        "results": best_series_data
    }
    
    with open(best_series_json_path, 'w', encoding='utf-8') as f:
        json.dump(output_json, f, ensure_ascii=False, indent=2, default=str)

    print(f"\nBenchmark on real data completed. Results saved to: {output_path}")
    print(f"Best series data saved for statistical analysis to: {best_series_json_path}")
    print(f"Total de configuraciones is_best encontradas: {len(best_series_data)}")

    # Análisis de rendimiento por tipo de serie
    print("\n" + "="*80)
    print(" "*20 + "ANÁLISIS POR TIPO DE SERIE")
    print("="*80)
    
    # Agregar análisis por categoría de serie desde best_series_data
    for best_entry in best_series_data:
        detailed_results = best_entry.get('detailed_test_results', [])
        if not detailed_results:
            continue
        
        # Agrupar por combinación de categorías
        resultados_por_categoria = {}
        for result in detailed_results:
            cat_ruido = result.get('clasificacion_categoria_ruido')
            cat_cambio = result.get('clasificacion_categoria_cambio')
            tipo_cambio = result.get('clasificacion_tipo_cambio')
            
            if cat_ruido and cat_cambio:
                key = f"{cat_ruido}_ruido_{cat_cambio}_cambio"
                if key not in resultados_por_categoria:
                    resultados_por_categoria[key] = []
                resultados_por_categoria[key].append(result)
        
        if resultados_por_categoria:
            print(f"\nAlgoritmo: {best_entry['algorithm_key']}")
            for categoria, series_list in sorted(resultados_por_categoria.items()):
                f1_scores = [s['f1'] for s in series_list if s.get('f1') is not None]
                if f1_scores:
                    # Formatear el nombre de la categoría para que sea legible
                    cat_name = categoria.replace('_', ' ').title()
                    print(f"  {cat_name:30s}: {len(series_list):2d} series | F1 promedio: {np.mean(f1_scores):.3f}")

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