import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Any
import os
import pickle


def sample_with_min_distance(n_changes: int, n_points: int, min_dist: int = 3, seed: int | None = None) -> List[int]:
    """
    Genera `n_changes` enteros en el rango [5, n_points-5) asegurando una distancia mínima `min_dist`.
    """
    if seed is not None:
        np.random.seed(seed)

    selected = []
    candidates = list(range(5, n_points - 5))

    while len(selected) < n_changes and candidates:
        val = int(np.random.choice(candidates))
        selected.append(val)
        candidates = [c for c in candidates if abs(c - val) >= min_dist]

    return sorted(selected)


def generate_synthetic_series_nsr(
    n_points: int = 500,
    n_changes: int = 3,
    nsr_target: float = 1.5,
    change_strength: str = 'medio',
    seed: int | None = None,
    plot: bool = False,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Genera una serie sintética con cambios en tendencia (pendiente) y saltos de nivel.

    Returns:
        series: señal con ruido (n_points,)
        signal: señal base sin ruido (n_points,)
        change_points: lista de índices de cambio
    """
    if seed is not None:
        np.random.seed(seed)

    if change_strength == 'suave':
        slope_change = 0.15
        level_jump = 0
    elif change_strength == 'medio':
        slope_change = 0.6
        level_jump = 4
    elif change_strength == 'fuerte':
        slope_change = 2.4
        level_jump = 8
    else:
        raise ValueError("change_strength debe ser 'suave', 'medio' o 'fuerte'")

    change_points = sample_with_min_distance(n_changes, n_points, min_dist=30, seed=seed)

    signal = np.zeros(n_points)
    slope = np.random.uniform(-0.1, 0.1)
    level = 0
    j = 0

    for i in range(n_points):
        # usar len(change_points) por si el muestreo no alcanza n_changes
        if j < len(change_points) and i == change_points[j]:
            slope += np.random.choice([-1, 1]) * slope_change
            level += np.random.choice([-1, 1]) * level_jump
            j += 1
        level += slope
        signal[i] = level

    var_signal = np.var(signal)
    var_noise = nsr_target * var_signal
    noise_std = np.sqrt(var_noise)
    noise = np.random.normal(0, noise_std, size=n_points)
    series = signal + noise

    if plot:
        plt.figure(figsize=(12, 5))
        plt.plot(series, label="Serie sintética (con ruido)")
        plt.plot(signal, label="Señal base", linestyle='--')
        for cp in change_points:
            plt.axvline(cp, color='red', linestyle='--', alpha=0.5)
        plt.title(f"Serie con puntos de cambio en tendencia ({change_strength}), NSR={nsr_target}")
        plt.xlabel("Tiempo")
        plt.ylabel("Valor")
        plt.legend()
        plt.grid(True)
        plt.show()

    return series, signal, change_points


def generate_dataset(
    n_series_per_level: int = 10,
    n_points: int = 500,
    n_changes: int = 3,
    nsr_target: float = 1.5,
    seed: int | None = None,
    plot: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """
    Genera un dataset con las mismas claves que devuelve `load_all()` en `data_processing.py`.

    Devuelve un dict con claves 'fuerte','medio','suave', cada uno con:
      - 'time_index': np.ndarray
      - 'series': pandas.DataFrame (cada fila es una serie)
      - 'changepoints': list de listas de enteros
    """
    if seed is not None:
        np.random.seed(seed)

    strengths = {
        'fuerte': 'fuerte',
        'medio': 'medio',
        'suave': 'suave'
    }

    data: Dict[str, Dict[str, Any]] = {}
    time_index = np.arange(n_points)

    for level, strength in strengths.items():
        series_list = []
        changepoints_list = []
        for i in range(n_series_per_level):
            s_seed = None if seed is None else int(seed + hash((level, i)) % (2 ** 31))
            series, signal, cps = generate_synthetic_series_nsr(
                n_points=n_points,
                n_changes=n_changes,
                nsr_target=nsr_target,
                change_strength=strength,
                seed=s_seed,
                plot=False,
            )
            series_list.append(series)
            changepoints_list.append(cps)

            if plot:
                # mostrar solo la primera serie de cada nivel para no saturar
                if i == 0:
                    _, _, _ = generate_synthetic_series_nsr(
                        n_points=n_points,
                        n_changes=n_changes,
                        nsr_target=nsr_target,
                        change_strength=strength,
                        seed=s_seed,
                        plot=True,
                    )

        # Cada fila representará una serie (misma estructura que los CSV antiguos)
        df_series = pd.DataFrame(series_list)
        data[level] = {
            'time_index': time_index,
            'series': df_series,
            'changepoints': changepoints_list,
        }

    return data


def generate_balanced_dataset(
    n_series_per_level: int = 300,
    seed: int | None = None,
    n_points_choices: List[int] | None = None,
    n_changes_choices: List[int] | None = None,
    nsr_ranges: Dict[str, Tuple[float, float]] | None = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Genera un dataset balanceado por nivel de ruido y fuerza de cambio.

    - Niveles de ruido (claves): 'fuerte', 'medio', 'suave' (se guardan en los mismos CSVs)
    - Para cada nivel de ruido: genera exactamente n_series_per_level series
      distribuidas equitativamente por fuerza de cambio: 1/3 'fuerte', 1/3 'medio', 1/3 'suave'.
    - Para cada serie, n_points y n_changes se eligen aleatoriamente de listas provistas.
    - nsr_target se muestrea uniformemente del rango provisto para el nivel de ruido.

    Devuelve un dict con claves por nivel de ruido, cada uno con:
      - 'time_index': np.ndarray (longitud = max(n_points) del nivel)
      - 'series': pandas.DataFrame (filas=series, columnas=tiempo; padding con NaN si aplica)
      - 'changepoints': lista de listas de enteros
    """
    if seed is not None:
        np.random.seed(seed)

    if n_points_choices is None:
        n_points_choices = [100, 200, 300]
    if n_changes_choices is None:
        n_changes_choices = [1, 2, 3, 4]
    if nsr_ranges is None:
        # Rango sugerido por usuario para cada nivel de ruido
        nsr_ranges = {
            'fuerte': (1.3, 45.0),     # ruido alto
            'medio': (0.3, 1.3),       # ruido medio
            'suave': (0.05, 0.3),      # ruido bajo (evitar 0 exacto)
        }

    strengths = ['fuerte', 'medio', 'suave']  # fuerza de cambio
    per_strength = max(1, n_series_per_level // 3)

    data: Dict[str, Dict[str, Any]] = {}

    for noise_level in ['fuerte', 'medio', 'suave']:
        series_list: List[np.ndarray] = []
        changepoints_list: List[List[int]] = []
        lengths: List[int] = []

        low, high = nsr_ranges[noise_level]

        # Distribuir equitativamente por fuerza de cambio
        for chg_strength in strengths:
            for i in range(per_strength):
                # parametros aleatorios por serie
                n_points = int(np.random.choice(n_points_choices))
                n_changes = int(np.random.choice(n_changes_choices))
                nsr_target = float(np.random.uniform(low, high))

                s_seed = None if seed is None else int(
                    seed + hash((noise_level, chg_strength, i)) % (2 ** 31)
                )
                series, _, cps = generate_synthetic_series_nsr(
                    n_points=n_points,
                    n_changes=n_changes,
                    nsr_target=nsr_target,
                    change_strength=chg_strength,
                    seed=s_seed,
                    plot=False,
                )
                series_list.append(series)
                changepoints_list.append(cps)
                lengths.append(n_points)

        # Padding a la longitud maxima del nivel para tener DataFrame rectangular
        max_len = max(lengths) if lengths else 0
        padded = []
        for s in series_list:
            if s.size < max_len:
                pad = np.full(max_len - s.size, np.nan)
                padded.append(np.concatenate([s, pad]))
            else:
                padded.append(s)

        df_series = pd.DataFrame(padded)
        time_index = np.arange(max_len)
        data[noise_level] = {
            'time_index': time_index,
            'series': df_series,
            'changepoints': changepoints_list,
        }

    return data

def save_dataset_to_disk(data: Dict[str, Dict[str, Any]], data_dir: str) -> None:
    """
    Guarda el dataset generado en disco en la misma estructura que espera `data_processing.py`.

    - series para cada nivel se guardan en CSV: cada fila es una serie (sin header),
      la primera fila será un índice de tiempo (0..n_points-1) para compatibilidad.
    - changepoints para cada nivel se guardan en pickles (lista de listas).
    """
    os.makedirs(data_dir, exist_ok=True)

    level_to_series_filename = {
        'fuerte': 'ruidoAlto.csv',
        'medio': 'ruidoMedio.csv',
        'suave': 'ruidoSuave.csv'
    }

    level_to_cps_filename = {
        'fuerte': 'cpsFuerte',
        'medio': 'cpsMedio',
        'suave': 'cpsSuave'
    }

    for level in ['fuerte', 'medio', 'suave']:
        info = data[level]
        time_index = info['time_index']
        df_series = info['series']
        cps = info['changepoints']

        # Compose CSV: first row time_index, then each serie as row
        csv_path = os.path.join(data_dir, level_to_series_filename[level])
        with open(csv_path, 'w', encoding='utf8') as f:
            # time index
            f.write(','.join(map(str, list(time_index))) + '\n')
            # series rows
            for _, row in df_series.iterrows():
                f.write(','.join(map(str, row.values.tolist())) + '\n')

        # save changepoints pickle
        cps_path = os.path.join(data_dir, level_to_cps_filename[level])
        with open(cps_path, 'wb') as f:
            pickle.dump(cps, f)

