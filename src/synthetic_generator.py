import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple


def generar_serie_sintetica(
    longitud: int = 500,
    nivel_ruido: float = 1.0,
    num_cambios: int = 3,
    fuerza_cambio: float = 5.0,
    tipo_cambio: str = 'escalon',
    seed: int | None = None,
) -> tuple[np.ndarray, List[int]]:
    """
    Genera una serie de tiempo sintética con control sobre ruido y puntos de cambio.

    Parámetros
    -----------
    longitud : int
        Longitud de la serie (default=500)
    nivel_ruido : float
        Escala del ruido gaussiano (std) (default=1.0)
    num_cambios : int
        Número de puntos de cambio (default=3)
    fuerza_cambio : float
        Magnitud del cambio por punto (default=5.0)
    tipo_cambio : str
        'escalon' (salto permanente) o 'pendiente' (cambio gradual) (default='escalon')
    seed : int | None
        Semilla para reproducibilidad

    Retorna
    --------
    serie : np.ndarray (longitud,)
    puntos_cambio : list[int]
    """
    if seed is not None:
        np.random.seed(seed)

    # Elegir tendencia base aleatoria
    trend_type = np.random.choice(["creciente", "decreciente", "plana"])  # noqa: S311
    if trend_type == "creciente":
        slope = float(np.random.uniform(0.01, 0.1))
    elif trend_type == "decreciente":
        slope = float(np.random.uniform(-0.1, -0.01))
    else:
        slope = 0.0

    t = np.arange(longitud)
    tendencia = slope * t

    serie_base = tendencia.copy()
    serie_con_cambios = tendencia.copy()

    # Puntos de cambio separados (evitar extremos)
    puntos_cambio: List[int] = []
    if num_cambios > 0:
        min_distancia = max(30, longitud // 30)
        espacio_disponible = list(range(min_distancia, longitud - min_distancia))
        for _ in range(num_cambios):
            if not espacio_disponible:
                break
            punto = int(np.random.choice(espacio_disponible))  # noqa: S311
            puntos_cambio.append(punto)
            espacio_disponible = [x for x in espacio_disponible if abs(x - punto) >= min_distancia]
    puntos_cambio.sort()

    # Aplicar cambios
    for punto in puntos_cambio:
        direccion = 1 if np.random.random() > 0.5 else -1  # noqa: S311
        magnitud = float(direccion * fuerza_cambio * (0.8 + 0.4 * np.random.random()))  # noqa: S311

        if tipo_cambio == 'escalon':
            # Cambio abrupto permanente
            serie_con_cambios[punto:] += magnitud
        elif tipo_cambio == 'pendiente':
            # Cambio gradual a lo largo del resto de la serie
            incremento = np.linspace(0, magnitud, longitud - punto)
            serie_con_cambios[punto:longitud] += incremento
        else:
            raise ValueError("tipo_cambio debe ser 'escalon' o 'pendiente'")

    ruido = nivel_ruido * np.random.normal(0, 1, longitud)
    serie_final = serie_con_cambios + ruido

    return serie_final.astype(float), puntos_cambio


def _pad_to_max_length(rows: List[np.ndarray]) -> tuple[pd.DataFrame, np.ndarray]:
    max_len = max((r.size for r in rows), default=0)
    padded: List[np.ndarray] = []
    for r in rows:
        if r.size < max_len:
            pad = np.full(max_len - r.size, np.nan)
            padded.append(np.concatenate([r, pad]))
        else:
            padded.append(r)
    return pd.DataFrame(padded), np.arange(max_len)


def generate_dataset(
    n_series_per_level: int = 90,
    seed: int | None = None,
    n_points_choices: List[int] | None = None,
    n_changes_choices: List[int] | None = None,
    nivel_ruido_ranges: Dict[str, Tuple[float, float]] | None = None,
    fuerza_cambio_ranges: Dict[str, Tuple[float, float]] | None = None,
    tipo_cambio: str = 'mixed',  # 'escalon' | 'pendiente' | 'mixed'
) -> Dict[str, Dict[str, Any]]:
    """
    Genera un dataset balanceado por nivel de ruido (claves: 'fuerte','medio','suave')
    y fuerza de cambio ('fuerte','medio','suave'), usando la lógica de Allan.

    Devuelve un dict por nivel de ruido con:
      - 'time_index': np.ndarray
      - 'series': pd.DataFrame (filas=series)
      - 'changepoints': list[list[int]]
    """
    if seed is not None:
        np.random.seed(seed)

    if n_points_choices is None:
        n_points_choices = [100, 200, 300]
    if n_changes_choices is None:
        n_changes_choices = [1, 2, 3, 4]
    if nivel_ruido_ranges is None:
        # Solo dos niveles de ruido: alto y bajo
        nivel_ruido_ranges = {
            'alto': (3.0, 6.0),
            'bajo': (0.0, 0.3),
        }
    if fuerza_cambio_ranges is None:
        fuerza_cambio_ranges = {
            'fuerte': (3.0, 6.0),
            'medio': (1.0, 3.0),
            'suave': (0.3, 1.0),
        }

    per_strength = max(1, n_series_per_level // 3)
    strengths = ['fuerte', 'medio', 'suave']

    data: Dict[str, Dict[str, Any]] = {}

    for noise_level in ['alto', 'bajo']:
        series_list: List[np.ndarray] = []
        changepoints_list: List[List[int]] = []

        ruido_low, ruido_high = nivel_ruido_ranges[noise_level]

        for chg_strength in strengths:
            chg_low, chg_high = fuerza_cambio_ranges[chg_strength]
            for i in range(per_strength):
                n_points = int(np.random.choice(n_points_choices))
                n_changes = int(np.random.choice(n_changes_choices))
                nivel_ruido = float(np.random.uniform(ruido_low, ruido_high))
                fuerza_cambio = float(np.random.uniform(chg_low, chg_high))

                if tipo_cambio == 'mixed':
                    tipo = np.random.choice(['escalon', 'pendiente'])  # noqa: S311
                else:
                    tipo = tipo_cambio

                s_seed = None if seed is None else int(seed + hash((noise_level, chg_strength, i)) % (2 ** 31))
                serie, cps = generar_serie_sintetica(
                    longitud=n_points,
                    nivel_ruido=nivel_ruido,
                    num_cambios=n_changes,
                    fuerza_cambio=fuerza_cambio,
                    tipo_cambio=tipo,
                    seed=s_seed,
                )

                series_list.append(serie)
                changepoints_list.append(cps)

        df_series, time_index = _pad_to_max_length(series_list)
        data[noise_level] = {
            'time_index': time_index,
            'series': df_series,
            'changepoints': changepoints_list,
        }

    return data


def save_dataset_to_disk(data: Dict[str, Dict[str, Any]], data_dir: str) -> None:
    """
    Guarda el dataset generado en disco con 2 niveles de ruido: 'alto' y 'bajo'.

    - series por nivel en CSV: primera fila el índice temporal, resto filas cada serie
    - changepoints por nivel en pickle
    """
    import os
    import pickle

    os.makedirs(data_dir, exist_ok=True)

    level_to_series_filename = {
        'alto': 'ruidoAlto.csv',
        'bajo': 'ruidoBajo.csv',
    }

    level_to_cps_filename = {
        'alto': 'cpsAlto',
        'bajo': 'cpsBajo',
    }

    for level in ['alto', 'bajo']:
        info = data[level]
        time_index = info['time_index']
        df_series = info['series']
        cps = info['changepoints']

        # CSV: first row is time_index, then rows are series values
        csv_path = os.path.join(data_dir, level_to_series_filename[level])
        with open(csv_path, 'w', encoding='utf8') as f:
            f.write(','.join(map(str, list(time_index))) + '\n')
            for _, row in df_series.iterrows():
                f.write(','.join(map(str, row.values.tolist())) + '\n')

        cps_path = os.path.join(data_dir, level_to_cps_filename[level])
        with open(cps_path, 'wb') as f:
            pickle.dump(cps, f)
