# --- Grid search: rangos de parámetros ---
import numpy as np
import random
from datetime import datetime

# Importar funciones de src y cargar los datos
import sys
import os
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from src.split_train_test import split_train_test
from src.allan_synthetic import generate_dataset, save_dataset_to_disk
from src.algorithms.page_hinkley import detect_changepoints_page_hinkley
from src.algorithms.ewma import detect_changepoints_ewma
from src.f1_score import f1_score_with_tolerance
from src.mttd import mean_time_to_detection
import pandas as pd

# Configuration via environment variables (easy to change without editing code)
N_SERIES_PER_LEVEL = int(os.environ.get('N_SERIES_PER_LEVEL', '100'))
N_POINTS = int(os.environ.get('N_POINTS', '500'))
N_CHANGES = int(os.environ.get('N_CHANGES', '9'))
NSR_TARGET = float(os.environ.get('NSR_TARGET', '1.5'))
SEED = int(os.environ.get('SEED', '42'))
VERBOSE = os.environ.get('VERBOSE', '0') in ('1', 'true', 'True')
N_TEST = int(os.environ.get('N_TEST', '1'))
ALGS = [a.strip() for a in os.environ.get('ALGS', 'page_hinkley,ewma').split(',') if a.strip()]

# Generate synthetic data using the new Allan-based generator (two noise levels: alto/bajo)
TIPO_CAMBIO = os.environ.get('TIPO_CAMBIO', 'mixed')    # 'escalon' | 'pendiente' | 'mixed'
data = generate_dataset(
    n_series_per_level=N_SERIES_PER_LEVEL,
    seed=SEED,
    n_points_choices=[100, 200, 300],
    n_changes_choices=[1, 2, 3, 4],
    tipo_cambio=TIPO_CAMBIO,
)
# Persist generated data for reproducibility
data_dir = os.path.join(os.path.dirname(__file__), 'data')
save_dataset_to_disk(data, data_dir)

split = split_train_test(data, n_test=N_TEST, seed=SEED)
LEVELS = list(split.keys())  # e.g., ['alto', 'bajo']

# Parameter grids (expandable) for each algorithm
param_grids = {
    'page_hinkley': {
        # Fast, minimal grid (override in code if needed)
        'thresholds': [20, 50, 80],
        'min_instances': [5, 20],
        'deltas': [0.001, 0.005]
    },
    'ewma': {
        'alphas': [0.05, 0.2],
        'thresholds': [2.0, 3.0],
        'min_instances': [5, 10]
    }
}

# Evaluation tolerance for matching change points
DELTA_EVAL = int(os.environ.get('DELTA_EVAL', '10'))

if VERBOSE:
    print(f"Generé dataset sintético: niveles=['fuerte','medio','suave'], n_series_per_level={N_SERIES_PER_LEVEL}, n_points={N_POINTS}")

def grid_search_and_select_best(algorithm_name: str):
    """Run grid search on train and return best params per level and best global params.

    Returns dict with 'best_by_level' and 'best_global' entries.
    """
    best_by_level = {}
    best_global = None
    best_global_score = -np.inf

    if algorithm_name == 'page_hinkley':
        thresholds = param_grids['page_hinkley']['thresholds']
        min_instances_list = param_grids['page_hinkley']['min_instances']
        deltas = param_grids['page_hinkley']['deltas']

        # Per-level
        for level in LEVELS:
            series_train = split[level]['train']['series']
            changepoints_train = split[level]['train']['changepoints']
            best_score = -np.inf
            best_params = None

            for threshold in thresholds:
                for min_instances in min_instances_list:
                    for delta in deltas:
                        scores = []
                        for idx, serie in series_train.iterrows():
                            detected = detect_changepoints_page_hinkley(
                                serie.values, threshold=threshold, min_instances=min_instances, delta=delta
                            )
                            r = f1_score_with_tolerance(changepoints_train[idx], detected, DELTA_EVAL)
                            scores.append(r['f1'])
                        avg = np.mean(scores) if scores else -np.inf
                        if avg > best_score:
                            best_score = avg
                            best_params = {'threshold': threshold, 'min_instances': min_instances, 'delta': delta, 'avg_f1_train': float(avg)}

            best_by_level[level] = best_params

        # Global
        for threshold in thresholds:
            for min_instances in min_instances_list:
                for delta in deltas:
                    scores = []
                    for level in LEVELS:
                        series_train = split[level]['train']['series']
                        changepoints_train = split[level]['train']['changepoints']
                        for idx, serie in series_train.iterrows():
                            detected = detect_changepoints_page_hinkley(
                                serie.values, threshold=threshold, min_instances=min_instances, delta=delta
                            )
                            r = f1_score_with_tolerance(changepoints_train[idx], detected, DELTA_EVAL)
                            scores.append(r['f1'])
                    avg = np.mean(scores) if scores else -np.inf
                    if avg > best_global_score:
                        best_global_score = avg
                        best_global = {'threshold': threshold, 'min_instances': min_instances, 'delta': delta, 'avg_f1_train': float(avg)}

    elif algorithm_name == 'ewma':
        alphas = param_grids['ewma']['alphas']
        thresholds = param_grids['ewma']['thresholds']
        min_instances_list = param_grids['ewma']['min_instances']

        for level in LEVELS:
            series_train = split[level]['train']['series']
            changepoints_train = split[level]['train']['changepoints']
            best_score = -np.inf
            best_params = None

            for alpha in alphas:
                for threshold in thresholds:
                    for min_instances in min_instances_list:
                        scores = []
                        for idx, serie in series_train.iterrows():
                            detected = detect_changepoints_ewma(serie.values, alpha=alpha, threshold=threshold, min_instances=min_instances)
                            r = f1_score_with_tolerance(changepoints_train[idx], detected, DELTA_EVAL)
                            scores.append(r['f1'])
                        avg = np.mean(scores) if scores else -np.inf
                        if avg > best_score:
                            best_score = avg
                            best_params = {'alpha': alpha, 'threshold': threshold, 'min_instances': min_instances, 'avg_f1_train': float(avg)}

            best_by_level[level] = best_params

        for alpha in alphas:
            for threshold in thresholds:
                for min_instances in min_instances_list:
                    scores = []
                    for level in LEVELS:
                        series_train = split[level]['train']['series']
                        changepoints_train = split[level]['train']['changepoints']
                        for idx, serie in series_train.iterrows():
                            detected = detect_changepoints_ewma(serie.values, alpha=alpha, threshold=threshold, min_instances=min_instances)
                            r = f1_score_with_tolerance(changepoints_train[idx], detected, DELTA_EVAL)
                            scores.append(r['f1'])
                    avg = np.mean(scores) if scores else -np.inf
                    if avg > best_global_score:
                        best_global_score = avg
                        best_global = {'alpha': alpha, 'threshold': threshold, 'min_instances': min_instances, 'avg_f1_train': float(avg)}

    return {'best_by_level': best_by_level, 'best_global': best_global}


def evaluate_on_test(algorithm_name: str, best_by_level: dict, best_global: dict):
    """Evaluate both strategies on test and return aggregated results and a summary row.
    Only minimal logging is produced.
    """
    results = []

    # strategy: per-level
    for level in LEVELS:
        params = best_by_level.get(level)
        series_test = split[level]['test']['series']
        changepoints_test = split[level]['test']['changepoints']
        for idx, serie in series_test.iterrows():
            if algorithm_name == 'page_hinkley':
                detected = detect_changepoints_page_hinkley(serie.values, threshold=params['threshold'], min_instances=params['min_instances'], delta=params['delta'])
            else:
                detected = detect_changepoints_ewma(serie.values, alpha=params.get('alpha', 0.1), threshold=params.get('threshold', 3.0), min_instances=params.get('min_instances', 1))
            eval_r = f1_score_with_tolerance(changepoints_test[idx], detected, DELTA_EVAL)
            mttd = mean_time_to_detection(changepoints_test[idx], detected, DELTA_EVAL)
            results.append({'algorithm': algorithm_name, 'strategy': 'per_level', 'level': level, 'id_series': idx, **params, 'TP': eval_r['TP'], 'FP': eval_r['FP'], 'FN': eval_r['FN'], 'precision': eval_r['precision'], 'recall': eval_r['recall'], 'f1': eval_r['f1'], 'MTTD': mttd})

    # strategy: global
    params = best_global
    for level in LEVELS:
        series_test = split[level]['test']['series']
        changepoints_test = split[level]['test']['changepoints']
        for idx, serie in series_test.iterrows():
            if algorithm_name == 'page_hinkley':
                detected = detect_changepoints_page_hinkley(serie.values, threshold=params['threshold'], min_instances=params['min_instances'], delta=params['delta'])
            else:
                detected = detect_changepoints_ewma(serie.values, alpha=params.get('alpha', 0.1), threshold=params.get('threshold', 3.0), min_instances=params.get('min_instances', 1))
            eval_r = f1_score_with_tolerance(changepoints_test[idx], detected, DELTA_EVAL)
            mttd = mean_time_to_detection(changepoints_test[idx], detected, DELTA_EVAL)
            results.append({'algorithm': algorithm_name, 'strategy': 'global', 'level': level, 'id_series': idx, **params, 'TP': eval_r['TP'], 'FP': eval_r['FP'], 'FN': eval_r['FN'], 'precision': eval_r['precision'], 'recall': eval_r['recall'], 'f1': eval_r['f1'], 'MTTD': mttd})

    # summary row to store in CSV: best params per level + best global
    summary = {
        'algorithm': algorithm_name,
        'fecha_evaluacion': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'mejores_parametros_por_nivel': json.dumps(best_by_level),
        'mejores_parametros_globales': json.dumps(best_global),
    }

    return results, summary


all_results = []
summaries = []

for alg in ALGS:
    if VERBOSE:
        print(f"Grid search para: {alg} ... (esto puede tardar)")
    bests = grid_search_and_select_best(alg)
    results, summary = evaluate_on_test(alg, bests['best_by_level'], bests['best_global'])
    all_results.extend(results)
    summaries.append(summary)

# Save summaries to CSV (fixed schema) and keep minimal console output
csv_filename = 'resultados_algoritmos.csv'

# Define a stable schema with 3 params per level and 3 for global (param meaning depends on algorithm)
cols = [
    'algorithm', 'fecha_evaluacion', 'n_series_per_level', 'n_points', 'seed',
    # Page-Hinkley per-level (alto/bajo)
    'alto_ph_threshold', 'alto_ph_min_instances', 'alto_ph_delta', 'alto_ph_avg_f1_train',
    'bajo_ph_threshold', 'bajo_ph_min_instances', 'bajo_ph_delta', 'bajo_ph_avg_f1_train',
    # Page-Hinkley global
    'global_ph_threshold', 'global_ph_min_instances', 'global_ph_delta', 'global_ph_avg_f1_train',
    # EWMA per-level (alto/bajo)
    'alto_ewma_alpha', 'alto_ewma_threshold', 'alto_ewma_min_instances', 'alto_ewma_avg_f1_train',
    'bajo_ewma_alpha', 'bajo_ewma_threshold', 'bajo_ewma_min_instances', 'bajo_ewma_avg_f1_train',
    # EWMA global
    'global_ewma_alpha', 'global_ewma_threshold', 'global_ewma_min_instances', 'global_ewma_avg_f1_train',
    # Test aggregates
    'global_test_f1_avg', 'global_test_mttd_avg'
]

rows_to_write = []
for s in summaries:
    alg = s['algorithm']
    fecha = s['fecha_evaluacion']
    best_by_level = json.loads(s.get('mejores_parametros_por_nivel', '{}')) if isinstance(s.get('mejores_parametros_por_nivel'), str) else s.get('mejores_parametros_por_nivel', {})
    best_global = json.loads(s.get('mejores_parametros_globales', '{}')) if isinstance(s.get('mejores_parametros_globales'), str) else s.get('mejores_parametros_globales', {})

    # helpers to extract values safely
    def ph_triplet(d):
        if not d:
            return (None, None, None, None)
        return (d.get('threshold'), d.get('min_instances'), d.get('delta'), d.get('avg_f1_train'))

    def ewma_triplet(d):
        if not d:
            return (None, None, None, None)
        return (d.get('alpha'), d.get('threshold'), d.get('min_instances'), d.get('avg_f1_train'))

    a_ph_t, a_ph_mi, a_ph_d, a_ph_avg = ph_triplet(best_by_level.get('alto', {}))
    b_ph_t, b_ph_mi, b_ph_d, b_ph_avg = ph_triplet(best_by_level.get('bajo', {}))

    a_ew_a, a_ew_t, a_ew_mi, a_ew_avg = ewma_triplet(best_by_level.get('alto', {}))
    b_ew_a, b_ew_t, b_ew_mi, b_ew_avg = ewma_triplet(best_by_level.get('bajo', {}))

    g_ph_t, g_ph_mi, g_ph_d, g_ph_avg = ph_triplet(best_global if alg == 'page_hinkley' else {})
    g_ew_a, g_ew_t, g_ew_mi, g_ew_avg = ewma_triplet(best_global if alg == 'ewma' else {})

    # compute global test aggregates from all_results
    alg_global_results = [r for r in all_results if r.get('algorithm') == alg and r.get('strategy') == 'global']
    f1_vals = [r['f1'] for r in alg_global_results if r.get('f1') is not None]
    mttd_vals = [r['MTTD'] for r in alg_global_results if r.get('MTTD') is not None]
    test_f1_avg = float(np.mean(f1_vals)) if f1_vals else None
    test_mttd_avg = float(np.mean(mttd_vals)) if mttd_vals else None

    row = {
        'algorithm': alg,
        'fecha_evaluacion': fecha,
        'n_series_per_level': N_SERIES_PER_LEVEL,
        'n_points': 'mixed',
        'seed': SEED,
        'alto_ph_threshold': a_ph_t,
        'alto_ph_min_instances': a_ph_mi,
        'alto_ph_delta': a_ph_d,
        'alto_ph_avg_f1_train': a_ph_avg,
        'bajo_ph_threshold': b_ph_t,
        'bajo_ph_min_instances': b_ph_mi,
        'bajo_ph_delta': b_ph_d,
        'bajo_ph_avg_f1_train': b_ph_avg,
        'global_ph_threshold': g_ph_t,
        'global_ph_min_instances': g_ph_mi,
        'global_ph_delta': g_ph_d,
        'global_ph_avg_f1_train': g_ph_avg,
        'alto_ewma_alpha': a_ew_a,
        'alto_ewma_threshold': a_ew_t,
        'alto_ewma_min_instances': a_ew_mi,
        'alto_ewma_avg_f1_train': a_ew_avg,
        'bajo_ewma_alpha': b_ew_a,
        'bajo_ewma_threshold': b_ew_t,
        'bajo_ewma_min_instances': b_ew_mi,
        'bajo_ewma_avg_f1_train': b_ew_avg,
        'global_ewma_alpha': g_ew_a,
        'global_ewma_threshold': g_ew_t,
        'global_ewma_min_instances': g_ew_mi,
        'global_ewma_avg_f1_train': g_ew_avg,
        'global_test_f1_avg': test_f1_avg,
        'global_test_mttd_avg': test_mttd_avg,
    }
    rows_to_write.append(row)

df_new = pd.DataFrame(rows_to_write, columns=cols)

# If existing file has same schema, append; else backup old file and write new CSV
if os.path.exists(csv_filename):
    try:
        df_exist = pd.read_csv(csv_filename, nrows=0)
        exist_cols = list(df_exist.columns)
    except Exception:
        exist_cols = []

    if exist_cols == cols:
        # safe to append
        df_old = pd.read_csv(csv_filename)
        df_comb = pd.concat([df_old, df_new], ignore_index=True)
        df_comb.to_csv(csv_filename, index=False)
    else:
        # backup old CSV and create a fresh one with consistent schema
        ts = datetime.now().strftime('%Y%m%dT%H%M%S')
        backup_name = f"{csv_filename}.backup.{ts}"
        os.rename(csv_filename, backup_name)
        df_new.to_csv(csv_filename, index=False)
        print(f"Se encontró un CSV con formato antiguo; lo he renombrado a: {backup_name}")
else:
    df_new.to_csv(csv_filename, index=False)

# Minimal console output requested
print(f"Ejecución completa. Resúmenes guardados en: {csv_filename}")
for r in rows_to_write:
    alg = r['algorithm']
    if alg == 'page_hinkley':
        g = {
            'threshold': r.get('global_ph_threshold'),
            'min_instances': r.get('global_ph_min_instances'),
            'delta': r.get('global_ph_delta'),
            'avg_f1_train': r.get('global_ph_avg_f1_train')
        }
    else:
        g = {
            'alpha': r.get('global_ewma_alpha'),
            'threshold': r.get('global_ewma_threshold'),
            'min_instances': r.get('global_ewma_min_instances'),
            'avg_f1_train': r.get('global_ewma_avg_f1_train')
        }
    test_f1 = r.get('global_test_f1_avg')
    test_mttd = r.get('global_test_mttd_avg')
    print(f"{alg}: mejores globales -> {g}; test_f1_avg={test_f1}, test_mttd_avg={test_mttd}")

# Summary by noise levels (alto/bajo)
for alg in ALGS:
    tag = 'ph' if alg == 'page_hinkley' else 'ewma'
    alg_global_results = [x for x in all_results if x.get('algorithm') == alg and x.get('strategy') == 'global']

    def _lvl_avg(level):
        subset = [x for x in alg_global_results if x.get('level') == level]
        f1_vals = [x['f1'] for x in subset if x.get('f1') is not None]
        mttd_vals = [x['MTTD'] for x in subset if x.get('MTTD') is not None]
        f1 = float(np.mean(f1_vals)) if f1_vals else None
        mttd = float(np.mean(mttd_vals)) if mttd_vals else None
        return f1, mttd

    b_f1, b_mttd = _lvl_avg('bajo')
    a_f1, a_mttd = _lvl_avg('alto')

    # global averages across levels
    f1_vals_all = [x['f1'] for x in alg_global_results if x.get('f1') is not None]
    mttd_vals_all = [x['MTTD'] for x in alg_global_results if x.get('MTTD') is not None]
    g_f1 = float(np.mean(f1_vals_all)) if f1_vals_all else None
    g_mttd = float(np.mean(mttd_vals_all)) if mttd_vals_all else None

    print(f"{tag} bajo -> f1={b_f1}, mttd={b_mttd}")
    print(f"{tag} alto -> f1={a_f1}, mttd={a_mttd}")
    print(f"{tag} global -> f1={g_f1}, mttd={g_mttd}")
