"""
Script para analizar resultados de los benchmarks y generar contenido para la sección de Resultados
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

# Cargar datos
print("="*80)
print("ANÁLISIS DE RESULTADOS - BENCHMARKS DE DETECCIÓN DE CHANGE POINTS")
print("="*80)

# ============================================================================
# 1. BENCHMARK SYNTHETIC DATA
# ============================================================================
print("\n" + "="*80)
print("1. BENCHMARK: SYNTHETIC DATA")
print("="*80)

df_synthetic = pd.read_csv("10-03-2025-resultados_algoritmos_main2-synthetic.csv")

# Filtrar solo los mejores trials (is_best = True)
df_synthetic_best = df_synthetic[df_synthetic['is_best'] == True].copy()

print(f"\nTotal de experimentos: {len(df_synthetic)}")
print(f"Mejores configuraciones por algoritmo-categoría: {len(df_synthetic_best)}")

# Ranking general por F1-score en test
print("\n" + "-"*80)
print("1.1 RANKING GENERAL DE ALGORITMOS (F1-score en test)")
print("-"*80)

ranking_general = df_synthetic_best.groupby('algorithm_key').agg({
    'test_f1_mean': 'mean',
    'test_precision_mean': 'mean',
    'test_recall_mean': 'mean',
    'test_mttd_mean': 'mean',
    'test_mmd_mean': 'mean'
}).sort_values('test_f1_mean', ascending=False)

print("\nTop 10 algoritmos por F1-score promedio:")
print(ranking_general.head(10).to_string())

# Análisis por categoría
print("\n" + "-"*80)
print("1.2 RENDIMIENTO POR CATEGORÍA")
print("-"*80)

categorias = df_synthetic_best.groupby(['tipo_cambio', 'nivel_ruido', 'fuerza_cambio'])['algorithm_key'].count()
print(f"\nCategorías evaluadas: {len(categorias)}")
print(categorias.to_string())

# Mejor algoritmo por categoría
print("\n" + "-"*80)
print("1.3 MEJOR ALGORITMO POR CATEGORÍA (F1-score)")
print("-"*80)

for tipo in ['escalon', 'pendiente']:
    for ruido in ['alto', 'bajo']:
        for fuerza in ['alto', 'bajo']:
            subset = df_synthetic_best[
                (df_synthetic_best['tipo_cambio'] == tipo) &
                (df_synthetic_best['nivel_ruido'] == ruido) &
                (df_synthetic_best['fuerza_cambio'] == fuerza)
            ]
            if len(subset) > 0:
                best_row = subset.loc[subset['test_f1_mean'].idxmax()]
                print(f"\n{tipo.upper()} | Ruido {ruido} | Cambio {fuerza}:")
                print(f"  Ganador: {best_row['algorithm_key']}")
                print(f"  F1: {best_row['test_f1_mean']:.4f}")
                print(f"  Precision: {best_row['test_precision_mean']:.4f}")
                print(f"  Recall: {best_row['test_recall_mean']:.4f}")
                print(f"  MTTD: {best_row['test_mttd_mean']:.2f}")

# Análisis por tipo de cambio
print("\n" + "-"*80)
print("1.4 RENDIMIENTO POR TIPO DE CAMBIO")
print("-"*80)

tipo_cambio_stats = df_synthetic_best.groupby(['tipo_cambio', 'algorithm_key'])['test_f1_mean'].mean().unstack()
print("\nF1-score promedio por tipo de cambio:")
print(tipo_cambio_stats.to_string())

# Análisis por nivel de ruido
print("\n" + "-"*80)
print("1.5 RENDIMIENTO POR NIVEL DE RUIDO")
print("-"*80)

ruido_stats = df_synthetic_best.groupby(['nivel_ruido', 'algorithm_key'])['test_f1_mean'].mean().unstack()
print("\nF1-score promedio por nivel de ruido:")
print(ruido_stats.to_string())

# Top 5 por librería
print("\n" + "-"*80)
print("1.6 RENDIMIENTO POR LIBRERÍA")
print("-"*80)

library_ranking = df_synthetic_best.groupby('algorithm_library').agg({
    'test_f1_mean': 'mean',
    'test_precision_mean': 'mean',
    'test_recall_mean': 'mean'
}).sort_values('test_f1_mean', ascending=False)

print("\nRendimiento promedio por librería:")
print(library_ranking.to_string())

# ============================================================================
# 2. BENCHMARK REAL CRIME DATA
# ============================================================================
print("\n\n" + "="*80)
print("2. BENCHMARK: REAL CRIME DATA")
print("="*80)

df_real = pd.read_csv("10-13-2025-resultados_algoritmos_main3_real-real.csv")

print(f"\nTotal de algoritmos evaluados: {len(df_real)}")
print(f"Series de entrenamiento: {df_real['train_series'].iloc[0]}")
print(f"Series de prueba: {df_real['test_series'].iloc[0]}")

# Ranking real data
print("\n" + "-"*80)
print("2.1 RANKING DE ALGORITMOS EN DATOS REALES")
print("-"*80)

ranking_real = df_real[['algorithm_key', 'test_f1_mean', 'test_precision_mean', 
                         'test_recall_mean', 'test_mttd_mean', 'test_mmd_mean']].sort_values(
    'test_f1_mean', ascending=False)

print("\nTop 10 algoritmos en datos reales:")
print(ranking_real.head(10).to_string(index=False))

# Comparación sintético vs real
print("\n" + "-"*80)
print("2.2 COMPARACIÓN: SINTÉTICO vs REAL")
print("-"*80)

# Obtener F1 promedio de sintético para comparación
synthetic_avg = df_synthetic_best.groupby('algorithm_key')['test_f1_mean'].mean()
real_avg = df_real.set_index('algorithm_key')['test_f1_mean']

comparison = pd.DataFrame({
    'F1_Synthetic': synthetic_avg,
    'F1_Real': real_avg
}).dropna()

comparison['Difference'] = comparison['F1_Real'] - comparison['F1_Synthetic']
comparison = comparison.sort_values('F1_Real', ascending=False)

print("\nComparación F1-score (Sintético vs Real):")
print(comparison.head(15).to_string())

# ============================================================================
# 3. BENCHMARK TCPD
# ============================================================================
print("\n\n" + "="*80)
print("3. BENCHMARK: TCPD REPOSITORY")
print("="*80)

df_tcpd = pd.read_csv("10-03-2025-resultados_tcpd_benchmark.csv")

print(f"\nTotal de experimentos: {len(df_tcpd)}")
print(f"Datasets evaluados: {df_tcpd['dataset_name'].nunique()}")
print(f"Algoritmos evaluados: {df_tcpd['algorithm_key'].nunique()}")

# Tasa de éxito por algoritmo
print("\n" + "-"*80)
print("3.1 TASA DE ÉXITO POR ALGORITMO")
print("-"*80)

success_rate = df_tcpd.groupby('algorithm_key').agg({
    'status': lambda x: (x == 'success').sum() / len(x) * 100,
    'runtime_seconds': 'mean',
    'n_changepoints_detected': 'mean'
}).round(2)

success_rate.columns = ['Success_Rate_%', 'Avg_Runtime_s', 'Avg_Detections']
success_rate = success_rate.sort_values('Success_Rate_%', ascending=False)

print("\nTasa de éxito y rendimiento por algoritmo:")
print(success_rate.to_string())

# Detecciones por dataset
print("\n" + "-"*80)
print("3.2 DETECCIONES POR DATASET")
print("-"*80)

dataset_stats = df_tcpd[df_tcpd['status'] == 'success'].groupby('dataset_name').agg({
    'n_changepoints_detected': ['mean', 'std', 'min', 'max'],
    'runtime_seconds': 'mean'
}).round(3)

print("\nEstadísticas de detección por dataset (solo experimentos exitosos):")
print(dataset_stats.head(15).to_string())

# ============================================================================
# 4. RESUMEN EJECUTIVO
# ============================================================================
print("\n\n" + "="*80)
print("4. RESUMEN EJECUTIVO")
print("="*80)

print("\n📊 HALLAZGOS PRINCIPALES:\n")

# Top 3 global (sintético)
top3_synthetic = ranking_general.head(3)
print("🏆 Top 3 Algoritmos (Datos Sintéticos):")
for i, (alg, row) in enumerate(top3_synthetic.iterrows(), 1):
    print(f"  {i}. {alg}: F1={row['test_f1_mean']:.4f}")

# Top 3 real
top3_real = ranking_real.head(3)
print("\n🏆 Top 3 Algoritmos (Datos Reales):")
for i, row in top3_real.iterrows():
    print(f"  {i+1}. {row['algorithm_key']}: F1={row['test_f1_mean']:.4f}")

# Algoritmos más robustos (success rate TCPD)
top3_robust = success_rate.head(3)
print("\n🛡️ Top 3 Más Robustos (TCPD):")
for i, (alg, row) in enumerate(top3_robust.iterrows(), 1):
    print(f"  {i}. {alg}: {row['Success_Rate_%']:.1f}% éxito, {row['Avg_Runtime_s']:.3f}s")

# Mejor por condición
print("\n🎯 Mejor Algoritmo por Condición:")
print("  • Alto ruido + alto cambio (escalón):")
subset = df_synthetic_best[
    (df_synthetic_best['tipo_cambio'] == 'escalon') &
    (df_synthetic_best['nivel_ruido'] == 'alto') &
    (df_synthetic_best['fuerza_cambio'] == 'alto')
]
best = subset.loc[subset['test_f1_mean'].idxmax()]
print(f"    {best['algorithm_key']} (F1={best['test_f1_mean']:.4f})")

print("  • Bajo ruido + bajo cambio (pendiente):")
subset = df_synthetic_best[
    (df_synthetic_best['tipo_cambio'] == 'pendiente') &
    (df_synthetic_best['nivel_ruido'] == 'bajo') &
    (df_synthetic_best['fuerza_cambio'] == 'bajo')
]
if len(subset) > 0:
    best = subset.loc[subset['test_f1_mean'].idxmax()]
    print(f"    {best['algorithm_key']} (F1={best['test_f1_mean']:.4f})")

# Librerías
print("\n📚 Mejor Librería:")
best_lib = library_ranking.idxmax()['test_f1_mean']
print(f"  {best_lib}: F1={library_ranking.loc[best_lib, 'test_f1_mean']:.4f}")

print("\n" + "="*80)
print("Análisis completado. Resultados guardados.")
print("="*80)

# Guardar resumen en JSON
summary = {
    "synthetic": {
        "top_3_algorithms": [
            {"algorithm": alg, "f1": float(row['test_f1_mean'])} 
            for alg, row in ranking_general.head(3).iterrows()
        ],
        "best_library": best_lib,
        "total_experiments": len(df_synthetic)
    },
    "real": {
        "top_3_algorithms": [
            {"algorithm": row['algorithm_key'], "f1": float(row['test_f1_mean'])}
            for _, row in ranking_real.head(3).iterrows()
        ],
        "train_series": int(df_real['train_series'].iloc[0]),
        "test_series": int(df_real['test_series'].iloc[0])
    },
    "tcpd": {
        "total_experiments": len(df_tcpd),
        "datasets": int(df_tcpd['dataset_name'].nunique()),
        "algorithms": int(df_tcpd['algorithm_key'].nunique()),
        "most_robust": [
            {"algorithm": alg, "success_rate": float(row['Success_Rate_%'])}
            for alg, row in success_rate.head(3).iterrows()
        ]
    }
}

with open("results_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n✅ Resumen guardado en: results_summary.json")
