"""
Script para calcular el nivel de agreement entre las etiquetas de Martín y Allan
usando F1 score con tolerancia delta.

Martín es considerado como Ground Truth (GOT) y Allan como predicción.
"""

import os
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import json
from datetime import datetime
from src.f1_score import f1_score_with_tolerance


def extract_series_number(filename: str) -> str:
    """
    Extrae el número de serie del nombre del archivo.
    Ejemplo: 'serie_etiquetada_allan_datasAll-s1_...' -> 's1'
    """
    parts = filename.split('datasAll-')
    if len(parts) < 2:
        return None
    series_part = parts[1].split('_')[0]
    return series_part


def load_labels_from_csv(filepath: str) -> List[int]:
    """
    Carga las etiquetas de changepoints desde un archivo CSV.
    Retorna una lista de índices donde Is_ChangePoint es TRUE.
    """
    df = pd.read_csv(filepath)
    # La columna Is_ChangePoint puede ser booleana o string
    if df['Is_ChangePoint'].dtype == bool:
        changepoints = df[df['Is_ChangePoint'] == True]['Index'].tolist()
    else:
        changepoints = df[df['Is_ChangePoint'].str.upper() == 'TRUE']['Index'].tolist()
    return changepoints


def load_all_labels(data_path: str, labeler_prefix: str) -> Dict[str, Tuple[str, List[int]]]:
    """
    Carga todas las etiquetas de un etiquetador específico.
    
    Args:
        data_path: Ruta al directorio con los archivos
        labeler_prefix: Prefijo del etiquetador ('allan' o 'Mart_n_Sol_s_Salazar')
    
    Returns:
        Dict con key=número de serie, value=(filepath, lista de changepoints)
    """
    labels_dict = {}
    
    for filename in os.listdir(data_path):
        if filename.startswith(f'serie_etiquetada_{labeler_prefix}_datasAll-'):
            series_num = extract_series_number(filename)
            if series_num:
                filepath = os.path.join(data_path, filename)
                changepoints = load_labels_from_csv(filepath)
                labels_dict[series_num] = (filepath, changepoints)
    
    return labels_dict


def calculate_agreement(data_path: str, delta: int = 5) -> Dict:
    """
    Calcula el agreement entre las etiquetas de Martín (GOT) y Allan (predicción).
    
    Args:
        data_path: Ruta al directorio con los archivos de etiquetas
        delta: Tolerancia temporal para el F1 score
    
    Returns:
        Dict con resultados detallados y resumen
    """
    print("="*80)
    print("CALCULANDO AGREEMENT ENTRE ETIQUETADORES")
    print("="*80)
    print(f"Ground Truth (GOT): Martín")
    print(f"Predicción: Allan")
    print(f"Delta (tolerancia): {delta}")
    print("="*80)
    print()
    
    # Cargar todas las etiquetas
    martin_labels = load_all_labels(data_path, 'Mart_n_Sol_s_Salazar')
    allan_labels = load_all_labels(data_path, 'allan')
    
    # Encontrar series comunes
    common_series = sorted(set(martin_labels.keys()) & set(allan_labels.keys()))
    
    print(f"Series etiquetadas por Martín: {len(martin_labels)}")
    print(f"Series etiquetadas por Allan: {len(allan_labels)}")
    print(f"Series en común: {len(common_series)}")
    print()
    
    if not common_series:
        print("⚠️  No hay series en común entre ambos etiquetadores!")
        return None
    
    # Calcular F1 score para cada serie
    results_per_series = []
    
    for series_num in common_series:
        martin_file, martin_cps = martin_labels[series_num]
        allan_file, allan_cps = allan_labels[series_num]
        
        # Martín es GOT (real_changes), Allan es predicción (detected_changes)
        f1_result = f1_score_with_tolerance(
            real_changes=martin_cps,
            detected_changes=allan_cps,
            delta=delta
        )
        
        results_per_series.append({
            'series': series_num,
            'martin_file': os.path.basename(martin_file),
            'allan_file': os.path.basename(allan_file),
            'martin_changepoints': len(martin_cps),
            'allan_changepoints': len(allan_cps),
            'martin_cps_list': martin_cps,
            'allan_cps_list': allan_cps,
            **f1_result
        })
    
    # Calcular estadísticas generales
    f1_scores = [r['f1'] for r in results_per_series]
    precisions = [r['precision'] for r in results_per_series]
    recalls = [r['recall'] for r in results_per_series]
    
    summary = {
        'total_series': len(common_series),
        'delta': delta,
        'f1_score_mean': np.mean(f1_scores),
        'f1_score_std': np.std(f1_scores),
        'f1_score_median': np.median(f1_scores),
        'f1_score_min': np.min(f1_scores),
        'f1_score_max': np.max(f1_scores),
        'precision_mean': np.mean(precisions),
        'recall_mean': np.mean(recalls),
        'total_tp': sum(r['TP'] for r in results_per_series),
        'total_fp': sum(r['FP'] for r in results_per_series),
        'total_fn': sum(r['FN'] for r in results_per_series),
        'series_with_perfect_f1': sum(1 for f1 in f1_scores if f1 == 1.0),
        'series_with_f1_above_0.8': sum(1 for f1 in f1_scores if f1 >= 0.8),
        'series_with_f1_above_0.5': sum(1 for f1 in f1_scores if f1 >= 0.5),
        'series_with_f1_below_0.5': sum(1 for f1 in f1_scores if f1 < 0.5),
    }
    
    return {
        'summary': summary,
        'results_per_series': results_per_series,
        'timestamp': datetime.now().isoformat()
    }


def print_results(results: Dict):
    """Imprime los resultados de manera legible."""
    if not results:
        return
    
    summary = results['summary']
    
    print()
    print("="*80)
    print("RESUMEN GENERAL")
    print("="*80)
    print(f"Total de series comparadas: {summary['total_series']}")
    print(f"Delta usado: {summary['delta']}")
    print()
    print(f"📊 F1 SCORE ESTADÍSTICAS:")
    print(f"   Media: {summary['f1_score_mean']:.4f}")
    print(f"   Mediana: {summary['f1_score_median']:.4f}")
    print(f"   Desv. Est.: {summary['f1_score_std']:.4f}")
    print(f"   Mínimo: {summary['f1_score_min']:.4f}")
    print(f"   Máximo: {summary['f1_score_max']:.4f}")
    print()
    print(f"📈 PRECISION Y RECALL:")
    print(f"   Precision media: {summary['precision_mean']:.4f}")
    print(f"   Recall medio: {summary['recall_mean']:.4f}")
    print()
    print(f"🎯 TOTALES:")
    print(f"   True Positives: {summary['total_tp']}")
    print(f"   False Positives: {summary['total_fp']}")
    print(f"   False Negatives: {summary['total_fn']}")
    print()
    print(f"📋 DISTRIBUCIÓN DE F1 SCORES:")
    print(f"   Series con F1 = 1.0 (perfecto): {summary['series_with_perfect_f1']}")
    print(f"   Series con F1 ≥ 0.8 (muy bueno): {summary['series_with_f1_above_0.8']}")
    print(f"   Series con F1 ≥ 0.5 (aceptable): {summary['series_with_f1_above_0.5']}")
    print(f"   Series con F1 < 0.5 (bajo): {summary['series_with_f1_below_0.5']}")
    print()
    
    # Interpretación
    print("="*80)
    print("INTERPRETACIÓN")
    print("="*80)
    mean_f1 = summary['f1_score_mean']
    
    if mean_f1 >= 0.8:
        print("✅ EXCELENTE AGREEMENT (F1 ≥ 0.8)")
        print("   Los etiquetadores tienen un nivel de acuerdo MUY ALTO.")
        print("   ➡️  Se pueden usar ambos etiquetadores con confianza.")
    elif mean_f1 >= 0.6:
        print("⚠️  AGREEMENT MODERADO (0.6 ≤ F1 < 0.8)")
        print("   Los etiquetadores tienen un nivel de acuerdo ACEPTABLE pero con diferencias.")
        print("   ➡️  Se recomienda revisar las series con F1 bajo y decidir caso por caso.")
    else:
        print("❌ AGREEMENT BAJO (F1 < 0.6)")
        print("   Los etiquetadores tienen un nivel de acuerdo BAJO.")
        print("   ➡️  NO se recomienda usar ambos. Trabajar únicamente con Martín (GOT).")
    
    print("="*80)
    print()


def print_detailed_series(results: Dict, show_all: bool = False, min_f1: float = None):
    """
    Imprime resultados detallados por serie.
    
    Args:
        results: Diccionario de resultados
        show_all: Si True, muestra todas las series. Si False, solo las problemáticas
        min_f1: Si se especifica, solo muestra series con F1 menor a este valor
    """
    if not results:
        return
    
    series_results = results['results_per_series']
    
    if min_f1 is not None:
        series_results = [r for r in series_results if r['f1'] < min_f1]
        print(f"\n{'='*80}")
        print(f"SERIES CON F1 < {min_f1}")
        print(f"{'='*80}")
    elif not show_all:
        series_results = [r for r in series_results if r['f1'] < 0.8]
        print(f"\n{'='*80}")
        print(f"SERIES CON F1 < 0.8 (Problemáticas)")
        print(f"{'='*80}")
    else:
        print(f"\n{'='*80}")
        print(f"TODAS LAS SERIES")
        print(f"{'='*80}")
    
    if not series_results:
        print("No hay series en esta categoría.")
        return
    
    # Ordenar por F1 score ascendente
    series_results = sorted(series_results, key=lambda x: x['f1'])
    
    for r in series_results:
        print(f"\n📄 Serie: {r['series']}")
        print(f"   F1 Score: {r['f1']:.4f} | Precision: {r['precision']:.4f} | Recall: {r['recall']:.4f}")
        print(f"   TP: {r['TP']} | FP: {r['FP']} | FN: {r['FN']}")
        print(f"   Martín detectó: {r['martin_changepoints']} changepoints")
        print(f"   Allan detectó: {r['allan_changepoints']} changepoints")
        if r['martin_changepoints'] <= 10 and r['allan_changepoints'] <= 10:
            print(f"   Martín CPs: {r['martin_cps_list']}")
            print(f"   Allan CPs: {r['allan_cps_list']}")


def save_results(results: Dict, output_dir: str = "."):
    """Guarda los resultados en un archivo JSON."""
    if not results:
        return
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{timestamp}_agreement_analysis_martin_vs_allan.json"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Resultados guardados en: {filepath}")


def main():
    """Función principal."""
    # Configuración
    data_path = os.path.join(os.path.dirname(__file__), "data", "data_real")
    delta = 5  # Tolerancia temporal
    
    # Verificar que existe el directorio
    if not os.path.exists(data_path):
        print(f"❌ Error: No se encuentra el directorio {data_path}")
        return
    
    # Calcular agreement
    results = calculate_agreement(data_path, delta=delta)
    
    if not results:
        return
    
    # Imprimir resultados
    print_results(results)
    
    # Imprimir series problemáticas
    print_detailed_series(results, show_all=False)
    
    # Opción para ver todas las series
    print("\n" + "="*80)
    print("¿Deseas ver el detalle de TODAS las series? (y/n)")
    response = input().strip().lower()
    if response == 'y':
        print_detailed_series(results, show_all=True)
    
    # Guardar resultados
    save_results(results)
    
    print("\n🎉 Análisis completado!")


if __name__ == "__main__":
    main()
