def f1_score_with_tolerance(real_changes, detected_changes, delta):
    """
    Calcula F1 Score para detección de puntos de cambio con tolerancia temporal delta.
    Args:
        real_changes (list): Lista de posiciones de cambios reales.
        detected_changes (list): Lista de posiciones de cambios detectados.
        delta (int): Tolerancia temporal.
    Returns:
        dict: TP, FP, FN, precision, recall, f1
    """
    real_changes = sorted(real_changes)
    detected_changes = sorted(detected_changes)
    tp = 0
    fp = 0
    fn = 0
    matched_detected = set()
    # Emparejamiento
    for rc in real_changes:
        match = [dc for dc in detected_changes if abs(dc - rc) <= delta and dc not in matched_detected]
        if match:
            tp += 1
            matched_detected.add(match[0])
        else:
            fn += 1
    for dc in detected_changes:
        if dc not in matched_detected:
            fp += 1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return {
        'TP': tp,
        'FP': fp,
        'FN': fn,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }