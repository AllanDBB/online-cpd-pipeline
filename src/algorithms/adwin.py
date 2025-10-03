from river.drift import ADWIN
import pandas as pd

def detect_changepoints_adwin(series, delta=0.002, clock=32, max_buckets=5, min_window_length=5, grace_period=10):
    """
    Detecta changepoints usando ADWIN (Adaptive Windowing) de river.
    
    Parameters:
    -----------
    series : array-like
        Serie temporal a analizar
    delta : float, default=0.002
        Confidence value for detecting change. Smaller values make detection more sensitive.
    clock : int, default=32
        Number of instances between subsequent drift checks
    max_buckets : int, default=5
        Maximum number of buckets to keep in the drift detector
    min_window_length : int, default=5
        Minimum window length for detecting drift
    grace_period : int, default=10
        Number of instances before starting drift detection
        
    Returns:
    --------
    list : Índices donde se detectaron changepoints
    """
    detector = ADWIN(
        delta=delta,
        clock=clock,
        max_buckets=max_buckets,
        min_window_length=min_window_length,
        grace_period=grace_period
    )
    
    changepoints = []
    for i, value in enumerate(series):
        if pd.isna(value):
            break  # Detener el procesamiento si se encuentra un NaN
        detector.update(value)
        if detector.drift_detected:
            changepoints.append(i)
    
    return changepoints
