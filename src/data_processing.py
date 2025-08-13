"""
Data processing script for loading time series and changepoints.
"""
import pandas as pd
import os

# Paths to data
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
SERIES_FILES = {
    'fuerte': 'ruidoAlto.csv',
    'medio': 'ruidoMedio.csv',
    'suave': 'ruidoSuave.csv'
}
CHGPOINT_FILES = {
    'fuerte': 'cpsFuerte',
    'medio': 'cpsMedio',
    'suave': 'cpsSuave'
}

def load_series(noise_level):
    """Load time series for a given noise level."""
    path = os.path.join(DATA_DIR, SERIES_FILES[noise_level])
    df = pd.read_csv(path, header=None)
    # First row is time index, rest are series
    time_index = df.iloc[0].values
    series = df.iloc[1:].reset_index(drop=True)
    return time_index, series

def load_changepoints(noise_level):
    """Load changepoints for a given noise level."""
    import pickle
    path = os.path.join(DATA_DIR, CHGPOINT_FILES[noise_level])
    with open(path, 'rb') as f:
        changepoints = pickle.load(f)
    return changepoints

def load_all():
    """Load all series and changepoints."""
    data = {}
    for level in SERIES_FILES:
        time_index, series = load_series(level)
        changepoints = load_changepoints(level)
        data[level] = {
            'time_index': time_index,
            'series': series,
            'changepoints': changepoints
        }
    return data

if __name__ == "__main__":
    # Example usage
    all_data = load_all()
    for level, info in all_data.items():
        print(f"Nivel de ruido: {level}")
        print(f"Series shape: {info['series'].shape}")
        print(f"Ejemplo de changepoints: {info['changepoints'][:3]}")
