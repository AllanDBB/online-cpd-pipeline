import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.allan_synthetic import generate_dataset

D = generate_dataset(
    n_series_per_level=6,
    seed=123,
    n_points_choices=[50, 60],
    n_changes_choices=[1, 2],
    tipo_cambio='escalon',
)

print({k: (v['series'].shape, len(v['changepoints'])) for k, v in D.items()})
