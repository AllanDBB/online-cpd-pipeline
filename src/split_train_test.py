import numpy as np
import random

def split_train_test(data, n_test=3, seed=None):
	"""
	Splits time series and changepoints into calibration (train) and testing (test) sets for each noise level.
	Selects n_test random series for test and the rest for calibration.
	Returns a dictionary:
	{
		'fuerte': {
			'train': {'series': ..., 'changepoints': ...},
			'test': {'series': ..., 'changepoints': ...}
		},
		'medio': { ... },
		'suave': { ... }
	}
	Each test set contains 3 series and their changepoints (total 9 for all levels).
	"""
	if seed is not None:
		np.random.seed(seed)
		random.seed(seed)
	split = {}
	for nivel, info in data.items():
		n_series = info['series'].shape[0]
		indices = np.arange(n_series)
		test_idx = np.random.choice(indices, size=n_test, replace=False)
		train_idx = np.setdiff1d(indices, test_idx)
		split[nivel] = {
			'train': {
				'series': info['series'].iloc[train_idx].reset_index(drop=True),
				'changepoints': [info['changepoints'][i] for i in train_idx]
			},
			'test': {
				'series': info['series'].iloc[test_idx].reset_index(drop=True),
				'changepoints': [info['changepoints'][i] for i in test_idx]
			}
		}
	return split
