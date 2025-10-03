import numpy as np
import random
from typing import Any, Dict, List

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


def split_train_test_synthetic(
    dataset: Dict[str, Any], 
    test_size: float = 0.3, 
    seed: int = 42
) -> Dict[str, Dict[str, Any]]:
    """
    Split synthetic dataset into train and test sets.
    
    This function is designed for the modern benchmark pipeline that uses
    lists of numpy arrays instead of pandas DataFrames.
    
    Args:
        dataset: Dictionary with 'series', 'changepoints', 'lengths'
        test_size: Fraction of data to use for test (default 0.3)
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary with 'train' and 'test' keys, each containing series, changepoints, lengths
    """
    np.random.seed(seed)
    n_series = len(dataset["series"])
    n_test = max(1, int(n_series * test_size))
    n_train = n_series - n_test
    
    # Random shuffling
    indices = np.arange(n_series)
    np.random.shuffle(indices)
    
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]
    
    return {
        "train": {
            "series": [dataset["series"][i] for i in train_idx],
            "changepoints": [dataset["changepoints"][i] for i in train_idx],
            "lengths": [dataset["lengths"][i] for i in train_idx],
        },
        "test": {
            "series": [dataset["series"][i] for i in test_idx],
            "changepoints": [dataset["changepoints"][i] for i in test_idx],
            "lengths": [dataset["lengths"][i] for i in test_idx],
        }
    }
