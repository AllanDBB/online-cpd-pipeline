from river.drift import PageHinkley
import pandas as pd

def detect_changepoints_page_hinkley(series, threshold=50, min_instances=30, delta=0.005, reset_on_change=True):
	detector = PageHinkley(threshold=threshold, min_instances=min_instances, delta=delta)
	changepoints = []
	for i, value in enumerate(series):
		if pd.isna(value):
			break  # Detener el procesamiento si se encuentra un NaN
		detector.update(value)
		if detector.drift_detected:
			changepoints.append(i)
	return changepoints

