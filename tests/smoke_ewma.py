import os, sys, json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from synthetic_data_generator import generate_balanced_dataset
from algorithms.ewma import detect_changepoints_ewma
import numpy as np


def main():
    D = generate_balanced_dataset(
        n_series_per_level=3,
        seed=123,
        n_points_choices=[100],
        n_changes_choices=[2],
        nsr_ranges={
            'fuerte': (1.5, 2.0),
            'medio': (0.6, 0.9),
            'suave': (0.1, 0.2),
        },
    )

    out = {}
    for lvl in ['fuerte','medio','suave']:
        s = D[lvl]['series'].iloc[0].to_numpy()
        s = s[~np.isnan(s)]
        cps = D[lvl]['changepoints'][0]
        det = detect_changepoints_ewma(s, alpha=0.2, threshold=3.0, min_instances=5)
        out[lvl] = {
            'len': int(len(s)),
            'true_cps': cps,
            'detected_sample': det[:5],
        }

    # early-stop por NaN
    s2 = np.concatenate([np.linspace(0,1,50), np.array([np.nan]*10)])
    det2 = detect_changepoints_ewma(s2, alpha=0.2, threshold=3.0, min_instances=5)

    print(json.dumps({'levels': out, 'nan_stop_detect_len': len(det2)}, ensure_ascii=False))


if __name__ == '__main__':
    main()

