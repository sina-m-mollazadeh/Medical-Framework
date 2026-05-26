"""
Deployment driver.

Loads a cloudpickled `DeployablePipeline` bundle and runs it on a raw,
unseen CSV. The bundle carries its own preprocessing (column sanitisation,
date expansion, categorical mapping, imputation, outlier taming,
normalisation, feature selection, classifier, threshold), so the only
thing this script does is read the CSV and forward it to the pipeline.

Usage
-----
    python driver.py path/to/test.csv               # uses ./pipeline.pkl
    python driver.py path/to/test.csv path/bundle.pkl
"""
from __future__ import annotations

import sys
import warnings

import cloudpickle
import pandas as pd

warnings.filterwarnings('ignore')


def load_bundle(pkl_path: str):
    with open(pkl_path, 'rb') as f:
        return cloudpickle.load(f)


def main(csv_path: str, pkl_path: str = 'pipeline.pkl') -> None:
    pipe = load_bundle(pkl_path)
    raw = pd.read_csv(csv_path, sep=None, engine='python',
                       na_values=[' ', '', 'NA', 'NaN'])

    labels = pipe.predict_labels(raw)
    proba = pipe.predict_proba(raw)
    confidences = proba.max(axis=1)

    print(pipe.describe())
    print()
    for i, (lbl, conf) in enumerate(zip(labels, confidences), start=1):
        print(f'Sample {i:>4d} -> Prediction: {lbl}   Confidence: {conf:.2%}')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python driver.py <test.csv> [pipeline.pkl]')
        sys.exit(1)
    csv = sys.argv[1]
    pkl = sys.argv[2] if len(sys.argv) > 2 else 'pipeline.pkl'
    main(csv, pkl)
