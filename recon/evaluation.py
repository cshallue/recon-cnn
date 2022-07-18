import collections
import os

import numpy as np
import pandas as pd
from absl import logging

from recon import inference

METRICS_FILENAME = "metrics.csv"


def _metrics_filename(model_dir):
    return os.path.join(model_dir, METRICS_FILENAME)


def evaluate_dataset(predict_fn, dataset, verbose=True):
    output = collections.OrderedDict()
    for example in dataset:
        predicted_grid = predict_fn(example.input)
        mse = np.mean((example.target - predicted_grid)**2)
        if verbose:
            logging.info(f"Evaluated {example.name}. MSE: {mse:.2e}.")
        output[example.name] = mse
    return output


def write_metrics(metrics, step, model_dir, verbose=True):
    filename = _metrics_filename(model_dir)
    if os.path.exists(filename):
        df = pd.read_csv(filename, index_col="step")
    else:
        df = pd.DataFrame(columns=metrics.keys())
    df = df.append(
        pd.DataFrame([metrics.values()],
                     columns=metrics.keys(),
                     index=(step, )))
    df.index.name = "step"
    df.to_csv(filename)
    if verbose:
        logging.info(f"Saved metrics to {filename}")


def load_metrics(model_dir, verbose=True):
    return pd.read_csv(_metrics_filename(model_dir), index_col="step")
