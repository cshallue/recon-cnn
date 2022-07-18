import json
import os

from absl import logging
from flax.training import checkpoints
from ml_collections import ConfigDict

from recon.recon_model import ReconModel

CONFIG_FILENAME = "config.json"


def _config_filename(model_dir):
    return os.path.join(model_dir, CONFIG_FILENAME)


def save_config(model_dir, config, verbose=True):
    config_json = config.to_json(indent=2)
    if verbose:
        logging.info(f"Saving config: {config_json}")
    with open(_config_filename(model_dir), "w") as f:
        f.write(config_json)


def load_config(model_dir, verbose=True):
    with open(_config_filename(model_dir)) as f:
        config_json = f.read()
    if verbose:
        logging.info(f"Loaded config: {config_json}")
    return ConfigDict(json.loads(config_json))


def load_checkpoint(model_dir, step=None):
    return checkpoints.restore_checkpoint(model_dir, target=None, step=step)


def load_model_params(model_dir, step=None):
    ckpt = load_checkpoint(model_dir, step)
    return ckpt["target"]


def load_model(model_dir, step=None, verbose=True):
    config = load_config(model_dir, verbose)
    model = ReconModel(config.model)
    params = load_model_params(model_dir, step)
    return config, model, params
