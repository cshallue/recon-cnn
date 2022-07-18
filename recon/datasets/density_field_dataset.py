import dataclasses
import glob
import os

import asdf
import numpy as np
from absl import logging
from ml_collections import ConfigDict
from recon.datasets.base import Dataset, GridExample

DATA_BASE_DIR = "/mnt/marvin2/cshallue/reconstruction/data/"


def default_config():
    config = ConfigDict({
        "redshift": "0.500",
        "data_type": "all_A",
        "delta_name": "delta-reconstructed",
        "ic_name": "ic_dens_N576",
        "delta_rescale": 1 / 1.21,
        "ic_rescale": 1 / 0.023,
        "smoothing_sigmas": [],
        "gradient_sigmas": [],
        "shear_sigmas": [],
        "smoothing_rescale": [],
        "gradient_rescale": [],
        "shear_rescale": [],
    })
    config.lock()
    return config


@dataclasses.dataclass
class DensityFieldExampleSpec:
    name: str
    input_filename: str
    target_filename: str


def read_grid(filename, rescale, with_header=False, dtype=np.float32):
    with asdf.open(filename) as af:
        data = af.tree["data"]
        if isinstance(data, dict):
            # Support IC files.
            data = data["density"]
        grid = np.array(data, dtype=dtype, copy=True)
        header = af.tree["header"]
    grid *= rescale
    logging.info(f"Read {os.path.basename(filename)}. Shape: {grid.shape}. "
                 f"Mean: {grid.mean():.4e}. Std: {grid.std():.4e}.")
    return (grid, header) if with_header else grid


def find_examples(sim_patterns, delta_basename, ic_basename):
    sim_names = []
    for sim_pattern in sim_patterns.split(","):
        logging.info(f"Finding grids matching pattern '{sim_pattern}'")
        file_pattern = os.path.join(DATA_BASE_DIR, sim_pattern)
        data_dirs = glob.glob(file_pattern)
        if not data_dirs:
            raise ValueError(f"No files matching {file_pattern}")
        sim_names.extend([os.path.basename(d) for d in data_dirs])
    examples = []
    for sim_name in sim_names:
        delta_filename = os.path.join(DATA_BASE_DIR, sim_name, delta_basename)
        ic_filename = os.path.join(DATA_BASE_DIR, sim_name, ic_basename)
        for filename in (delta_filename, ic_filename):
            if not os.path.isfile(filename):
                raise ValueError(f"File does not exist: {filename}")
        example = DensityFieldExampleSpec(sim_name, delta_filename,
                                          ic_filename)
        logging.info(example)
        examples.append(example)

    return examples


class DensityFieldDataset(Dataset):
    def __init__(self, example_specs, delta_rescale, ic_rescale):
        self.example_specs = example_specs
        self.delta_rescale = delta_rescale
        self.ic_rescale = ic_rescale

    @property
    def names(self):
        return [spec.name for spec in self.example_specs]

    @property
    def nchannels(self):
        return 1

    def __len__(self):
        return len(self.example_specs)

    def __getitem__(self, i):
        return self.load_example(i)

    def load_example(self, i):
        spec = self.example_specs[i]
        logging.info(f"Reading grids for {spec.name}")
        input_grid, header = read_grid(spec.input_filename,
                                       self.delta_rescale,
                                       with_header=True)
        target_grid = read_grid(spec.target_filename, self.ic_rescale)
        metadata = {
            "header": header,
            "input_filename": spec.input_filename,
            "input_rescale": self.delta_rescale,
            "target_filename": spec.target_filename,
            "target_rescale": self.ic_rescale,
        }
        return GridExample(spec.name, input_grid, target_grid, metadata)


def create_dataset(config, sims):
    delta_basename = os.path.join(f"z{config.redshift}", config.data_type,
                                  f"{config.delta_name}.asdf")
    ic_basename = f"{config.ic_name}.asdf"
    specs = find_examples(sims, delta_basename, ic_basename)
    return DensityFieldDataset(specs, config.delta_rescale, config.ic_rescale)


def _validate_datasets(train_dataset, eval_dataset):
    # Ensure datasets are disjoint.
    train_names = set(s.name for s in train_dataset.example_specs)
    eval_names = set(s.name for s in eval_dataset.example_specs)
    intersection = train_names.intersection(eval_names)
    if intersection:
        raise ValueError(
            f"Training grids and evaluation grids overlap: {intersection}")


def create_train_and_eval_datasets(config, train_sims, eval_sims):
    train_dataset = create_dataset(config, train_sims)
    eval_dataset = create_dataset(config, eval_sims)
    _validate_datasets(train_dataset, eval_dataset)
    return train_dataset, eval_dataset
