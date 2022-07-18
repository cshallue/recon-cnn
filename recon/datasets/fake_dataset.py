import numpy as np
from ml_collections import ConfigDict
from recon.datasets.base import Dataset, GridExample


def default_config():
    config = ConfigDict({
        "grid_size": 64,
        "smoothing_sigmas": [],
        "gradient_sigmas": [],
        "shear_sigmas": [],
        "smoothing_rescale": [],
        "gradient_rescale": [],
        "shear_rescale": [],
    })
    config.lock()
    return config


class FakeDataset(Dataset):
    def __init__(self, grid_size, dtype=np.float32):
        grid_shape = [grid_size] * 3
        self.example = GridExample(name="fake",
                                   input=np.zeros(grid_shape, dtype),
                                   target=np.zeros(grid_shape, dtype),
                                   metadata={
                                       "input_rescale": 1.0,
                                       "target_rescale": 1.0
                                   })

    @property
    def names(self):
        return [self.example.name]

    @property
    def nchannels(self):
        return 1

    def __len__(self):
        return 1

    def __getitem__(self, i):
        return self.example


def create_dataset(config, *args):
    return FakeDataset(config.grid_size)


def create_train_and_eval_datasets(config, *args):
    train_dataset = create_dataset(config)
    eval_dataset = train_dataset
    return train_dataset, eval_dataset
