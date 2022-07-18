from recon.datasets import (ExampleSmoothingFn, PostprocessDataset,
                            density_field_dataset, fake_dataset)

_ALL_DATASETS = {"fake": fake_dataset, "density_field": density_field_dataset}


def dataset_names():
    return sorted(_ALL_DATASETS.keys())


def get_module(name):
    return _ALL_DATASETS[name]


def get_config(name):
    return get_module(name).default_config()


def _postprocess(config, *datasets):
    if (config.smoothing_sigmas or config.gradient_sigmas
            or config.shear_sigmas):
        smoothing_fn = ExampleSmoothingFn(
            smoothing_sigmas=config.smoothing_sigmas,
            gradient_sigmas=config.gradient_sigmas,
            shear_sigmas=config.shear_sigmas,
            smoothing_rescale=config.smoothing_rescale,
            gradient_rescale=config.gradient_rescale,
            shear_rescale=config.shear_rescale)
        datasets = [
            PostprocessDataset(d, smoothing_fn, smoothing_fn.noutput)
            for d in datasets
        ]
    return tuple(datasets) if len(datasets) > 1 else datasets[0]


def create_dataset(name, config, pattern):
    dataset = get_module(name).create_dataset(config, pattern)
    return _postprocess(config, dataset)


def create_train_and_eval_datasets(name,
                                   config,
                                   train_pattern=None,
                                   eval_pattern=None):
    train, eval = get_module(name).create_train_and_eval_datasets(
        config, train_pattern, eval_pattern)
    return _postprocess(config, train, eval)
