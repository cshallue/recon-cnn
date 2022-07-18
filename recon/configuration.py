import dataclasses
from contextlib import suppress

from ml_collections import ConfigDict

from recon.recon_model import parse_layer_specs


def get_model_config(layer_specs):
    layers = parse_layer_specs(layer_specs)
    config = ConfigDict(
        {"layers": [dataclasses.asdict(spec) for spec in layers]})
    config.activation_fn = "relu"
    config.residual_block_size = 0
    config.lock()
    return config


def get_training_config():
    config = ConfigDict()

    config.rng_seed = 12345

    config.train_sims = "AbacusSummit_base_c000_ph00[5-9],AbacusSummit_base_c000_ph01?"
    config.eval_sims = "AbacusSummit_base_c000_ph020"

    config.optimizer = "momentum"
    config.learning_rate = 0.008
    config.apply_cosine_lr_decay = False
    # Applies for optimizer = "momentum"
    config.momentum = 0.9
    # Applies for optimizer = "adam"
    config.adam_beta1 = 0.9
    config.adam_beta2 = 0.999
    config.adam_epsilon = 1e-8

    config.apply_data_augmentation = False

    config.params_ema_step_sizes = []

    config.train_box_size = 64
    config.eval_box_size = 64
    config.steps_per_grid = 10

    config.smoothed_loss_fraction = 0.0
    config.smoothing_kernel_size = 0
    config.smoothing_kernel_sigma = 0.0
    config.smoothing_kernel_sigma_z = -1.0

    config.variance_penalty_coefficient = 0.0
    config.variance_penalty_radius = 0

    config.train_steps = 50
    config.log_frequency = 10
    config.eval_frequency = 100
    config.checkpoint_frequency = 100
    config.keep_checkpoint_max = 1
    config.keep_checkpoint_every_n_steps = 0

    config.lock()
    return config


def _parse_override_value(value):
    if value[0] == "'" and value[-1] == "'":
        return value[1:-1]
    if value[0] == "[" and value[-1] == "]":
        return [_parse_override_value(v) for v in value[1:-1].split(";") if v]
    with suppress(ValueError):
        return int(value)
    with suppress(ValueError):
        return float(value)
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    return value


def _parse_config_overrides(overrides_str):
    overrides = {}
    while overrides_str:
        key, remainder = overrides_str.split("=", 1)
        if remainder.startswith("'"):
            value, overrides_str = remainder[1:].split("'", 1)
            # Remove leading comma (not present if this is the last override).
            if overrides_str.startswith(","):
                overrides_str = overrides_str[1:]
        else:
            split = remainder.split(",", 1)
            value = split[0]
            overrides_str = split[1] if len(split) == 2 else ""
        overrides[key] = _parse_override_value(value)
    return overrides


def update_from_string(config, overrides_str):
    config.update_from_flattened_dict(_parse_config_overrides(overrides_str))
