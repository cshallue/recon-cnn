import dataclasses
from typing import Callable, Sequence, Union

import numpy as np
from flax import linen as nn
from jax import numpy as jnp


@dataclasses.dataclass
class LayerSpec:
    num_filters: int
    dilation: int = 1
    kernel_size: int = 3


def parse_layer_specs(layer_specs):
    layers = []
    for spec_str in layer_specs.split(","):
        args = [int(a) for a in spec_str.split(":")]
        layers.append(LayerSpec(*args))
    return layers


def compute_receptive_radius(layers):
    return np.sum([l.dilation * (l.kernel_size - 1) // 2 for l in layers])


def _linear_activation_fn(input):
    return input


def _get_activation_fn(name):
    if name == "linear":
        return _linear_activation_fn
    if name == "relu":
        return nn.relu
    if name == "tanh":
        return nn.tanh

    raise ValueError(f"Unrecognized activation function: '{name}'")


class ReconModule(nn.Module):
    layers: Sequence[LayerSpec]
    activation_fn: Callable = nn.relu
    residual_block_size: int = 0
    dtype: Union[jnp.float16, jnp.float32] = jnp.float32

    @nn.compact
    def __call__(self, input_grid):
        input_grid = jnp.asarray(input_grid, dtype=self.dtype)
        if len(input_grid.shape) == 3:
            # Add channels dimension.
            input_grid = jnp.expand_dims(input_grid, axis=-1)

        if len(input_grid.shape) != 4:
            raise ValueError(
                f"Expected 3D or 4D input grid. Got shape: {input_grid.shape}")

        x = input_grid
        skip = None
        block_size = self.residual_block_size
        for i, layer_spec in enumerate(self.layers):
            x = nn.Conv(features=layer_spec.num_filters,
                        kernel_size=[layer_spec.kernel_size] * 3,
                        padding="VALID",
                        kernel_dilation=[layer_spec.dilation] * 3,
                        use_bias=True,
                        dtype=self.dtype)(x)
            if block_size > 0 and i % block_size == 0:
                if skip is not None:
                    # We need to account for cells lost due to valid padding.
                    shape_diff = np.array(skip.shape) - np.array(x.shape)
                    r1, r2, r3 = shape_diff[:3] // 2
                    skip_slice = skip[r1:-r1, r2:-r2, r3:-r3, :]
                    # We're only supporting skip connections between layers with
                    # the same number of features.
                    if skip_slice.shape == x.shape:
                        x += skip_slice
                skip = x
            if i < len(self.layers) - 1:
                x = self.activation_fn(x)
        x = jnp.squeeze(x)
        if len(x.shape) != 3:
            raise ValueError(f"Expected 3D output grid. Got shape: {x.shape}")
        return x


class ReconModel:
    def __init__(self, config):
        # Create the flax module.
        args = config.to_dict()
        layers = [LayerSpec(**l) for l in args.pop("layers")]
        activation_fn = _get_activation_fn(args.pop("activation_fn"))
        module = ReconModule(layers, activation_fn, **args)

        self._config = config
        self._layers = layers
        self._receptive_radius = compute_receptive_radius(layers)
        self._module = module

    @property
    def config(self):
        return self._config

    @property
    def layers(self):
        return self._layers

    @property
    def receptive_radius(self):
        return self._receptive_radius

    @property
    def module(self):
        return self._module

    def initialize_module(self, rng_key, input_features=1):
        dummy_input = jnp.zeros([1, 1, 1, input_features])
        return self.module.init(rng_key, dummy_input)
