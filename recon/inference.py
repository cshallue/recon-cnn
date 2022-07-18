import jax
import numpy as np

from recon.datasets import iterators


def make_predict_fn(model, box_size):
    @jax.jit
    def predict_box(params, inputs):
        return model.module.apply(params, inputs)

    def predict_grid(params, input_grid):
        params = jax.device_put(params)
        output_grid = np.zeros(input_grid.shape[:3], dtype=input_grid.dtype)

        n_output = 0
        for inputs, outputs in iterators.example_partition_iterator(
                input_grid, output_grid, box_size, model.receptive_radius):
            np.copyto(outputs, predict_box(params, inputs))
            n_output += outputs.size

        assert n_output == output_grid.size

        return output_grid

    return predict_grid


def predict_grid(model, params, input_grid, box_size):
    predict_fn = make_predict_fn(model, box_size)
    return predict_fn(params, input_grid)
