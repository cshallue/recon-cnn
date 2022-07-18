from jax import numpy as jnp


def mean_squared_error(predictions, targets):
    return jnp.mean((targets - predictions)**2)
