import jax
import numpy as np
from jax import lax
from jax import numpy as jnp


def gaussian(x, sigma):
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(x**2 / (-2 * sigma**2))


def make_gaussian_kernel(n, sigma, dx=0.001):
    assert n % 2 == 1  # Make sure n is odd

    # Compute gaussian on a symmetric grid
    x = np.arange((-n + dx) / 2, n / 2, dx)
    y = gaussian(x, sigma)

    # Integrate the gaussian over each cell
    # x = x.reshape((n, -1))
    # xint = np.median(x, axis=-1)
    y = y.reshape((n, -1))
    yint = np.trapz(y, dx=dx, axis=-1)

    # Make sure the kernel integrates to 1. It would anyway if n >> sigma.
    yint /= np.sum(yint)

    return yint


def compute_radial_distance_grid(rmax, ndim):
    n = 2 * rmax + 1

    # Compute the midpoint of each bin in each dimension.
    midpoints = np.arange(-rmax, rmax + 1)
    assert len(midpoints) == n

    # Compute the squared Euclidean distance to every bin midpoint.
    midsq = midpoints**2
    dsq = np.zeros((n, ) * ndim)
    for d in range(ndim):
        reshape = [1] * ndim
        reshape[d] = n
        dsq += midsq.reshape(reshape)

    return np.sqrt(dsq)


def make_spherical_top_hat(rmax, ndim, normalize=True):
    grid = compute_radial_distance_grid(rmax, ndim)
    np.less_equal(grid, rmax, out=grid)
    if normalize:
        grid /= np.sum(grid)
    return grid


def conv3d(grid, kernel, padding="SAME"):
    assert grid.ndim == 3
    assert kernel.ndim == 3
    # Put "batch" and "input feature" dimensions first.
    grid = jnp.expand_dims(grid, axis=(0, 1))
    kernel = jnp.expand_dims(kernel, axis=(0, 1))
    # Do the convolution.
    grid = lax.conv_general_dilated(grid,
                                    kernel,
                                    window_strides=(1, 1, 1),
                                    padding=padding)

    return jnp.squeeze(grid)


def conv3d_separable(grid, kernels, padding="SAME"):
    ndim = grid.ndim
    assert len(kernels) == ndim
    # Do ndim separate convolutions, aligning the kernel with each of the
    # spatial dimensions in turn.
    for i, kernel in enumerate(kernels):
        n, = kernel.shape
        shape_3d = np.ones(ndim, dtype=int)
        shape_3d[i] = n
        grid = conv3d(grid, kernel.reshape(shape_3d), padding)

    return jnp.squeeze(grid)
