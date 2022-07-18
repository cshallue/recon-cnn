import glob
import os
from functools import partial

import asdf
import numpy as np
import pyfftw
from absl import logging
from fftcorr.catalog import add_random_particles, read_density_field
from fftcorr.grid import ConfigSpaceGrid
from fftcorr.utils import Timer
from ml_collections import config_dict

_DEFAULT_CONFIG = config_dict.create(ngrid=576,
                                     xmin=-1000,
                                     xmax=1000,
                                     window_type=1,
                                     redshift_distortion=False,
                                     f_growth=0.0,
                                     gaussian_sigma=10,
                                     nrandom=int(1e9),
                                     bias=1.0,
                                     rng_seed=0)
_DEFAULT_CONFIG.lock()  # Prevent new fields from being added.


def default_config():
    return _DEFAULT_CONFIG.copy_and_resolve_references()


def compute_displacement_field(delta, cell_size, sigma, f_growth=None):
    logging.info("Computing displacement field")
    # indexing="ij" means the array are indexed A[xi][yi][zi]
    k = np.meshgrid(
        2 * np.pi * np.fft.fftfreq(delta.shape[0], cell_size),
        2 * np.pi * np.fft.fftfreq(delta.shape[1], cell_size),
        2 * np.pi * np.fft.rfftfreq(delta.shape[2], cell_size),  # Note rfft.
        indexing="ij")

    # Setup the FFTs.
    rshape = np.array(delta.shape)
    cshape = rshape.copy()
    cshape[-1] = (cshape[-1] // 2) + 1
    r_fftgrid = pyfftw.empty_aligned(rshape, dtype="float64")
    c_fftgrid = pyfftw.empty_aligned(cshape, dtype="complex128")
    fft = pyfftw.FFTW(r_fftgrid,
                      c_fftgrid,
                      axes=tuple(range(3)),
                      direction="FFTW_FORWARD")
    inv_fft = pyfftw.FFTW(c_fftgrid,
                          r_fftgrid,
                          axes=tuple(range(3)),
                          direction="FFTW_BACKWARD")

    # Convolve with Gaussian in Fourier space.
    logging.info(f"Using Gaussian smoothing with sigma = {sigma}")
    kgaussian = np.exp((-sigma**2 / 2) * (k[0]**2 + k[1]**2 + k[2]**2))
    np.copyto(r_fftgrid, delta)
    fft()
    kconv = c_fftgrid * kgaussian

    # Set all frequencies on the boundary (Nyquist frequency) to zero.
    assert np.all(np.array(delta.shape) % 2 == 0)
    kconv[delta.shape[0] // 2, :, :] = 0
    kconv[:, delta.shape[1] // 2, :] = 0
    kconv[:, :, delta.shape[2] // 2] = 0

    # Multiply kconv by i / k^2.
    if f_growth is None:
        f_growth = 0
    with np.errstate(divide='ignore', invalid='ignore'):
        # Ignoring divide by zero at zero.
        kconv *= 1J / (k[0]**2 + k[1]**2 + (1 + f_growth) * k[2]**2)
    # The mean of the displacement field, which we'll assume to be zero.
    kconv[0][0][0] = 0

    # Compute displacement field.
    disp_shape = np.concatenate([delta.shape, (3, )])
    disp = np.empty(disp_shape, dtype=np.float64)
    for i in range(3):
        np.copyto(c_fftgrid, kconv)
        c_fftgrid *= k[i]
        inv_fft()
        disp[:, :, :, i] = r_fftgrid

    return disp


def _summarize_displacement_field(disp):
    for i, name in enumerate(["x", "y", "z"]):
        dslice = disp[:, :, :, i]
        logging.info(
            f"{name} displacement: min: {dslice.min():.2f}, max: "
            f"{dslice.max():.2f}, RMS: {np.sqrt(np.mean(dslice**2)):.2f}")
    logging.info("Mean displacement magnitude: "
                 f"{np.mean(np.sqrt(np.sum(disp**2, axis=-1))):.2f}")


def compute_reconstructed_density_field(config,
                                        file_patterns,
                                        return_delta=False,
                                        reader=None,
                                        output_dir=None,
                                        delta_only=False,
                                        buffer_size=0):

    # Look up f_growth if necessary. It is only used with redshift distortions.
    f_growth = 0.0
    if config.redshift_distortion:
        if config.f_growth <= 0:
            filename = glob.glob(file_patterns[0])[0]
            with asdf.open(filename) as af:
                config.f_growth = af.tree["header"]["f_growth"]
            logging.info(f"Read f_growth = {config.f_growth} from {filename}")
        f_growth = config.f_growth
    logging.info(f"Using f_growth = {f_growth}")

    # Ensure the random seed is explicit in the config for reproducability.
    if not config.rng_seed:
        config.rng_seed = np.random.SeedSequence().entropy

    # Save the config.
    config_json = config.to_json(indent=2)
    logging.info(f"Config:\n{config_json}")
    if output_dir:
        with open(os.path.join(output_dir, "config.json"), "w") as f:
            f.write(config_json)

    # Create input function.
    read_catalog_to_grid = partial(
        read_density_field,
        file_patterns=file_patterns,
        window_type=config.window_type,
        reader=reader,
        periodic_wrap=True,
        redshift_distortion=config.redshift_distortion,
        flip_xz=(buffer_size > 0),
        buffer_size=buffer_size)

    # Allocate the grid.
    shape = [config.ngrid] * 3
    posmin = [config.xmin] * 3
    posmax = [config.xmax] * 3
    delta_grid = ConfigSpaceGrid(shape, posmin=posmin, posmax=posmax)

    # Create density field.
    logging.info("Reading density field")
    read_catalog_to_grid(grid=delta_grid)
    dens_mean = np.mean(delta_grid)
    delta_grid -= dens_mean
    delta_grid /= dens_mean
    if output_dir:
        delta_filename = os.path.join(output_dir, "delta.asdf")
        delta_grid.write(delta_filename)
        logging.info(f"Wrote density field to {delta_filename}")

    if delta_only:
        return delta_grid if return_delta else None

    # Compute displacement field.
    with Timer() as recon_timer:
        if config.bias != 1.0:
            logging.info(f"Dividing delta field by the bias: {config.bias}")
            delta_grid /= config.bias
        disp = compute_displacement_field(delta_grid.data,
                                          delta_grid.cell_size,
                                          config.gaussian_sigma, f_growth)
        logging.info("Computed real-space displacement field")
        _summarize_displacement_field(disp)
        disp = np.ascontiguousarray(disp)
        disp *= -1
    logging.info(
        f"Computing displacement field time: {recon_timer.elapsed:.2f} sec")

    # Create the reconstructed delta grid.
    if return_delta:
        recon_grid = ConfigSpaceGrid(shape, posmin=posmin, posmax=posmax)
    else:
        # We can reuse the delta grid.
        recon_grid = delta_grid
        recon_grid.clear()

    # Add the randoms first because the displacement field for particles has an
    # an additional term in redshift space.
    logging.info("Adding shifted random particles")
    random_weight = -dens_mean * np.prod(shape)
    add_random_particles(config.nrandom,
                         recon_grid,
                         window_type=config.window_type,
                         total_weight=random_weight,
                         periodic_wrap=True,
                         disp=disp,
                         seed=config.rng_seed)

    logging.info("Adding shifted particles")
    if config.redshift_distortion:
        assert f_growth > 0
        disp[:, :, :, 2] *= (1 + f_growth)
        logging.info("Computed redshift-space displacement field")
        _summarize_displacement_field(disp)
    read_catalog_to_grid(grid=recon_grid, disp=disp)

    recon_grid /= dens_mean
    if output_dir:
        recon_filename = os.path.join(output_dir, f"delta-reconstructed.asdf")
        recon_grid.write(recon_filename)
        logging.info(f"Wrote reconstructed density field to {recon_filename}")

    return (delta_grid, recon_grid) if return_delta else recon_grid
