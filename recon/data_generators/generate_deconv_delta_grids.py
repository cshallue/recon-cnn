import os
import shutil

import asdf
import numpy as np
import pyfftw
from absl import app, flags, logging
from fftcorr.grid import ConfigSpaceGrid

flags.DEFINE_string("grid_dir",
                    None,
                    "Directory of the input and output grid.",
                    required=True)
flags.DEFINE_enum("grid_name",
                  None, ["delta", "delta-reconstructed"],
                  "Name of the grid to process.",
                  required=True)
flags.DEFINE_integer("output_ngrid",
                     None,
                     "Size of output grid.",
                     required=True)
flags.DEFINE_bool(
    "overwrite", False,
    "Whether to overwrite files in an existing output directory.")

FLAGS = flags.FLAGS


def setup_fft_grid(ngrid):
    grid_shape = [ngrid, ngrid, 2 * (ngrid // 2 + 1)]
    logging.info("Creating FFT grid with shape %s", grid_shape)
    fftgrid = pyfftw.empty_aligned(grid_shape, dtype="float64")
    rgrid = fftgrid[:, :, :ngrid]
    cgrid = fftgrid.view("complex128")
    return rgrid, cgrid


def main(unused_argv):
    FLAGS.alsologtostderr = True
    output_ngrid = FLAGS.output_ngrid
    assert output_ngrid % 2 == 0

    grid_filename = os.path.join(FLAGS.grid_dir, f"{FLAGS.grid_name}.asdf")
    logging.info("Loading density field from %s", grid_filename)
    input_grid = ConfigSpaceGrid.read(grid_filename)
    xmin = input_grid.posmin[0]
    xmax = input_grid.posmax[0]
    input_ngrid = input_grid.shape[0]
    assert np.all(np.equal(input_grid.shape, input_ngrid))
    assert input_ngrid % 2 == 0
    assert output_ngrid <= input_ngrid

    output_basename = (
        f"{FLAGS.grid_name}_deconv_{input_ngrid}_{output_ngrid}")
    output_filename = os.path.join(FLAGS.grid_dir, f"{output_basename}.asdf")
    if os.path.exists(output_filename):
        if FLAGS.overwrite:
            logging.info("Existing file will be overwritten: %s",
                         output_filename)
            os.remove(output_filename)
        else:
            logging.fatal("File exists and --overwrite is false: %s",
                          output_filename)

    # Stream logs to disk.
    log_dir = os.path.join(FLAGS.grid_dir, f"logs-{output_basename}")
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir)
    logging.get_absl_handler().use_absl_log_file(log_dir=log_dir)

    # Setup the FFT grid and forward FFT.
    fft_rgrid, fft_cgrid = setup_fft_grid(input_ngrid)
    execute_fft = pyfftw.FFTW(fft_rgrid,
                              fft_cgrid,
                              axes=tuple(range(3)),
                              direction="FFTW_FORWARD")

    # FFT Forward.
    logging.info("Performing FFT")
    np.copyto(fft_rgrid, input_grid.data)
    execute_fft()

    # Set up the FFT subgrid and IFFT.
    ifft_rgrid, ifft_cgrid = setup_fft_grid(output_ngrid)
    execute_ifft = pyfftw.FFTW(ifft_cgrid,
                               ifft_rgrid,
                               axes=tuple(range(3)),
                               direction="FFTW_BACKWARD")

    # Extract output.
    logging.info("Extracting output subgrid")
    n = output_ngrid // 2
    ifft_cgrid[:n, :n, :] = fft_cgrid[:n, :n, :n + 1]
    ifft_cgrid[:n, n:, :] = fft_cgrid[:n, -n:, :n + 1]
    ifft_cgrid[n:, :n, :] = fft_cgrid[-n:, :n, :n + 1]
    ifft_cgrid[n:, n:, :] = fft_cgrid[-n:, -n:, :n + 1]

    # Perform window correction.
    # We must be careful about the different cell sizes we want.
    assert input_grid.metadata.get("window_type", 1) == 1
    logging.info("Performing TSC window correction")
    cell_size = (xmax - xmin) / output_ngrid
    _fx = np.fft.fftfreq(output_ngrid, cell_size)
    _fy = np.fft.fftfreq(output_ngrid, cell_size)
    _fz = np.fft.rfftfreq(output_ngrid, cell_size)  # Note rfft.
    freq = np.stack(np.meshgrid(_fx, _fy, _fz, indexing="ij"), axis=-1)

    # k = 2 * pi * freq and window = sinc(k * cell_size / 2)
    # The 2's cancel and the pi cancels with the numpy sinc convention.
    delta_cell_size = (xmax - xmin) / input_ngrid
    window = np.prod(np.sinc(freq * delta_cell_size), axis=-1)**3
    ifft_cgrid /= window

    # Perform IFFT.
    logging.info("Performing IFFT")
    execute_ifft()

    # Rescale for the different grid size.
    ifft_rgrid *= (output_ngrid / input_ngrid)**3

    # Save the output.
    logging.info("Saving IFFT grid to %s", output_filename)
    header = dict(
        input_grid=grid_filename,
        input_ngrid=input_ngrid,
        output_ngrid=output_ngrid,
        posmin=input_grid.posmin,
        posmax=input_grid.posmax,
    )
    header.update(input_grid.metadata)
    tree = {
        "header": header,
        "data": np.array(ifft_rgrid, dtype=np.float32),
    }
    with asdf.AsdfFile(tree) as af:
        af.write_to(output_filename)


if __name__ == '__main__':
    app.run(main)
