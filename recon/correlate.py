import collections
import os

import asdf
import numpy as np
from absl import app, flags, logging
from astropy.table import Table
from fftcorr.correlate import PeriodicCorrelator

GRID_SHAPE = [576] * 3
POSMIN = -1000
POSMAX = 1000

flags.DEFINE_string("grid_filename",
                    None,
                    "File patterns matching grids to correlate.",
                    required=True)
flags.DEFINE_string("output_dir", None,
                    "Output directory. Defaults to the grid directory.")
flags.DEFINE_string("reference_grid", None,
                    "Path of reference grid to cross correlate.")
flags.DEFINE_string("reference_grid_name", None, "Name of the reference grid.")
flags.DEFINE_bool("overwrite", False,
                  "Whether to overwrite previous correlation file.")
flags.DEFINE_integer("window_correct", 0,
                     "Window type for power spectrum correction.")
flags.DEFINE_float("kmax", 0.9, "Maximum k for power spectrum.")
flags.DEFINE_float("dk", 0.005, "Width of k bins for power spectrum.")
flags.DEFINE_float("rmax", 150, "Maximum r for correlations.")
flags.DEFINE_float("dr", 5, "Width of r bins for correlations.")
flags.DEFINE_integer("maxell", 2, "Maximum multipole moment.")

FLAGS = flags.FLAGS


def load_grid(filename):
    logging.info(f"Reading {filename}")
    with asdf.open(filename) as af:
        if hasattr(af.tree["data"], "shape"):
            return np.array(af.tree["data"], dtype=np.float64)
        return np.array(af.tree["data"]["density"], dtype=np.float64)


def main(unused_argv):
    grid_dir, grid_basename = os.path.split(FLAGS.grid_filename)
    output_dir = FLAGS.output_dir or grid_dir

    if grid_basename.endswith("_predicted_ic.asdf"):
        grid_name = grid_basename[:-18]
        sim_name = grid_name[:28]
    elif grid_basename == "ic_dens_N576.asdf":
        grid_name = "ic_dens_N576"
        sim_name = os.path.basename(grid_dir)
    elif "delta" in grid_basename:
        grid_name = grid_basename[:-5]
        sim_name = FLAGS.grid_filename.split("/")[-4]
    else:
        raise ValueError(f"Unexpected grid filename: {grid_basename}")
    assert sim_name.startswith("AbacusSummit_")

    cross_spectra_filename = os.path.join(output_dir,
                                          grid_name + "_cross_spectra.ecsv")
    correlations_filename = os.path.join(output_dir,
                                         grid_name + "_correlations.ecsv")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if (os.path.exists(cross_spectra_filename)
            or os.path.exists(correlations_filename)):
        if not FLAGS.overwrite:
            logging.warning(f"Files already exist and --overwrite is false.")
            return
        logging.warning("Files will be overwritten.")

    c = PeriodicCorrelator(GRID_SHAPE,
                           cell_size=(POSMAX - POSMIN) / GRID_SHAPE[0],
                           window_correct=FLAGS.window_correct,
                           rmax=FLAGS.rmax,
                           dr=FLAGS.dr,
                           kmax=FLAGS.kmax,
                           dk=FLAGS.dk,
                           maxell=FLAGS.maxell)

    # Load grid.
    pred_grid = load_grid(FLAGS.grid_filename)

    # Load reference grid.
    ref_grid = None
    ref_grid_name = None
    if FLAGS.reference_grid:
        ref_grid = load_grid(FLAGS.reference_grid)
        ref_grid_name = FLAGS.reference_grid_name or "initial_conditions"

    pow_columns = collections.OrderedDict()
    corr_columns = collections.OrderedDict()

    # Autocorrelate grid.
    logging.info("Autocorrelating grid")
    spectrum, corr = c.autocorrelate(pred_grid)
    pow_columns["k"] = spectrum["k"]
    pow_columns["count"] = spectrum["count"]
    corr_columns["r"] = corr["r"]
    corr_columns["count"] = corr["count"]
    colname = f"{grid_name}_X_{grid_name}"
    pow_columns[colname] = spectrum["ps"]
    corr_columns[colname] = corr["xi"]

    if ref_grid is not None:
        # Autocorrelate reference grid.
        logging.info("Autocorrelating reference grid")
        spectrum, corr = c.autocorrelate(ref_grid)
        colname = f"{ref_grid_name}_X_{ref_grid_name}"
        pow_columns[colname] = spectrum["ps"]
        corr_columns[colname] = corr["xi"]

        # Cross correlate grid with IC grid.
        logging.info("Cross correlating grid with reference grid")
        spectrum, corr = c.cross_correlate(pred_grid, ref_grid)
        colname = f"{grid_name}_X_{ref_grid_name}"
        pow_columns[colname] = spectrum["ps"]
        corr_columns[colname] = corr["xi"]

    meta = dict(grid_shape=GRID_SHAPE,
                cell_size=(POSMAX - POSMIN) / GRID_SHAPE[0],
                window_correct=FLAGS.window_correct,
                rmax=FLAGS.rmax,
                dr=FLAGS.dr,
                kmax=FLAGS.kmax,
                dk=FLAGS.dk,
                maxell=FLAGS.maxell)
    for columns, filename in [(pow_columns, cross_spectra_filename),
                              (corr_columns, correlations_filename)]:
        t = Table(list(columns.values()),
                  names=list(columns.keys()),
                  meta=meta)
        t.write(filename, format="ascii.ecsv", overwrite=FLAGS.overwrite)
        logging.info(f"Wrote {filename}")


if __name__ == "__main__":
    app.run(main)
