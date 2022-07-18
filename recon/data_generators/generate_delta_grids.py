import os
import shutil
import struct

from absl import app, flags, logging
from ml_collections.config_flags import config_flags
from recon.data_generators.galaxy_reader import GalaxyReader
from recon.data_generators.standard_reconstruction import (
    compute_reconstructed_density_field, default_config)

flags.DEFINE_string("sim_name",
                    None,
                    "Directory containing the input files",
                    required=True)

flags.DEFINE_string("sim_dir", "/mnt/marvin2/bigsims/AbacusSummit/",
                    "Directory containing simulation files.")
ALLOWED_DATA_TYPES = ["galaxies", "halos", "all_A", "all_B", "all_AB"]
flags.DEFINE_enum("data_type",
                  None,
                  ALLOWED_DATA_TYPES,
                  "Type of data to process",
                  required=True)
flags.DEFINE_string("output_dir",
                    None,
                    "Base directory for the output",
                    required=True)
ALLOWED_REDSHIFTS = [
    "0.100", "0.200", "0.300", "0.500", "0.800", "1.100", "1.400", "1.700",
    "2.000", "2.500", "3.000"
]
flags.DEFINE_enum("z",
                  None,
                  ALLOWED_REDSHIFTS,
                  "Redshift to process",
                  required=True)
flags.DEFINE_bool(
    "output_dir_exist_ok", False,
    "Whether to overwrite files in an existing output directory.")
config_flags.DEFINE_config_dict("config", default_config())
flags.DEFINE_bool(
    "delta_only", False,
    "Whether to only generate the delta grid (not the reconstructed grid).")

flags.DEFINE_integer("buffer_size", 0,
                     "Buffer size for multithreaded mass assignment.")

FLAGS = flags.FLAGS


def get_file_patterns(sim_name, sim_dir, redshift, data_type):
    if data_type == "galaxies":
        return [
            os.path.join(sim_dir, sim_name, f"z{redshift}",
                         "galaxies/galaxies2.asdf")
        ]

    data_dir = os.path.join(sim_dir, sim_name, "halos", f"z{redshift}")
    fps = []
    if data_type == "halos":
        fps.append(os.path.join(data_dir, "halo_info/halo_info_*.asdf"))
    if data_type in ["all_A", "all_AB"]:
        fps.append(os.path.join(data_dir, "field_rv_A/field_rv_A_*.asdf"))
    if data_type in ["all_B", "all_AB"]:
        fps.append(os.path.join(data_dir, "field_rv_B/field_rv_B_*.asdf"))
    if data_type in ["halo_A", "halo_AB", "all_A", "all_AB"]:
        fps.append(os.path.join(data_dir, "halo_rv_A/halo_rv_A_*.asdf"))
    if data_type in ["halo_B", "halo_AB", "all_B", "all_AB"]:
        fps.append(os.path.join(data_dir, "halo_rv_B/halo_rv_B_*.asdf"))
    if not fps:
        raise ValueError(f"Unrecognized data_type: {data_type}")

    return fps


def main(unused_argv):
    FLAGS.alsologtostderr = True
    config = FLAGS.config

    if config.rng_seed is None:
        config.rng_seed = int(struct.unpack("q", os.urandom(8))[0])

    file_patterns = get_file_patterns(FLAGS.sim_name, FLAGS.sim_dir, FLAGS.z,
                                      FLAGS.data_type)

    # Make the output directory.
    output_dir = os.path.join(FLAGS.output_dir, FLAGS.sim_name, f"z{FLAGS.z}",
                              FLAGS.data_type)
    if config.redshift_distortion:
        output_dir += "_rd"
    if config.ngrid != 576:
        output_dir += f"_{config.ngrid}"

    if os.path.exists(output_dir):
        logging.info("Output directory already exists: %s", output_dir)
        if FLAGS.output_dir_exist_ok:
            logging.info("Files may be overwritten")
        else:
            raise ValueError("Output directory already exists and "
                             "--output_dir_exist_ok is False")

    # Stream logs to disk.
    log_dir = os.path.join(output_dir, "logs")
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir)
    logging.get_absl_handler().use_absl_log_file(log_dir=log_dir)

    reader = GalaxyReader() if FLAGS.data_type == "galaxies" else None
    compute_reconstructed_density_field(config,
                                        file_patterns,
                                        reader=reader,
                                        output_dir=output_dir,
                                        buffer_size=FLAGS.buffer_size)


if __name__ == '__main__':
    app.run(main)
