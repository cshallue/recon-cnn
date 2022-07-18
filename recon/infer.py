import os

import asdf
from absl import app, flags, logging

from recon import checkpointing, configuration, inference
from recon.datasets import datasets
from recon.recon_model import ReconModel

flags.DEFINE_string("data_dir", "/mnt/marvin2/cshallue/reconstruction/data",
                    "Base directory containing grids for input.")
flags.DEFINE_string("sims",
                    None,
                    "Comma-separated list of sim names to predict.",
                    required=True)
flags.DEFINE_enum("dataset", "density_field", datasets.dataset_names(),
                  "Name of the dataset to use.")
flags.DEFINE_string(
    "dataset_config", None,
    "Overrides to the dataset configuration the model was trained with.")
flags.DEFINE_string("model_dir",
                    None,
                    "Directory containing model checkpoint.",
                    required=True)
flags.DEFINE_multi_string("ckpt_names", ["target"],
                          "Names of checkpoints to include.")
flags.DEFINE_integer("step", None, "Step of the model checkpoint to use.")
flags.DEFINE_string(
    "output_dir", None,
    "Directory in which to write the output. Defaults to model_dir.")
flags.DEFINE_bool("overwrite", False,
                  "Whether to overwrite files in --output_dir.")
flags.DEFINE_integer("box_size", None,
                     "Box size to use. Defaults to the eval_box_size.")

FLAGS = flags.FLAGS


def main(unused_argv):
    # Load the model config.
    config = checkpointing.load_config(FLAGS.model_dir)
    config.lock()

    model = ReconModel(config.model)
    ckpt = checkpointing.load_checkpoint(FLAGS.model_dir, FLAGS.step)

    configuration.update_from_string(config.dataset, FLAGS.dataset_config)

    box_size = FLAGS.box_size or config.training.eval_box_size
    predict_grid = inference.make_predict_fn(model, box_size)

    # Load the dataset.
    dataset = datasets.create_dataset(FLAGS.dataset, config.dataset,
                                      FLAGS.sims)

    # Make output directory.
    output_dir = FLAGS.output_dir or FLAGS.model_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Make and save predictions.
    for i, grid_name in enumerate(dataset.names):
        logging.info(f"Generating predictions for {grid_name}")
        grid_prefix = f"{grid_name}_z{config.dataset.redshift}"
        if FLAGS.step is not None:
            grid_prefix += f"_{FLAGS.step}"

        example = None  # We'll only load it if we have to.
        for ckpt_name in FLAGS.ckpt_names:
            logging.info(f"Using checkpoint {ckpt_name}")
            ckpt_prefix = grid_prefix
            if ckpt_name != "target":
                ckpt_prefix += f"_{ckpt_name}"
            filename = os.path.join(output_dir,
                                    f"{ckpt_prefix}_predicted_ic.asdf")
            if os.path.exists(filename) and not FLAGS.overwrite:
                logging.info(f"File already exists: {filename}. Skipping.")
                continue

            if example is None:
                example = dataset[i]
            params = ckpt[ckpt_name]
            predicted_grid = predict_grid(params, example.input)
            predicted_grid /= example.metadata.get("target_rescale", 1.0)

            with asdf.AsdfFile({"data": predicted_grid}) as af:
                af.write_to(filename)
            logging.info(f"Saved prediction to {filename}")


if __name__ == "__main__":
    app.run(main)
