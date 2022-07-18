import os
import shutil

from absl import app, flags, logging
from ml_collections import ConfigDict

from recon import checkpointing, configuration, trainer
from recon.datasets import datasets
from recon.recon_model import ReconModel

flags.DEFINE_string("model_spec", "2,2,1", "Specification of model layers.")
flags.DEFINE_enum("dataset", "density_field", datasets.dataset_names(),
                  "Name of the dataset to use.")
flags.DEFINE_string("config", None, "Overrides to the base configuration.")
flags.DEFINE_string("model_dir",
                    None,
                    "Directory in which to save the model and training logs.",
                    required=True)
flags.DEFINE_bool("overwrite", False,
                  "Whether to overwrite files in the existing --model_dir.")

FLAGS = flags.FLAGS


def main(unused_argv):
    if os.path.exists(FLAGS.model_dir):
        if FLAGS.overwrite:
            logging.info(f"Removing existing model dir: {FLAGS.model_dir}")
            shutil.rmtree(FLAGS.model_dir)
        else:
            raise ValueError("--model_dir exists and --overwrite=False")

    # Stream logs to stderr and to disk.
    log_dir = os.path.join(FLAGS.model_dir, "logs")
    os.makedirs(log_dir)
    logging.get_absl_handler().use_absl_log_file(log_dir=log_dir)
    FLAGS.alsologtostderr = True

    # Gather all configurations into a single ConfigDict.
    config = ConfigDict()
    config["dataset"] = datasets.get_config(FLAGS.dataset)
    config["model"] = configuration.get_model_config(FLAGS.model_spec)
    config["training"] = configuration.get_training_config()
    config.lock()

    # Apply configuration overrides.
    configuration.update_from_string(config, FLAGS.config)

    # Save the config.
    checkpointing.save_config(FLAGS.model_dir, config)

    # Create the model.
    model = ReconModel(config.model)

    # Create the train and eval datasets.
    train_dataset, eval_dataset = datasets.create_train_and_eval_datasets(
        FLAGS.dataset, config.dataset, config.training.train_sims,
        config.training.eval_sims)

    trainer.run(config=config.training,
                model=model,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                model_dir=FLAGS.model_dir)


if __name__ == "__main__":
    app.run(main)
