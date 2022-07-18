from functools import partial
from timeit import default_timer

import jax
import numpy as np
import optax
from absl import logging
from flax.training import checkpoints
from jax import numpy as jnp

from recon import evaluation, losses, smoothing
from recon.datasets import data_augmentation, iterators
from recon.inference import make_predict_fn
from recon.utils import Timer


def _random_int(rng, dtype):
    info = np.iinfo(dtype)
    return rng.integers(info.min, info.max + 1)


def initialize_module(rng, model, nchannels):
    rng_key = jax.random.PRNGKey(_random_int(rng, np.int64))
    initial_params = model.initialize_module(rng_key, nchannels)
    num_params = 0
    for layer_name, layer_params in initial_params["params"].items():
        kernel_shape = layer_params["kernel"].shape
        bias_shape = layer_params["bias"].shape
        logging.info(
            f"{layer_name}: kernel shape: {kernel_shape}, bias shape: "
            f"{bias_shape}")
        num_params += np.prod(kernel_shape) + np.prod(bias_shape)
    logging.info(f"Total parameters: {num_params:,}")
    return initial_params


def create_optimizer(config):
    if config.optimizer == "momentum":
        return optax.sgd(config.learning_rate, config.momentum)
    if config.optimizer == "adam":
        return optax.adam(config.learning_rate, config.adam_beta1,
                          config.adam_beta2, config.adam_epsilon)
    raise ValueError(f"Unrecognized optimizer: {config.optimizer}")


def smooth_grid(grid, kernel):
    return smoothing.conv3d_separable(grid, kernel)


def compute_sigmasq(grid, kernel):
    smoothed_grid = smoothing.conv3d(grid, kernel, padding="VALID")
    return jnp.mean(smoothed_grid**2)


def make_gaussian_kernel(kernel_size, sigma):
    logging.info(f"Smoothing kernel sigma = {sigma:.2e}")
    kernel = smoothing.make_gaussian_kernel(kernel_size, sigma)
    logging.info(f"Smoothing kernel: {kernel}")
    return kernel


def make_smoothing_kernel(kernel_size, sigma_xy, sigma_z=-1.0):
    kernel_xy = make_gaussian_kernel(kernel_size, sigma_xy)
    if sigma_z >= 0:
        kernel_z = make_gaussian_kernel(kernel_size, sigma_z)
    else:
        kernel_z = kernel_xy
    kernels = jnp.stack([kernel_xy, kernel_xy, kernel_z])
    return kernels


def make_variance_penalty_kernel(radius):
    kernel = smoothing.make_spherical_top_hat(radius, ndim=3)
    kernel = jax.device_put(kernel)
    return kernel


def make_loss_fn(model, config):
    # Make smoothing kernel.
    smoothing_kernel = None
    if config.smoothed_loss_fraction > 0:
        if config.smoothed_loss_fraction > 1:
            raise ValueError("smoothed_loss_fraction must be less than 1.0")
        if config.smoothing_kernel_size <= 0:
            raise ValueError("smoothing_kernel_size must be positive")
        if config.smoothing_kernel_sigma <= 0:
            raise ValueError("smoothing_kernel_sigma must be positive")
        smoothing_kernel = make_smoothing_kernel(
            config.smoothing_kernel_size, config.smoothing_kernel_sigma,
            config.smoothing_kernel_sigma_z)

    # Make variance penalty kernel.
    variance_penalty_kernel = None
    if config.variance_penalty_coefficient > 0:
        logging.info("Variance penalty coefficient = "
                     f"{config.variance_penalty_coefficient:.2e}")
        variance_penalty_kernel = make_variance_penalty_kernel(
            config.variance_penalty_radius)

    # Weights of smoothed and unsmoothed MSE in the loss function.
    smoothed_fraction = config.smoothed_loss_fraction
    unsmoothed_fraction = 1.0 - smoothed_fraction
    variance_penalty_coeff = config.variance_penalty_coefficient
    logging.info(
        f"Loss function: smoothed_fraction = {smoothed_fraction:.2g}, "
        f"unsmoothed_fraction = {unsmoothed_fraction:.2g}, "
        f"variance_penalty_coeff = {variance_penalty_coeff:.2g}")

    @jax.jit
    def compute_loss(params, inputs, targets):
        predictions = model.module.apply(params, inputs)
        loss = 0.0

        if unsmoothed_fraction > 0:
            mse = losses.mean_squared_error(predictions, targets)
            loss += unsmoothed_fraction * mse

        if smoothed_fraction > 0:
            assert smoothing_kernel is not None
            smoothed_predictions = smooth_grid(predictions, smoothing_kernel)
            smoothed_targets = smooth_grid(targets, smoothing_kernel)
            smoothed_mse = losses.mean_squared_error(smoothed_predictions,
                                                     smoothed_targets)
            loss += smoothed_fraction * smoothed_mse

        if variance_penalty_coeff > 0:
            assert variance_penalty_kernel is not None
            sigmasq1 = compute_sigmasq(predictions, variance_penalty_kernel)
            sigmasq2 = compute_sigmasq(targets, variance_penalty_kernel)
            loss += variance_penalty_coeff * (sigmasq2 - sigmasq1)**2

        return loss

    return compute_loss


def make_training_step_fn(loss_fn, optimizer, config):
    compute_loss_and_grad = jax.value_and_grad(loss_fn)

    @jax.jit
    def run_training_step(param_state, opt_state, inputs, targets):
        params = param_state["params"]
        loss, grad = compute_loss_and_grad(params, inputs, targets)
        updates, opt_state = optimizer.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
        new_param_state = {"params": params}
        for step_size in config.params_ema_step_sizes:
            ema_name = f"ema_{step_size}"
            new_param_state[ema_name] = optax.incremental_update(
                params, param_state[ema_name], step_size)
        return new_param_state, opt_state, loss

    return run_training_step


def make_grid_iter_fn(rng, config, receptive_radius):
    def make_grid_iter(example, num_steps):
        grid_iter = iterators.example_random_iterator(rng, example.input,
                                                      example.target,
                                                      config.train_box_size,
                                                      receptive_radius,
                                                      num_steps)
        for inputs, targets in grid_iter:
            if config.apply_data_augmentation:
                isometry = rng.choice(data_augmentation.NUM_ISOMETRIES)
                inputs = data_augmentation.apply_isometry(inputs, isometry)
                targets = data_augmentation.apply_isometry(targets, isometry)
            yield inputs, targets

    return make_grid_iter


def run(config, model, train_dataset, eval_dataset, model_dir=None):
    steps_per_grid = config.steps_per_grid
    log_frequency = config.log_frequency
    checkpoint_frequency = config.checkpoint_frequency
    eval_frequency = config.eval_frequency
    assert checkpoint_frequency % log_frequency == 0
    assert eval_frequency % log_frequency == 0

    if eval_frequency > 0 and eval_dataset is None:
        raise ValueError("config.eval_frequency requires eval_dataset")

    if checkpoint_frequency > 0 and model_dir is None:
        raise ValueError("config.checkpoint_frequency requires model_dir")

    rng_seed = config.rng_seed or np.random.SeedSequence().entropy
    logging.info(f"Running with seed {rng_seed}")
    rng = np.random.default_rng(rng_seed)

    initial_params = initialize_module(rng, model, train_dataset.nchannels)
    optimizer = create_optimizer(config)
    opt_state = optimizer.init(initial_params)
    if config.apply_cosine_lr_decay:
        raise NotImplementedError(
            "LR decay not implemented since switching to optax.")

    loss_fn = make_loss_fn(model, config)
    run_training_step = make_training_step_fn(loss_fn, optimizer, config)
    predict_fn = make_predict_fn(model, config.eval_box_size)

    param_state = {"params": initial_params}
    for step_size in config.params_ema_step_sizes:
        param_state[f"ema_{step_size}"] = initial_params

    last_log_time = default_timer()
    enqueue_timer = Timer()
    load_timer = Timer()
    wait_timer = Timer()

    dataset_iter = iterators.infinite_shuffle_iterator(rng, train_dataset)
    grid_iter_fn = make_grid_iter_fn(rng, config, model.receptive_radius)

    example = None  # Current training grid
    grid_iter = ()  # Iterator over the training grid
    step = 0  # Total number of training steps
    grid_step = 0  # Number of training steps on this grid
    loss = -1
    training_curve = []
    while True:
        # Enqueue asynchronous training steps on GPU.
        with enqueue_timer:
            for inputs, targets in grid_iter:
                param_state, opt_state, loss = run_training_step(
                    param_state, opt_state, inputs, targets)
                step += 1
                grid_step += 1

        training_finished = step >= config.train_steps

        # Set up the next grid iterator, if necessary. Do this before logging so
        # that loading the grid happens on the CPU while training is still
        # proceeding asychronously on the GPU.
        if not training_finished:
            if example is None or (grid_step >= steps_per_grid):
                with load_timer:
                    example = next(dataset_iter)
                    grid_step = 0
            next_log_step = (step // log_frequency + 1) * log_frequency
            subgrid_steps = min(steps_per_grid, next_log_step - step,
                                config.train_steps - step)
            grid_iter = grid_iter_fn(example, subgrid_steps)

        # Log loss and timing info.
        if training_finished or (step > 0 and step % log_frequency == 0):
            # Read value from GPU. Blocks until enqueued steps are finished.
            with wait_timer:
                loss = float(loss)
            if not np.isfinite(loss):
                logging.error(f"Non-finite loss: {loss}")
                break
            training_curve.append((step, loss))
            logging.info(f"Step {step}. Loss: {loss:.4e}.")
            total_time = default_timer() - last_log_time
            tps = total_time / log_frequency
            logging.info(f"Time per step: {tps:.2e}s")
            enqueue_time = enqueue_timer.elapsed
            load_time = load_timer.elapsed
            wait_time = wait_timer.elapsed
            other_time = total_time - load_time - enqueue_time - wait_time
            logging.info(
                f"Total time {total_time:.2e}s (enqueue {enqueue_time:.2e}s, "
                f"load next example {load_time:.2e}s, wait {wait_time:.2e}s, "
                f"other {other_time:.2e}s).")
            last_log_time = default_timer()
            enqueue_timer.clear()
            load_timer.clear()
            wait_timer.clear()

        # Save checkpoint.
        if checkpoint_frequency > 0 and step and (training_finished or step %
                                                  checkpoint_frequency == 0):
            ckpt_contents = {"opt_state": opt_state, "step": step}
            for key, value in param_state.items():
                if key == "params":
                    key = "target"  # For backwards compatibility.
                ckpt_contents[key] = value
            checkpoints.save_checkpoint(
                model_dir,
                ckpt_contents,
                step,
                keep=config.keep_checkpoint_max,
                keep_every_n_steps=config.keep_checkpoint_every_n_steps)

        # Run evaluations.
        predict_grid = partial(predict_fn, param_state["params"])
        if eval_frequency > 0 and (training_finished
                                   or step % eval_frequency == 0):
            logging.info(f"Starting evaluation at step {step}")
            metrics = evaluation.evaluate_dataset(predict_grid, eval_dataset)
            metrics["train"] = float(loss)
            if model_dir is not None:
                evaluation.write_metrics(metrics, step, model_dir)

        # Training finished!
        if training_finished:
            break

    return training_curve
