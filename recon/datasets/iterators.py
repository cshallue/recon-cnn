import itertools

import numpy as np


def infinite_shuffle_iterator(rng, items):
    while True:
        # Randomly shuffle the indices. Note that we shuffle within epochs, but
        # not between epochs.
        indices = rng.permutation(len(items))
        for i in indices:
            yield items[i]


def _slice_grid(grid, size, start):
    slices = tuple(slice(i, i + l) for i, l in zip(start, size))
    return grid[slices]


def _grid_iterator(grid, box_size, boxes):
    for box_start in boxes:
        box_start = np.asarray(box_start)
        if np.any(box_start < 0) or np.any(box_start >= grid.shape):
            raise ValueError(
                f"Invalid index: {box_start} (grid shape: {grid.shape}")
        yield _slice_grid(grid, box_size, box_start)


def as_shape(x, ndim):
    x = np.asarray(x, dtype=int)
    if x.shape == ():
        x = np.repeat(x, ndim)
    if x.shape != (ndim, ):
        raise ValueError(
            f"Invalid shape: {x}. Expected a scalar or array of length {ndim}")
    return x


def example_iterator(input_grid, target_grid, target_boxes, targets_box_size,
                     receptive_radius, pad_target_grid):
    input_ndim = input_grid.ndim
    target_ndim = target_grid.ndim
    if input_ndim < target_ndim:
        raise ValueError(f"Input grid dimension {input_ndim} < target grid "
                         f"dimension {target_ndim}.")

    if input_grid.shape[:target_ndim] != target_grid.shape:
        raise ValueError(f"Input grid shape {input_grid.shape} incompatible "
                         f"with target grid shape {target_grid.shape}")

    target_boxes = np.asarray(target_boxes)
    if target_boxes.ndim != 2:
        raise ValueError("Expected target_boxes to be 2D. Got shape: "
                         f"{target_boxes.shape}")
    if target_boxes.shape[1] != target_ndim:
        raise ValueError("Target box indices should match the dimension of "
                         f"target grid. Got dimension {target_boxes.shape[1]} "
                         f"vs {target_ndim}.")

    targets_box_size = as_shape(targets_box_size, target_ndim)
    if np.any(targets_box_size <= 0):
        raise ValueError("targets_box_size must be positive. Got: "
                         f"{targets_box_size}")
    if np.any(targets_box_size > target_grid.shape):
        raise ValueError(f"targets_box_size {targets_box_size} > "
                         f"input_grid shape {input_grid.shape}")

    receptive_radius = as_shape(receptive_radius, target_ndim)
    if np.any(receptive_radius < 0):
        raise ValueError("receptive_radius must be nonnegative. Got: "
                         f"{receptive_radius}")
    if np.any(2 * receptive_radius >= target_grid.shape):
        raise ValueError(f"receptive radius {receptive_radius} too large "
                         f"for input_grid shape {input_grid.shape}")

    if pad_target_grid:
        # Pad so that boxes wrap around right boundaries.
        required_target_shape = np.max(target_boxes, axis=0) + targets_box_size
        target_right_pad = np.where(required_target_shape > target_grid.shape,
                                    required_target_shape - target_grid.shape,
                                    0)
    else:
        target_right_pad = np.zeros_like(target_grid.shape, dtype=int)
    target_pad = np.array([(0, p) for p in target_right_pad])
    if np.any(target_pad > 0):
        target_grid = np.pad(target_grid, target_pad, mode="wrap")

    # Pad input grid to account for target padding and the receptive radius.
    input_pad = np.zeros((input_ndim, 2), dtype=int)
    input_pad[:target_ndim, :] = (target_pad +
                                  np.array([(r, r) for r in receptive_radius]))
    input_grid = np.pad(input_grid, input_pad, mode="wrap")

    input_boxes = np.zeros((len(target_boxes), input_ndim), dtype=int)
    input_boxes[:, :target_ndim] = target_boxes
    inputs_box_size = np.zeros((input_ndim), dtype=int)
    inputs_box_size[:target_ndim] = targets_box_size + 2 * receptive_radius
    inputs_box_size[target_ndim:] = input_grid.shape[target_ndim:]

    return zip(_grid_iterator(input_grid, inputs_box_size, input_boxes),
               _grid_iterator(target_grid, targets_box_size, target_boxes))


def _partition_grid(grid, box_size):
    box_size = as_shape(box_size, grid.ndim)
    return list(
        itertools.product(*(range(0, i, l)
                            for i, l in zip(grid.shape, box_size))))


def example_partition_iterator(input_grid, target_grid, targets_box_size,
                               receptive_radius):
    target_boxes = _partition_grid(target_grid, targets_box_size)
    return example_iterator(input_grid,
                            target_grid,
                            target_boxes,
                            targets_box_size,
                            receptive_radius,
                            pad_target_grid=False)


def example_random_iterator(rng, input_grid, target_grid, targets_box_size,
                            receptive_radius, num_steps):
    target_boxes = rng.integers(target_grid.shape, size=(num_steps, 3))
    return example_iterator(input_grid,
                            target_grid,
                            target_boxes,
                            targets_box_size,
                            receptive_radius,
                            pad_target_grid=True)
