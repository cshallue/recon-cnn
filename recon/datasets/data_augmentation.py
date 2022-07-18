import numpy as np

# The 24 distinct rotations of a cube, each decomposed into a pair of rotations
# in the coordinate planes. The rotations have the form (k, axes), which are the
# arguments to np.rot90.
S4_ROTATION_SPECS = [
    ((0, (0, 1)), (0, (0, 1))),
    ((0, (0, 1)), (1, (0, 1))),
    ((0, (0, 1)), (2, (0, 1))),
    ((0, (0, 1)), (3, (0, 1))),
    ((0, (0, 1)), (1, (0, 2))),
    ((0, (0, 1)), (2, (0, 2))),
    ((0, (0, 1)), (3, (0, 2))),
    ((0, (0, 1)), (1, (1, 2))),
    ((0, (0, 1)), (2, (1, 2))),
    ((0, (0, 1)), (3, (1, 2))),
    ((1, (0, 1)), (1, (0, 2))),
    ((1, (0, 1)), (2, (0, 2))),
    ((1, (0, 1)), (3, (0, 2))),
    ((1, (0, 1)), (1, (1, 2))),
    ((1, (0, 1)), (2, (1, 2))),
    ((1, (0, 1)), (3, (1, 2))),
    ((2, (0, 1)), (1, (0, 2))),
    ((2, (0, 1)), (3, (0, 2))),
    ((2, (0, 1)), (1, (1, 2))),
    ((2, (0, 1)), (3, (1, 2))),
    ((3, (0, 1)), (1, (0, 2))),
    ((3, (0, 1)), (3, (0, 2))),
    ((3, (0, 1)), (1, (1, 2))),
    ((3, (0, 1)), (3, (1, 2))),
]

NUM_ROTATIONS = len(S4_ROTATION_SPECS)
NUM_ISOMETRIES = 2 * NUM_ROTATIONS


def apply_rotation(m, i):
    rot1, rot2 = S4_ROTATION_SPECS[i]
    m = np.rot90(m, *rot1)
    m = np.rot90(m, *rot2)
    return m


def apply_isometry(m, i):
    apply_flip, rotation = np.divmod(i, NUM_ROTATIONS)
    if apply_flip:
        m = np.flip(m, 0)
    return apply_rotation(m, rotation)
