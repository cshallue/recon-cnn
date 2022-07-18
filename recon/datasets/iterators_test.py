import numpy as np
from absl.testing import absltest
from recon.datasets import iterators


class ExampleIteratorTest(absltest.TestCase):
    def _run_test(self, input_grid, target_grid, target_boxes,
                  targets_box_size, receptive_radius, expected_inputs,
                  expected_targets):
        with self.subTest(targets_box_size=targets_box_size,
                          receptive_radius=receptive_radius):
            iter = iterators.example_iterator(input_grid,
                                              target_grid,
                                              target_boxes,
                                              targets_box_size,
                                              receptive_radius,
                                              pad_target_grid=True)
            for i, (input_box, target_box) in enumerate(iter):
                np.testing.assert_array_equal(
                    input_box,
                    expected_inputs[i],
                    err_msg=f"Input box does not match at index {i}")
                np.testing.assert_array_equal(
                    target_box,
                    expected_targets[i],
                    err_msg=f"Target box does not match at index {i}")
            self.assertEqual(i, len(expected_inputs) - 1)

    def test_1d(self):
        input_grid = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        target_grid = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
        target_boxes = [[0], [1], [5], [8], [9]]

        self._run_test(input_grid,
                       target_grid,
                       target_boxes,
                       targets_box_size=1,
                       receptive_radius=0,
                       expected_inputs=[[0], [1], [5], [8], [9]],
                       expected_targets=[[0], [10], [50], [80], [90]])

        self._run_test(input_grid,
                       target_grid,
                       target_boxes,
                       targets_box_size=2,
                       receptive_radius=0,
                       expected_inputs=[
                           [0, 1],
                           [1, 2],
                           [5, 6],
                           [8, 9],
                           [9, 0],
                       ],
                       expected_targets=[
                           [0, 10],
                           [10, 20],
                           [50, 60],
                           [80, 90],
                           [90, 0],
                       ])

        self._run_test(input_grid,
                       target_grid,
                       target_boxes,
                       targets_box_size=2,
                       receptive_radius=3,
                       expected_inputs=[
                           [7, 8, 9, 0, 1, 2, 3, 4],
                           [8, 9, 0, 1, 2, 3, 4, 5],
                           [2, 3, 4, 5, 6, 7, 8, 9],
                           [5, 6, 7, 8, 9, 0, 1, 2],
                           [6, 7, 8, 9, 0, 1, 2, 3],
                       ],
                       expected_targets=[
                           [0, 10],
                           [10, 20],
                           [50, 60],
                           [80, 90],
                           [90, 0],
                       ])

    def test_2d(self):
        input_grid = np.array([
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [8, 9, 10, 11],
        ])
        target_grid = np.array([
            [0, 10, 20, 30],
            [40, 50, 60, 70],
            [80, 90, 100, 110],
        ])
        target_boxes = [[0, 1], [2, 2]]

        self._run_test(input_grid,
                       target_grid,
                       target_boxes,
                       targets_box_size=2,
                       receptive_radius=0,
                       expected_inputs=[
                           [[1, 2], [5, 6]],
                           [[10, 11], [2, 3]],
                       ],
                       expected_targets=[
                           [[10, 20], [50, 60]],
                           [[100, 110], [20, 30]],
                       ])

        self._run_test(input_grid,
                       target_grid,
                       target_boxes,
                       targets_box_size=2,
                       receptive_radius=(1, 0),
                       expected_inputs=[
                           [[9, 10], [1, 2], [5, 6], [9, 10]],
                           [[6, 7], [10, 11], [2, 3], [6, 7]],
                       ],
                       expected_targets=[
                           [[10, 20], [50, 60]],
                           [[100, 110], [20, 30]],
                       ])

        self._run_test(input_grid,
                       target_grid,
                       target_boxes,
                       targets_box_size=2,
                       receptive_radius=(0, 1),
                       expected_inputs=[
                           [[0, 1, 2, 3], [4, 5, 6, 7]],
                           [[9, 10, 11, 8], [1, 2, 3, 0]],
                       ],
                       expected_targets=[
                           [[10, 20], [50, 60]],
                           [[100, 110], [20, 30]],
                       ])

        input_grid = np.array([
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [8, 9, 10, 11],
        ])
        self._run_test(input_grid,
                       target_grid,
                       target_boxes,
                       targets_box_size=2,
                       receptive_radius=1,
                       expected_inputs=[[[8, 9, 10, 11], [0, 1, 2, 3],
                                         [4, 5, 6, 7], [8, 9, 10, 11]],
                                        [[5, 6, 7, 4], [9, 10, 11, 8],
                                         [1, 2, 3, 0], [5, 6, 7, 4]]],
                       expected_targets=[
                           [[10, 20], [50, 60]],
                           [[100, 110], [20, 30]],
                       ])

    def test_2d_multifeature(self):
        input_grid = np.array([
            [[0, 0], [1, -1], [2, -2], [3, -3]],
            [[4, -4], [5, -5], [6, -6], [7, -7]],
            [[8, -8], [9, -9], [10, -10], [11, -11]],
        ])
        target_grid = np.array([
            [0, 10, 20, 30],
            [40, 50, 60, 70],
            [80, 90, 100, 110],
        ])
        target_boxes = [[0, 1], [2, 2]]

        self._run_test(input_grid,
                       target_grid,
                       target_boxes,
                       targets_box_size=2,
                       receptive_radius=0,
                       expected_inputs=[
                           [[[1, -1], [2, -2]], [[5, -5], [6, -6]]],
                           [[[10, -10], [11, -11]], [[2, -2], [3, -3]]],
                       ],
                       expected_targets=[
                           [[10, 20], [50, 60]],
                           [[100, 110], [20, 30]],
                       ])


class ExamplePartitionIteratorTest(absltest.TestCase):
    def _run_test(self, input_grid, target_grid, targets_box_size,
                  receptive_radius, expected_inputs, expected_targets):
        with self.subTest(targets_box_size=targets_box_size,
                          receptive_radius=receptive_radius):
            iter = iterators.example_partition_iterator(
                input_grid, target_grid, targets_box_size, receptive_radius)
            for i, (input_box, target_box) in enumerate(iter):
                np.testing.assert_array_equal(
                    input_box,
                    expected_inputs[i],
                    err_msg=f"Input box does not match at index {i}")
                np.testing.assert_array_equal(
                    target_box,
                    expected_targets[i],
                    err_msg=f"Target box does not match at index {i}")
            self.assertEqual(i, len(expected_inputs) - 1)

    def test_1d(self):
        input_grid = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        target_grid = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90])

        self._run_test(input_grid,
                       target_grid,
                       targets_box_size=2,
                       receptive_radius=0,
                       expected_inputs=[
                           [0, 1],
                           [2, 3],
                           [4, 5],
                           [6, 7],
                           [8, 9],
                       ],
                       expected_targets=[
                           [0, 10],
                           [20, 30],
                           [40, 50],
                           [60, 70],
                           [80, 90],
                       ])

        self._run_test(input_grid,
                       target_grid,
                       targets_box_size=2,
                       receptive_radius=3,
                       expected_inputs=[
                           [7, 8, 9, 0, 1, 2, 3, 4],
                           [9, 0, 1, 2, 3, 4, 5, 6],
                           [1, 2, 3, 4, 5, 6, 7, 8],
                           [3, 4, 5, 6, 7, 8, 9, 0],
                           [5, 6, 7, 8, 9, 0, 1, 2],
                       ],
                       expected_targets=[
                           [0, 10],
                           [20, 30],
                           [40, 50],
                           [60, 70],
                           [80, 90],
                       ])

        self._run_test(input_grid,
                       target_grid,
                       targets_box_size=3,
                       receptive_radius=2,
                       expected_inputs=[
                           [8, 9, 0, 1, 2, 3, 4],
                           [1, 2, 3, 4, 5, 6, 7],
                           [4, 5, 6, 7, 8, 9, 0],
                           [7, 8, 9, 0, 1],
                       ],
                       expected_targets=[
                           [0, 10, 20],
                           [30, 40, 50],
                           [60, 70, 80],
                           [90],
                       ])

    def test_2d(self):
        input_grid = np.array([
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [8, 9, 10, 11],
        ])
        target_grid = np.array([
            [0, 10, 20, 30],
            [40, 50, 60, 70],
            [80, 90, 100, 110],
        ])

        self._run_test(input_grid,
                       target_grid,
                       targets_box_size=2,
                       receptive_radius=0,
                       expected_inputs=[
                           [[0, 1], [4, 5]],
                           [[2, 3], [6, 7]],
                           [[8, 9]],
                           [[10, 11]],
                       ],
                       expected_targets=[
                           [[0, 10], [40, 50]],
                           [[20, 30], [60, 70]],
                           [[80, 90]],
                           [[100, 110]],
                       ])

        self._run_test(input_grid,
                       target_grid,
                       targets_box_size=2,
                       receptive_radius=(1, 0),
                       expected_inputs=[
                           [[8, 9], [0, 1], [4, 5], [8, 9]],
                           [[10, 11], [2, 3], [6, 7], [10, 11]],
                           [[4, 5], [8, 9], [0, 1]],
                           [[6, 7], [10, 11], [2, 3]],
                       ],
                       expected_targets=[
                           [[0, 10], [40, 50]],
                           [[20, 30], [60, 70]],
                           [[80, 90]],
                           [[100, 110]],
                       ])

        self._run_test(input_grid,
                       target_grid,
                       targets_box_size=2,
                       receptive_radius=(0, 1),
                       expected_inputs=[
                           [[3, 0, 1, 2], [7, 4, 5, 6]],
                           [[1, 2, 3, 0], [5, 6, 7, 4]],
                           [[11, 8, 9, 10]],
                           [[9, 10, 11, 8]],
                       ],
                       expected_targets=[
                           [[0, 10], [40, 50]],
                           [[20, 30], [60, 70]],
                           [[80, 90]],
                           [[100, 110]],
                       ])

        self._run_test(input_grid,
                       target_grid,
                       targets_box_size=2,
                       receptive_radius=1,
                       expected_inputs=[
                           [[11, 8, 9, 10], [3, 0, 1, 2], [7, 4, 5, 6],
                            [11, 8, 9, 10]],
                           [[9, 10, 11, 8], [1, 2, 3, 0], [5, 6, 7, 4],
                            [9, 10, 11, 8]],
                           [[7, 4, 5, 6], [11, 8, 9, 10], [3, 0, 1, 2]],
                           [[5, 6, 7, 4], [9, 10, 11, 8], [1, 2, 3, 0]],
                       ],
                       expected_targets=[
                           [[0, 10], [40, 50]],
                           [[20, 30], [60, 70]],
                           [[80, 90]],
                           [[100, 110]],
                       ])

    def test_3d(self):
        grid_shape = (5, 6, 7)
        grid_size = np.prod(grid_shape)
        input_grid = np.arange(grid_size, dtype=int).reshape(grid_shape)
        target_grid = 10 * input_grid

        # It's too tricky to write down the expected inputs/outputs explicitly
        # in the 3D case.
        def _run_test_3d(targets_box_size, receptive_radius):
            with self.subTest(targets_box_size=targets_box_size,
                              receptive_radius=receptive_radius):
                iter = iterators.example_partition_iterator(
                    input_grid, target_grid, targets_box_size,
                    receptive_radius)
                target_values = []
                receptive_radius = iterators.as_shape(receptive_radius, 3)
                for input_box, target_box in iter:
                    input_shape = np.array(input_box.shape)
                    target_shape = np.array(target_box.shape)
                    # Boxes have expected shapes.
                    self.assertTrue(np.all(target_shape <= targets_box_size))
                    np.testing.assert_array_equal(
                        input_shape, target_shape + 2 * receptive_radius)
                    # Input grid corresponds to output grid in the expected way.
                    slices = tuple(
                        slice(r, n - r)
                        for r, n in zip(receptive_radius, input_shape))
                    np.testing.assert_array_equal(input_box[slices],
                                                  target_box / 10)
                    target_values.extend(target_box.flatten())

                # All target values were seen exactly once.
                self.assertItemsEqual(target_values, target_grid.flatten())

        for targets_box_size, receptive_radius in [(2, 0), (3, 2), (4, 1),
                                                   ((2, 3, 4), 2),
                                                   (3, (0, 1, 2))]:
            _run_test_3d(targets_box_size, receptive_radius)


if __name__ == "__main__":
    absltest.main()
