import numpy as np
import pyfftw
from absl import logging
from recon.datasets.base import GridExample


class GradientComputer:
    def __init__(self, shape, dtype, cell_size=1.0):
        # indexing="ij" means the array are indexed A[xi][yi][zi]
        k = np.meshgrid(
            2 * np.pi * np.fft.fftfreq(shape[0], cell_size),
            2 * np.pi * np.fft.fftfreq(shape[1], cell_size),
            2 * np.pi * np.fft.rfftfreq(shape[2], cell_size),  # Note rfft.
            indexing="ij")
        ksq = k[0]**2 + k[1]**2 + k[2]**2

        # Setup the FFTs.
        rshape = np.array(shape)
        cshape = rshape.copy()
        cshape[-1] = (cshape[-1] // 2) + 1
        r_fftgrid = pyfftw.empty_aligned(rshape, dtype="float64")
        c_fftgrid = pyfftw.empty_aligned(cshape, dtype="complex128")
        execute_fft = pyfftw.FFTW(r_fftgrid,
                                  c_fftgrid,
                                  axes=tuple(range(3)),
                                  direction="FFTW_FORWARD")
        execute_inv_fft = pyfftw.FFTW(c_fftgrid,
                                      r_fftgrid,
                                      axes=tuple(range(3)),
                                      direction="FFTW_BACKWARD")

        self._k = k
        self._ksq = ksq
        self._input_dtype = dtype
        self._r_fftgrid = r_fftgrid
        self._c_fftgrid = c_fftgrid
        self._execute_fft = execute_fft
        self._execute_inv_fft = execute_inv_fft

    def _compute_fft(self, input_grid):
        np.copyto(self._r_fftgrid, input_grid)
        self._execute_fft()
        return np.copy(self._c_fftgrid)

    def _compute_ifft(self, kgrid, out):
        np.copyto(self._c_fftgrid, kgrid)
        self._execute_inv_fft()
        np.copyto(out, self._r_fftgrid)

    def __call__(self,
                 input_grid,
                 sigma,
                 smooth_out=None,
                 gradient_out=None,
                 shear_out=None):
        if smooth_out is True:
            smooth_out = np.empty(input_grid.shape, input_grid.dtype)
        if gradient_out is True:
            gradient_out = np.empty(input_grid.shape + (3, ), input_grid.dtype)
        if shear_out is True:
            shear_out = np.empty(input_grid.shape + (6, ), input_grid.dtype)

        if smooth_out is None and gradient_out is None and shear_out is None:
            raise ValueError(
                "One of smooth_out, gradient_out, shear_out is required.")

        input_fft = self._compute_fft(input_grid)

        # Smooth the input grid.
        logging.info(f"Smoothing input grid with sigma = {sigma}")
        kconv = input_fft * np.exp((-sigma**2 / 2) * self._ksq)

        if smooth_out is not None:
            self._compute_ifft(kconv, smooth_out)

        if gradient_out is None and shear_out is None:
            return smooth_out

        # Compute gradients and shears.
        shear_i = 0
        for i in range(3):
            kgrad = 1J * self._k[i] * kconv
            if gradient_out is not None:
                logging.info(f"Computing gradient along axis {i}")
                self._compute_ifft(kgrad, gradient_out[:, :, :, i])

            if shear_out is not None:
                for j in range(i, 3):
                    logging.info(f"Computing shear along axes ({i}, {j})")
                    kshear = 1J * self._k[j] * kgrad
                    self._compute_ifft(kshear, shear_out[:, :, :, shear_i])
                    shear_i += 1

        return tuple(x for x in (smooth_out, gradient_out, shear_out)
                     if x is not None)


def _prepare_sigmas(sigmas, rescale):
    if rescale is None or not len(rescale):
        rescale = np.ones_like(sigmas)
    assert len(rescale) == len(sigmas)
    return {s: r for s, r in zip(sigmas, rescale)}


class ExampleSmoothingFn:
    def __init__(self,
                 smoothing_sigmas,
                 gradient_sigmas,
                 shear_sigmas,
                 smoothing_rescale=None,
                 gradient_rescale=None,
                 shear_rescale=None):
        self._smoothing_sigmas = _prepare_sigmas(smoothing_sigmas,
                                                 smoothing_rescale)
        self._gradient_sigmas = _prepare_sigmas(gradient_sigmas,
                                                gradient_rescale)
        self._shear_sigmas = _prepare_sigmas(shear_sigmas, shear_rescale)

        self._all_sigmas = sorted(
            set(smoothing_sigmas + gradient_sigmas + shear_sigmas))
        self._noutput = (len(smoothing_sigmas) + 3 * len(gradient_sigmas) +
                         6 * len(shear_sigmas))
        self._gradient_computer = None

    @property
    def noutput(self):
        return self._noutput

    def __call__(self, example):
        shape = example.input.shape
        dtype = example.input.dtype
        assert len(shape) == 3
        if self._gradient_computer is None:
            # We're using cell_size = 1.0, which means that sigmas are in units
            # of the cell size, and that gradients are per unit cell size.
            self._gradient_computer = GradientComputer(shape, dtype)

        output = np.empty(shape + (self.noutput, ), dtype)
        i = 0
        for sigma in self._all_sigmas:
            if (sigma == 0 and sigma not in self._gradient_sigmas
                    and sigma not in self._shear_sigmas):
                # We can skip the FFTs in this case.
                assert sigma in self._smoothing_sigmas
                np.copyto(output[:, :, :, i], example.input)
                i += 1
                continue

            smooth_out = None
            gradient_out = None
            shear_out = None
            if sigma in self._smoothing_sigmas:
                smooth_out = output[:, :, :, i]
                i += 1
            if sigma in self._gradient_sigmas:
                gradient_out = output[:, :, :, i:i + 3]
                i += 3
            if sigma in self._shear_sigmas:
                shear_out = output[:, :, :, i:i + 6]
                i += 6
            self._gradient_computer(example.input, sigma, smooth_out,
                                    gradient_out, shear_out)

            # Rescale the grids.
            if smooth_out is not None:
                rescale = self._smoothing_sigmas[sigma]
                logging.info(f"Rescaling smoothed grid by {rescale:.4e}")
                smooth_out *= rescale
            if gradient_out is not None:
                rescale = self._gradient_sigmas[sigma]
                logging.info(f"Rescaling gradient grid by {rescale:.4e}")
                gradient_out *= rescale
            if shear_out is not None:
                rescale = self._shear_sigmas[sigma]
                logging.info(f"Rescaling shear grid by {rescale:.4e}")
                shear_out *= rescale

        assert i == self.noutput

        return GridExample(name=example.name,
                           input=output,
                           target=example.target,
                           metadata=example.metadata)
