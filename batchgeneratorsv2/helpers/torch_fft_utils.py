"""
Pure-torch replacement for scipy.ndimage.fourier_gaussian.
Works on any device (CPU or CUDA) without numpy/scipy dependencies.
"""
import torch
from torch import Tensor


def torch_fourier_gaussian(x: Tensor, sigmas) -> Tensor:
    """
    Apply Gaussian smoothing in the Fourier domain, equivalent to
    scipy.ndimage.fourier_gaussian applied to numpy FFT output.

    This operates entirely in torch and works on any device.

    Args:
        x: Input tensor (real-valued, any shape). Will be FFT'd internally.
        sigmas: Scalar or per-axis sigma values. If scalar, applied uniformly.
                If sequence, length must match x.ndim.

    Returns:
        Real-valued smoothed tensor, same shape and device as input.
    """
    ndim = x.ndim
    if not hasattr(sigmas, '__len__'):
        sigmas = [sigmas] * ndim

    # Compute FFT
    dims = tuple(range(ndim))
    x_fft = torch.fft.fftn(x, dim=dims)

    # Build Gaussian filter in frequency domain
    # For each axis, the Gaussian in frequency domain is:
    #   exp(-2 * pi^2 * sigma^2 * freq^2)
    # where freq goes from 0 to N-1, normalized by N
    for axis in range(ndim):
        n = x.shape[axis]
        sigma = sigmas[axis]
        # Frequency indices: 0, 1, ..., N-1
        freq = torch.arange(n, device=x.device, dtype=x.dtype)
        # Normalize frequencies to [0, 1) range (matching scipy convention)
        freq = freq / n
        # Gaussian in frequency domain
        gauss = torch.exp(-2.0 * (torch.pi ** 2) * (sigma ** 2) * (freq ** 2))
        # Reshape for broadcasting
        shape = [1] * ndim
        shape[axis] = n
        gauss = gauss.reshape(shape)
        x_fft = x_fft * gauss

    # Inverse FFT
    result = torch.fft.ifftn(x_fft, dim=dims)
    return result.real
