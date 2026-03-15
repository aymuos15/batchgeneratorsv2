import torch
from abc import ABC
from typing import Tuple, Union
from batchgeneratorsv2.helpers.scalar_type import RandomScalar, sample_scalar

# Standard normal distribution for CDF computation
_NORMAL = torch.distributions.Normal(0, 1)


def _norm_cdf(x: torch.Tensor, loc: float, scale: float) -> torch.Tensor:
    """Compute normal CDF, equivalent to scipy.stats.norm.cdf."""
    return _NORMAL.cdf((x - loc) / scale)


class LocalTransform(ABC):
    def __init__(self, scale: RandomScalar, loc: RandomScalar = (-1, 2)):
        self.loc = loc
        self.scale = scale

    def _generate_kernel(self, img_shp: Tuple[int, ...], device=None) -> torch.Tensor:
        ndim = len(img_shp)
        dev = device if device is not None else 'cpu'
        kernels = []

        for d in range(ndim):
            loc_val = sample_scalar(self.loc, img_shp, d)
            scale_val = sample_scalar(self.scale, img_shp, d)
            loc_rescaled = loc_val * img_shp[d]
            x_grid = torch.arange(-0.5, img_shp[d] + 0.5, dtype=torch.float32, device=dev)
            cdf = _norm_cdf(x_grid, loc=loc_rescaled, scale=scale_val)
            kernels.append(cdf[1:] - cdf[:-1])  # diff

        # Outer product to build spatial kernel
        kernel = kernels[0].unsqueeze(1) * kernels[1].unsqueeze(0)
        if ndim == 3:
            kernel = kernel.unsqueeze(2) * kernels[2].unsqueeze(0).unsqueeze(0)

        kernel -= kernel.min()
        kernel_max = kernel.max()
        if kernel_max > 0:
            kernel /= kernel_max
        return kernel

    def _generate_multiple_kernel_image(self, img_shp: Tuple[int, ...],
                                         num_kernels: int,
                                         device=None) -> torch.Tensor:
        """
        Places multiple additive Gaussians in the image and normalizes the sum to [0, 1].

        Parameters:
            img_shp (Tuple[int, ...]): Spatial shape (e.g., (X, Y[, Z]))
            num_kernels (int): Number of kernels to generate and sum
            device: Target device

        Returns:
            torch.Tensor: Combined kernel image with values in [0, 1]
        """
        dev = device if device is not None else 'cpu'
        kernel_image = torch.zeros(img_shp, dtype=torch.float32, device=dev)
        for _ in range(num_kernels):
            kernel_image += self._generate_kernel(img_shp, device=device)

        kernel_image -= kernel_image.min()
        kernel_max = kernel_image.max()
        if kernel_max > 0:
            kernel_image /= kernel_max
        return kernel_image

    @staticmethod
    def invert_kernel(kernel_image: torch.Tensor) -> torch.Tensor:
        """
        Inverts a normalized kernel: 1 - kernel

        Assumes input is in [0, 1].

        Parameters:
            kernel_image (torch.Tensor): Input kernel in [0, 1]

        Returns:
            torch.Tensor: Inverted kernel in [0, 1]
        """
        return 1.0 - kernel_image

    @staticmethod
    def run_interpolation(original_image: torch.Tensor,
                          modified_image: torch.Tensor,
                          kernel_image: torch.Tensor) -> torch.Tensor:
        """
        Blends original and modified images using the given kernel as a per-pixel weight map.

        Parameters:
            original_image (torch.Tensor): Unmodified input image
            modified_image (torch.Tensor): Modified version (e.g., gamma-corrected)
            kernel_image (torch.Tensor): Kernel in [0, 1], where 0 = keep original, 1 = keep modified

        Returns:
            torch.Tensor: Blended result
        """
        return original_image * (1.0 - kernel_image) + modified_image * kernel_image
