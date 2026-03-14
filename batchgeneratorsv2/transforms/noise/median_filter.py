import numpy as np
import torch
import torch.nn.functional as F
from typing import Union, Tuple

from batchgeneratorsv2.transforms.base.basic_transform import ImageOnlyTransform


def _median_filter_torch_3d(x: torch.Tensor, size: int) -> torch.Tensor:
    """Apply 3D median filter using unfold + median. x shape: (D, H, W)."""
    pad = size // 2
    # Pad with replicate to match scipy behavior at boundaries
    x_padded = F.pad(x.unsqueeze(0).unsqueeze(0), [pad] * 6, mode='replicate')[0, 0]
    # Use unfold along each dimension to extract sliding windows
    unfolded = x_padded.unfold(0, size, 1).unfold(1, size, 1).unfold(2, size, 1)
    # unfolded shape: (D, H, W, size, size, size)
    return unfolded.contiguous().view(*x.shape, -1).median(dim=-1).values


def _median_filter_torch_2d(x: torch.Tensor, size: int) -> torch.Tensor:
    """Apply 2D median filter using unfold + median. x shape: (H, W)."""
    pad = size // 2
    x_padded = F.pad(x.unsqueeze(0).unsqueeze(0), [pad] * 4, mode='replicate')[0, 0]
    unfolded = x_padded.unfold(0, size, 1).unfold(1, size, 1)
    # unfolded shape: (H, W, size, size)
    return unfolded.contiguous().view(*x.shape, -1).median(dim=-1).values


class MedianFilterTransform(ImageOnlyTransform):
    """
    Applies a median filter to selected image channels.

    Attributes:
        filter_size (int or Tuple[int, int]): Either fixed filter size or range for random sampling.
        p_same_for_each_channel (float): Probability that all channels share the same filter size.
        p_per_channel (float): Probability of applying the filter to a given channel.
    """

    def __init__(self,
                 filter_size: Union[int, Tuple[int, int]],
                 p_same_for_each_channel: float = 0.0,
                 p_per_channel: float = 1.0):
        super().__init__()
        self.filter_size = filter_size
        self.p_same_for_each_channel = p_same_for_each_channel
        self.p_per_channel = p_per_channel

    def get_parameters(self, image: torch.Tensor, **kwargs) -> dict:
        C = image.shape[0]
        use_same = np.random.rand() < self.p_same_for_each_channel

        if isinstance(self.filter_size, int):
            sizes = [self.filter_size] * C
        elif use_same:
            sampled_size = int(np.random.randint(*self.filter_size))
            sizes = [sampled_size] * C
        else:
            sizes = [int(np.random.randint(*self.filter_size)) for _ in range(C)]

        apply_channel = [np.random.rand() < self.p_per_channel for _ in range(C)]

        return {
            'filter_sizes': sizes,
            'apply_channel': apply_channel
        }

    def _apply_to_image(self, img: torch.Tensor, **params) -> torch.Tensor:
        ndim = img.ndim - 1  # spatial dims
        for c, (apply, size) in enumerate(zip(params['apply_channel'], params['filter_sizes'])):
            if not apply:
                continue
            if ndim == 3:
                img[c] = _median_filter_torch_3d(img[c], size)
            elif ndim == 2:
                img[c] = _median_filter_torch_2d(img[c], size)
            else:
                raise ValueError(f"Unsupported spatial dimensions: {ndim}")
        return img
