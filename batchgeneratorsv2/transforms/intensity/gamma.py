from typing import Optional
import torch

from batchgeneratorsv2.helpers.scalar_type import RandomScalar, sample_scalar
from batchgeneratorsv2.transforms.base.basic_transform import ImageOnlyTransform


class GammaTransform(ImageOnlyTransform):
    def __init__(self,
                 gamma: RandomScalar,
                 p_invert_image: float,
                 synchronize_channels: bool,
                 p_per_channel: float,
                 p_retain_stats: float):
        super().__init__()
        self.gamma = gamma
        self.p_invert_image = float(p_invert_image)
        self.synchronize_channels = synchronize_channels
        self.p_per_channel = float(p_per_channel)
        self.p_retain_stats = float(p_retain_stats)

    def get_parameters(self, **data_dict) -> dict:
        img: torch.Tensor = data_dict["image"]
        c = img.shape[0]
        device = img.device
        dtype = img.dtype

        apply_idx = (torch.rand(c, device=device) < self.p_per_channel).nonzero(as_tuple=False).flatten()
        n = apply_idx.numel()
        if n == 0:
            return {"apply_to_channel": apply_idx,
                    "retain_stats": None,
                    "invert_image": None,
                    "gamma": None}

        retain_stats = (torch.rand(n, device=device) < self.p_retain_stats)
        invert_image = (torch.rand(n, device=device) < self.p_invert_image)

        if self.synchronize_channels:
            g = float(sample_scalar(self.gamma, image=img, channel=None))
            gamma = torch.full((n,), g, device=device, dtype=dtype)
        else:
            # sample_scalar is scalar-based; keep loop but avoid tensor scalar iteration
            gs = [float(sample_scalar(self.gamma, image=img, channel=int(ch))) for ch in apply_idx.tolist()]
            gamma = torch.as_tensor(gs, device=device, dtype=dtype)

        return {
            "apply_to_channel": apply_idx,
            "retain_stats": retain_stats,
            "invert_image": invert_image,
            "gamma": gamma,
        }

    def _apply_to_image(self, img: torch.Tensor, **params) -> torch.Tensor:
        idx: torch.Tensor = params["apply_to_channel"]
        if idx.numel() == 0:
            return img

        retain_stats: torch.Tensor = params["retain_stats"]
        invert_image: torch.Tensor = params["invert_image"]
        gamma: torch.Tensor = params["gamma"]

        eps = 1e-7

        # Gather selected channels: shape (n, *spatial)
        x = img[idx]
        spatial_dims = tuple(range(1, x.ndim))

        # Invert where needed
        inv_mask = invert_image.view(-1, *([1] * (x.ndim - 1)))
        x = torch.where(inv_mask, -x, x)

        # Pre-compute stats for retain_stats channels
        ret_mask = retain_stats
        if ret_mask.any():
            means_orig = x.mean(dim=spatial_dims, keepdim=True)
            stds_orig = x.std(dim=spatial_dims, keepdim=True)

        # Per-channel min/max for gamma correction
        minm = x.amin(dim=spatial_dims, keepdim=True)
        maxm = x.amax(dim=spatial_dims, keepdim=True)
        rnge = maxm - minm
        denom = rnge.clamp(min=eps)

        # Vectorized gamma: ((x - min) / denom) ** gamma * range + min
        gamma_vec = gamma.view(-1, *([1] * (x.ndim - 1)))
        x = ((x - minm) / denom).pow(gamma_vec) * rnge + minm

        # Retain stats where needed
        if ret_mask.any():
            mn_here = x.mean(dim=spatial_dims, keepdim=True)
            std_here = x.std(dim=spatial_dims, keepdim=True)
            x_retained = (x - mn_here) * (stds_orig / std_here.clamp(min=eps)) + means_orig
            ret_mask_bc = ret_mask.view(-1, *([1] * (x.ndim - 1)))
            x = torch.where(ret_mask_bc, x_retained, x)

        # Undo invert where needed
        x = torch.where(inv_mask, -x, x)

        # Write back
        img[idx] = x
        return img



if __name__ == '__main__':
    from time import time
    import numpy as np
    import os

    os.environ['OMP_NUM_THREADS'] = '1'
    torch.set_num_threads(1)

    mbt = GammaTransform((0.7, 1.5), 0, False, 1, 1)

    times_torch = []
    for _ in range(100):
        data_dict = {'image': torch.ones((2, 128, 192, 64))}
        st = time()
        out = mbt(**data_dict)
        times_torch.append(time() - st)
    print('torch', np.mean(times_torch))

    from batchgenerators.transforms.color_transforms import GammaTransform as BGGamma

    gnt_bg = BGGamma((0.7, 1.5), False, True, retain_stats=True, p_per_sample=1)
    times_bg = []
    for _ in range(100):
        data_dict = {'data': np.ones((1, 2, 128, 192, 64))}
        st = time()
        out = gnt_bg(**data_dict)
        times_bg.append(time() - st)
    print('bg', np.mean(times_bg))
