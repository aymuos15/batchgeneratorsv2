"""
nnUNet full augmentation pipeline: CPU vs GPU comparison.

Runs the exact same transform pipeline used in nnUNet training,
once on CPU and once on GPU, and compares per-transform and total time.

Usage:
    python -m benchmarks.benchmark_nnunet_pipeline
"""
import numpy as np
import torch
import torch.utils.benchmark as bench

from batchgeneratorsv2.transforms.intensity.brightness import MultiplicativeBrightnessTransform
from batchgeneratorsv2.transforms.intensity.contrast import BGContrast, ContrastTransform
from batchgeneratorsv2.transforms.intensity.gamma import GammaTransform
from batchgeneratorsv2.transforms.intensity.gaussian_noise import GaussianNoiseTransform
from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform
from batchgeneratorsv2.transforms.spatial.low_resolution import SimulateLowResolutionTransform
from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.deep_supervision_downsampling import DownsampleSegForDSTransform
from batchgeneratorsv2.transforms.utils.nnunet_masking import MaskImageTransform
from batchgeneratorsv2.transforms.utils.remove_label import RemoveLabelTansform
from batchgeneratorsv2.transforms.utils.seg_to_regions import ConvertSegmentationToRegionsTransform


def build_pipeline(device=None):
    """Build the standard nnUNet augmentation pipeline."""
    regions = ((1, 2, 3), (2, 3), (3,))
    patch_size = (128, 128, 128)
    rotation_for_DA = (0, 2 * np.pi)
    deep_supervision_scales = ((1, 1, 1), (0.5, 0.5, 0.5), (0.25, 0.25, 0.25))

    sp_kwargs = dict(
        patch_size=patch_size,
        patch_center_dist_from_border=0,
        random_crop=False,
        p_elastic_deform=0,
        p_rotation=1,
        rotation=rotation_for_DA,
        p_scaling=1,
        scaling=(0.7, 1.4),
        p_synchronize_scaling_across_axes=1,
    )
    if device is not None:
        sp_kwargs['device'] = device

    transforms = [
        SpatialTransform(**sp_kwargs),
        GaussianNoiseTransform(noise_variance=(0, 0.1), p_per_channel=1, synchronize_channels=True),
        GaussianBlurTransform(blur_sigma=(0.5, 1.), synchronize_channels=False,
                              synchronize_axes=False, p_per_channel=1),
        MultiplicativeBrightnessTransform(multiplier_range=BGContrast((0.75, 1.25)),
                                          synchronize_channels=False, p_per_channel=1),
        ContrastTransform(contrast_range=BGContrast((0.75, 1.25)), preserve_range=True,
                          synchronize_channels=False, p_per_channel=1),
        SimulateLowResolutionTransform(scale=(0.5, 1), synchronize_channels=False,
                                       synchronize_axes=True, ignore_axes=None, p_per_channel=1),
        GammaTransform(gamma=BGContrast((0.7, 1.5)), p_invert_image=1,
                       synchronize_channels=False, p_per_channel=1, p_retain_stats=1),
        GammaTransform(gamma=BGContrast((0.7, 1.5)), p_invert_image=0,
                       synchronize_channels=False, p_per_channel=1, p_retain_stats=1),
        MirrorTransform(allowed_axes=(0, 1, 2)),
        MaskImageTransform(apply_to_channels=[0, 1, 2, 3], channel_idx_in_seg=0, set_outside_to=0),
        RemoveLabelTansform(-1, 0),
        ConvertSegmentationToRegionsTransform(regions=regions, channel_in_seg=0),
        DownsampleSegForDSTransform(ds_scales=deep_supervision_scales),
    ]
    return ComposeTransforms(transforms)


def make_data(device='cpu'):
    """Create a single nnUNet-style training sample."""
    img = torch.rand((4, 128, 128, 128), device=device)
    seg = torch.round(4.5 * torch.rand((1, 128, 128, 128), device=device) - 1, decimals=0).to(torch.int8)
    return {'image': img, 'segmentation': seg}


def benchmark_pipeline(pipeline, device_str, n_runs=10):
    """Benchmark the full pipeline with torch.utils.benchmark."""
    template = make_data(device_str)
    timer = bench.Timer(
        stmt="""\
data = {k: v.clone() for k, v in template.items()}
with torch.no_grad():
    pipeline(**data)
""",
        globals={'pipeline': pipeline, 'template': template, 'torch': torch},
        num_threads=torch.get_num_threads(),
        label="nnUNet pipeline",
        sub_label=device_str,
    )

    medians = []
    for _ in range(n_runs):
        m = timer.blocked_autorange(min_run_time=0.1)
        medians.append(m.median * 1000.0)

    import statistics
    return statistics.median(medians), statistics.stdev(medians)


if __name__ == '__main__':
    n_runs = 50
    n_threads = torch.get_num_threads()

    print(f"nnUNet Full Augmentation Pipeline Benchmark")
    print(f"Patch: 128x128x128, 4ch image + 1ch seg")
    print(f"CPU threads: {n_threads}, Runs: {n_runs}")
    print()

    # CPU pipeline
    print("Building CPU pipeline...")
    pipe_cpu = build_pipeline(device=None)
    print(f"Benchmarking CPU ({n_runs} runs)...")
    cpu_median, cpu_std = benchmark_pipeline(pipe_cpu, 'cpu', n_runs)

    # GPU pipeline
    if torch.cuda.is_available():
        print(f"Building GPU pipeline...")
        pipe_gpu = build_pipeline(device=torch.device('cuda'))
        print(f"Benchmarking GPU ({n_runs} runs)...")
        gpu_median, gpu_std = benchmark_pipeline(pipe_gpu, 'cuda', n_runs)

        print()
        print(f"{'':=<50}")
        print(f"  CPU:     {cpu_median:>8.1f} ms +/- {cpu_std:.1f}")
        print(f"  GPU:     {gpu_median:>8.1f} ms +/- {gpu_std:.1f}")
        print(f"  Speedup: {cpu_median / gpu_median:>8.1f}x")
        print(f"  GPU mem: {torch.cuda.max_memory_allocated() / 1024**2:.0f} MB peak")
        print(f"{'':=<50}")
    else:
        print()
        print(f"CPU: {cpu_median:.1f} ms +/- {cpu_std:.1f}")
        print("(CUDA not available)")
