"""
Per-transform benchmarks for CPU vs GPU performance comparison.

Usage:
    python benchmarks/benchmark_transforms.py [--shapes small|medium|normal|all]

Sizes:
  - Small:  (32, 32, 32)   — 32K voxels, overhead-dominated
  - Medium: (64, 64, 64)   — 262K voxels
  - Normal: (128, 128, 128) — 2M voxels, typical training patch
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import torch

try:
    from benchmarks.benchmark_utils import BenchmarkRunner
except ModuleNotFoundError:
    from benchmark_utils import BenchmarkRunner


# ---------------------------------------------------------------------------
# Data generation helpers
# ---------------------------------------------------------------------------


def make_image(shape, channels=4, device="cpu", seed=42):
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    img = torch.rand((channels, *shape), generator=gen, dtype=torch.float32)
    return img.to(device)


def make_seg(shape, n_labels=5, channels=1, device="cpu", seed=42):
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    seg = torch.randint(
        0, n_labels, (channels, *shape), generator=gen, dtype=torch.int16
    )
    return seg.to(device)


# ---------------------------------------------------------------------------
# Per-transform data dict factories
# ---------------------------------------------------------------------------


def spatial_data_fn(device="cpu"):
    def fn(shape, seed):
        return {
            "image": make_image(shape, channels=4, device=device, seed=seed),
            "segmentation": make_seg(
                shape, n_labels=5, channels=1, device=device, seed=seed
            ),
        }

    return fn


def image_only_data_fn(device="cpu", channels=4):
    def fn(shape, seed):
        return {"image": make_image(shape, channels=channels, device=device, seed=seed)}

    return fn


# ---------------------------------------------------------------------------
# Transform factories
# ---------------------------------------------------------------------------


def get_transforms(device_str="cpu"):
    """Return list of (name, transform_or_kwargs, data_fn, needs_patch_size) tuples."""
    from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
    from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform
    from batchgeneratorsv2.transforms.spatial.low_resolution import (
        SimulateLowResolutionTransform,
    )
    from batchgeneratorsv2.transforms.intensity.gamma import GammaTransform
    from batchgeneratorsv2.transforms.intensity.contrast import ContrastTransform
    from batchgeneratorsv2.transforms.noise.sharpen import SharpeningTransform
    from batchgeneratorsv2.transforms.noise.median_filter import MedianFilterTransform
    from batchgeneratorsv2.transforms.local.local_gamma import LocalGammaTransform
    from batchgeneratorsv2.transforms.local.local_contrast import LocalContrastTransform
    from batchgeneratorsv2.transforms.local.local_smoothing import (
        LocalSmoothingTransform,
    )
    from batchgeneratorsv2.transforms.local.brightness_gradient import (
        BrightnessGradientAdditiveTransform,
    )

    device = torch.device(device_str) if device_str == "cuda" else None
    transforms = []

    # SpatialTransform — nnUNet defaults
    sp_kwargs = dict(
        patch_size=None,  # will be set per shape
        patch_center_dist_from_border=0,
        random_crop=False,
        p_elastic_deform=0,
        p_rotation=1,
        rotation=(0, 2 * np.pi),
        p_scaling=1,
        scaling=(0.7, 1.4),
        p_synchronize_scaling_across_axes=1,
    )
    if device is not None:
        sp_kwargs["device"] = device
    transforms.append(
        ("SpatialTransform", sp_kwargs, spatial_data_fn(device_str), True)
    )

    # GaussianBlurTransform
    transforms.append(
        (
            "GaussianBlurTransform",
            GaussianBlurTransform(
                blur_sigma=(0.5, 1.0),
                synchronize_channels=False,
                synchronize_axes=False,
                p_per_channel=1,
            ),
            image_only_data_fn(device_str),
            False,
        )
    )

    # SimulateLowResolutionTransform
    transforms.append(
        (
            "SimulateLowResolutionTransform",
            SimulateLowResolutionTransform(
                scale=(0.5, 1),
                synchronize_channels=False,
                synchronize_axes=True,
                ignore_axes=None,
                p_per_channel=1,
            ),
            image_only_data_fn(device_str),
            False,
        )
    )

    # GammaTransform
    transforms.append(
        (
            "GammaTransform",
            GammaTransform(
                gamma=(0.7, 1.5),
                p_invert_image=1,
                synchronize_channels=False,
                p_per_channel=1,
                p_retain_stats=1,
            ),
            image_only_data_fn(device_str),
            False,
        )
    )

    # ContrastTransform
    transforms.append(
        (
            "ContrastTransform",
            ContrastTransform(
                contrast_range=(0.75, 1.25),
                preserve_range=True,
                synchronize_channels=False,
                p_per_channel=1,
            ),
            image_only_data_fn(device_str),
            False,
        )
    )

    # SharpeningTransform
    transforms.append(
        (
            "SharpeningTransform",
            SharpeningTransform(strength=(0.1, 0.3), p_per_channel=1),
            image_only_data_fn(device_str),
            False,
        )
    )

    # MedianFilterTransform (smaller shape for speed — unfold is memory-intensive)
    transforms.append(
        (
            "MedianFilterTransform",
            MedianFilterTransform(filter_size=3, p_per_channel=1),
            image_only_data_fn(device_str),
            False,
        )
    )

    # LocalGammaTransform
    transforms.append(
        (
            "LocalGammaTransform",
            LocalGammaTransform(scale=(10, 20), gamma=(0.5, 1.5), p_per_channel=1),
            image_only_data_fn(device_str),
            False,
        )
    )

    # LocalContrastTransform
    transforms.append(
        (
            "LocalContrastTransform",
            LocalContrastTransform(
                scale=(10, 20), new_contrast=(0.5, 1.5), p_per_channel=1
            ),
            image_only_data_fn(device_str),
            False,
        )
    )

    # LocalSmoothingTransform
    transforms.append(
        (
            "LocalSmoothingTransform",
            LocalSmoothingTransform(
                scale=(10, 20), kernel_size=(1, 3), p_per_channel=1
            ),
            image_only_data_fn(device_str),
            False,
        )
    )

    # BrightnessGradientAdditiveTransform
    transforms.append(
        (
            "BrightnessGradientAdditive",
            BrightnessGradientAdditiveTransform(
                scale=(10, 20), max_strength=(0.1, 0.5), p_per_channel=1
            ),
            image_only_data_fn(device_str),
            False,
        )
    )

    return transforms


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Benchmark transforms CPU vs GPU")
    parser.add_argument(
        "--shapes", default="normal", choices=["small", "medium", "normal", "all"]
    )
    args = parser.parse_args()

    shape_map = {
        "small": [(32, 32, 32)],
        "medium": [(64, 64, 64)],
        "normal": [(128, 128, 128)],
        "all": [(32, 32, 32), (64, 64, 64), (128, 128, 128)],
    }
    shapes = shape_map[args.shapes]

    # torch.utils.benchmark handles warmup + adaptive iteration count + GPU sync
    runner = BenchmarkRunner(min_run_time=0.1, n_runs=50)

    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")

    all_results = []

    for device in devices:
        transforms = get_transforms(device)
        for shape in shapes:
            for name, transform_or_kwargs, data_fn, needs_patch_size in transforms:
                if needs_patch_size:
                    from batchgeneratorsv2.transforms.spatial.spatial import (
                        SpatialTransform,
                    )

                    kwargs = dict(transform_or_kwargs)
                    kwargs["patch_size"] = shape
                    transform = SpatialTransform(**kwargs)
                else:
                    transform = transform_or_kwargs

                print(f"Benchmarking {name} on {device} with shape {shape}...")
                result = runner.benchmark_transform(transform, data_fn, device, shape)
                all_results.append(result)

    print("\n")
    runner.print_table(all_results)

    # Save detailed results
    out_path = os.path.join(
        os.path.dirname(__file__), "results", "benchmark_results.json"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    runner.save_json(all_results, out_path)
    print(f"\nDetailed results saved to {out_path}")


if __name__ == "__main__":
    main()
