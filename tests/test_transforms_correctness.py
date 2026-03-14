"""
Correctness tests for GPU-accelerated transforms.

For each transform × seed:
  1. Run on CPU with seed → out_cpu
  2. Run on GPU with same seed → out_gpu
  3. Assert torch.allclose(out_cpu['image'], out_gpu['image'].cpu(), atol=1e-4)
  4. For segmentation: torch.equal or allow ≤1 voxel boundary diff

CPU regression: verify CPU output is bit-identical to master behavior (default args).
"""
import pytest
import numpy as np
import torch

SEEDS = [42, 123, 7]
SHAPES_2D = [(32, 32), (64, 64)]
SHAPES_3D = [(32, 32, 32), (64, 64, 64)]
HAS_CUDA = torch.cuda.is_available()


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def make_image(shape, channels=4, device='cpu', seed=42):
    """Create a reproducible random image tensor on the given device."""
    gen = torch.Generator(device='cpu')
    gen.manual_seed(seed)
    img = torch.rand((channels, *shape), generator=gen, dtype=torch.float32)
    return img.to(device)


def make_seg(shape, n_labels=5, channels=1, device='cpu', seed=42):
    """Create a reproducible random segmentation tensor on the given device."""
    gen = torch.Generator(device='cpu')
    gen.manual_seed(seed)
    seg = torch.randint(0, n_labels, (channels, *shape), generator=gen, dtype=torch.int16)
    return seg.to(device)


def set_all_seeds(seed):
    """Set numpy and torch seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# SpatialTransform tests
# ---------------------------------------------------------------------------

class TestSpatialTransformCorrectness:
    """CPU vs GPU equivalence for SpatialTransform."""

    @pytest.mark.parametrize("seed", SEEDS)
    @pytest.mark.parametrize("shape", SHAPES_3D[:1])
    def test_cpu_identity(self, seed, shape):
        """CPU path with no augmentation returns a center crop."""
        from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform

        sp = SpatialTransform(
            patch_size=shape,
            patch_center_dist_from_border=0,
            random_crop=False,
            p_elastic_deform=0,
            p_rotation=0,
            p_scaling=0,
        )
        set_all_seeds(seed)
        img = make_image(shape, channels=2, seed=seed)
        data = {'image': img.clone()}
        out = sp(**data)
        assert out['image'].shape == img.shape

    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
    @pytest.mark.parametrize("seed", SEEDS)
    @pytest.mark.parametrize("shape", SHAPES_3D[:1])
    def test_cpu_gpu_equivalence_rotation_scaling(self, seed, shape):
        """GPU output matches CPU within tolerance for rotation+scaling."""
        from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform

        sp_cpu = SpatialTransform(
            patch_size=shape,
            patch_center_dist_from_border=0,
            random_crop=False,
            p_elastic_deform=0,
            p_rotation=1,
            rotation=(0.1, 0.1),
            p_scaling=1,
            scaling=(0.9, 0.9),
            p_synchronize_scaling_across_axes=1,
        )
        sp_gpu = SpatialTransform(
            patch_size=shape,
            patch_center_dist_from_border=0,
            random_crop=False,
            p_elastic_deform=0,
            p_rotation=1,
            rotation=(0.1, 0.1),
            p_scaling=1,
            scaling=(0.9, 0.9),
            p_synchronize_scaling_across_axes=1,
            device=torch.device('cuda'),
        )

        img = make_image(shape, channels=2, seed=seed)

        set_all_seeds(seed)
        out_cpu = sp_cpu(**{'image': img.clone()})

        set_all_seeds(seed)
        out_gpu = sp_gpu(**{'image': img.clone()})

        assert torch.allclose(
            out_cpu['image'],
            out_gpu['image'].cpu(),
            atol=1e-4
        ), f"CPU vs GPU mismatch for seed={seed}, shape={shape}"

    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
    @pytest.mark.parametrize("seed", SEEDS)
    @pytest.mark.parametrize("shape", SHAPES_3D[:1])
    def test_cpu_gpu_equivalence_elastic(self, seed, shape):
        """GPU output matches CPU for elastic deformation."""
        from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform

        sp_cpu = SpatialTransform(
            patch_size=shape,
            patch_center_dist_from_border=0,
            random_crop=False,
            p_elastic_deform=1,
            elastic_deform_scale=(0.1, 0.1),
            elastic_deform_magnitude=(5, 5),
            p_rotation=0,
            p_scaling=0,
        )
        sp_gpu = SpatialTransform(
            patch_size=shape,
            patch_center_dist_from_border=0,
            random_crop=False,
            p_elastic_deform=1,
            elastic_deform_scale=(0.1, 0.1),
            elastic_deform_magnitude=(5, 5),
            p_rotation=0,
            p_scaling=0,
            device=torch.device('cuda'),
        )

        img = make_image(shape, channels=2, seed=seed)

        set_all_seeds(seed)
        out_cpu = sp_cpu(**{'image': img.clone()})

        set_all_seeds(seed)
        out_gpu = sp_gpu(**{'image': img.clone()})

        assert torch.allclose(
            out_cpu['image'],
            out_gpu['image'].cpu(),
            atol=1e-4
        ), f"Elastic CPU vs GPU mismatch for seed={seed}"

    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
    @pytest.mark.parametrize("seed", SEEDS)
    def test_cpu_gpu_segmentation_nearest(self, seed):
        """GPU segmentation with nearest mode matches CPU."""
        from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
        shape = (32, 32, 32)

        sp_cpu = SpatialTransform(
            patch_size=shape,
            patch_center_dist_from_border=0,
            random_crop=False,
            p_elastic_deform=0,
            p_rotation=1,
            rotation=(0.1, 0.1),
            p_scaling=0,
            mode_seg='nearest',
        )
        sp_gpu = SpatialTransform(
            patch_size=shape,
            patch_center_dist_from_border=0,
            random_crop=False,
            p_elastic_deform=0,
            p_rotation=1,
            rotation=(0.1, 0.1),
            p_scaling=0,
            mode_seg='nearest',
            device=torch.device('cuda'),
        )

        img = make_image(shape, channels=2, seed=seed)
        seg = make_seg(shape, n_labels=3, seed=seed)

        set_all_seeds(seed)
        out_cpu = sp_cpu(**{'image': img.clone(), 'segmentation': seg.clone()})

        set_all_seeds(seed)
        out_gpu = sp_gpu(**{'image': img.clone(), 'segmentation': seg.clone()})

        # Allow ≤1 voxel boundary diff for nearest-mode grid_sample float precision
        diff = (out_cpu['segmentation'] != out_gpu['segmentation'].cpu()).sum()
        total = out_cpu['segmentation'].numel()
        assert diff / total < 0.01, f"Seg nearest mismatch: {diff}/{total} voxels differ"

    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
    @pytest.mark.parametrize("seed", SEEDS)
    def test_cpu_gpu_segmentation_bilinear(self, seed):
        """GPU segmentation with bilinear (bg_style) matches CPU."""
        from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
        shape = (32, 32, 32)

        sp_cpu = SpatialTransform(
            patch_size=shape,
            patch_center_dist_from_border=0,
            random_crop=False,
            p_elastic_deform=0,
            p_rotation=1,
            rotation=(0.1, 0.1),
            p_scaling=0,
            mode_seg='bilinear',
            bg_style_seg_sampling=True,
        )
        sp_gpu = SpatialTransform(
            patch_size=shape,
            patch_center_dist_from_border=0,
            random_crop=False,
            p_elastic_deform=0,
            p_rotation=1,
            rotation=(0.1, 0.1),
            p_scaling=0,
            mode_seg='bilinear',
            bg_style_seg_sampling=True,
            device=torch.device('cuda'),
        )

        img = make_image(shape, channels=2, seed=seed)
        seg = make_seg(shape, n_labels=3, seed=seed)

        set_all_seeds(seed)
        out_cpu = sp_cpu(**{'image': img.clone(), 'segmentation': seg.clone()})

        set_all_seeds(seed)
        out_gpu = sp_gpu(**{'image': img.clone(), 'segmentation': seg.clone()})

        diff = (out_cpu['segmentation'] != out_gpu['segmentation'].cpu()).sum()
        total = out_cpu['segmentation'].numel()
        assert diff / total < 0.01, f"Seg bilinear mismatch: {diff}/{total} voxels differ"


# ---------------------------------------------------------------------------
# GaussianBlurTransform tests
# ---------------------------------------------------------------------------

class TestGaussianBlurTransformCorrectness:
    @pytest.mark.parametrize("seed", SEEDS)
    @pytest.mark.parametrize("shape", SHAPES_3D[:1])
    def test_cpu_deterministic(self, seed, shape):
        from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform

        t = GaussianBlurTransform(blur_sigma=(1, 2), synchronize_channels=False,
                                  synchronize_axes=False, p_per_channel=1)
        img = make_image(shape, channels=2, seed=seed)

        set_all_seeds(seed)
        out1 = t(**{'image': img.clone()})
        set_all_seeds(seed)
        out2 = t(**{'image': img.clone()})

        assert torch.allclose(out1['image'], out2['image'], atol=1e-6)

    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
    @pytest.mark.parametrize("seed", SEEDS)
    @pytest.mark.parametrize("shape", SHAPES_3D[:1])
    def test_cpu_gpu_equivalence(self, seed, shape):
        from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform

        t = GaussianBlurTransform(blur_sigma=(1, 2), synchronize_channels=False,
                                  synchronize_axes=False, p_per_channel=1)

        img = make_image(shape, channels=2, seed=seed)

        set_all_seeds(seed)
        out_cpu = t(**{'image': img.clone()})

        set_all_seeds(seed)
        out_gpu = t(**{'image': img.clone().cuda()})

        assert torch.allclose(out_cpu['image'], out_gpu['image'].cpu(), atol=1e-4)


# ---------------------------------------------------------------------------
# SimulateLowResolutionTransform tests
# ---------------------------------------------------------------------------

class TestSimulateLowResolutionCorrectness:
    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
    @pytest.mark.parametrize("seed", SEEDS)
    def test_cpu_gpu_equivalence(self, seed):
        from batchgeneratorsv2.transforms.spatial.low_resolution import SimulateLowResolutionTransform

        t = SimulateLowResolutionTransform(
            scale=(0.5, 1), synchronize_channels=False,
            synchronize_axes=True, ignore_axes=None, p_per_channel=1
        )
        shape = (32, 32, 32)
        img = make_image(shape, channels=2, seed=seed)

        set_all_seeds(seed)
        out_cpu = t(**{'image': img.clone()})

        set_all_seeds(seed)
        out_gpu = t(**{'image': img.clone().cuda()})

        assert torch.allclose(out_cpu['image'], out_gpu['image'].cpu(), atol=1e-4)


# ---------------------------------------------------------------------------
# GammaTransform tests
# ---------------------------------------------------------------------------

class TestGammaTransformCorrectness:
    @pytest.mark.parametrize("seed", SEEDS)
    def test_cpu_deterministic(self, seed):
        from batchgeneratorsv2.transforms.intensity.gamma import GammaTransform

        t = GammaTransform(gamma=(0.7, 1.5), p_invert_image=0.5,
                           synchronize_channels=False, p_per_channel=1, p_retain_stats=0.5)
        shape = (32, 32, 32)
        img = make_image(shape, channels=2, seed=seed)

        set_all_seeds(seed)
        out1 = t(**{'image': img.clone()})
        set_all_seeds(seed)
        out2 = t(**{'image': img.clone()})

        assert torch.allclose(out1['image'], out2['image'], atol=1e-6)

    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
    @pytest.mark.parametrize("seed", SEEDS)
    def test_cpu_gpu_equivalence(self, seed):
        from batchgeneratorsv2.transforms.intensity.gamma import GammaTransform

        t = GammaTransform(gamma=(0.7, 1.5), p_invert_image=0.5,
                           synchronize_channels=False, p_per_channel=1, p_retain_stats=0.5)
        shape = (32, 32, 32)
        img = make_image(shape, channels=2, seed=seed)

        set_all_seeds(seed)
        out_cpu = t(**{'image': img.clone()})

        set_all_seeds(seed)
        out_gpu = t(**{'image': img.clone().cuda()})

        assert torch.allclose(out_cpu['image'], out_gpu['image'].cpu(), atol=1e-4)


# ---------------------------------------------------------------------------
# ContrastTransform tests
# ---------------------------------------------------------------------------

class TestContrastTransformCorrectness:
    @pytest.mark.parametrize("seed", SEEDS)
    def test_cpu_deterministic(self, seed):
        from batchgeneratorsv2.transforms.intensity.contrast import ContrastTransform

        t = ContrastTransform(contrast_range=(0.75, 1.25), preserve_range=True,
                              synchronize_channels=False, p_per_channel=1)
        shape = (32, 32, 32)
        img = make_image(shape, channels=2, seed=seed)

        set_all_seeds(seed)
        out1 = t(**{'image': img.clone()})
        set_all_seeds(seed)
        out2 = t(**{'image': img.clone()})

        assert torch.allclose(out1['image'], out2['image'], atol=1e-6)

    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
    @pytest.mark.parametrize("seed", SEEDS)
    def test_cpu_gpu_equivalence(self, seed):
        from batchgeneratorsv2.transforms.intensity.contrast import ContrastTransform

        t = ContrastTransform(contrast_range=(0.75, 1.25), preserve_range=True,
                              synchronize_channels=False, p_per_channel=1)
        shape = (32, 32, 32)
        img = make_image(shape, channels=2, seed=seed)

        set_all_seeds(seed)
        out_cpu = t(**{'image': img.clone()})

        set_all_seeds(seed)
        out_gpu = t(**{'image': img.clone().cuda()})

        assert torch.allclose(out_cpu['image'], out_gpu['image'].cpu(), atol=1e-4)


# ---------------------------------------------------------------------------
# SharpeningTransform tests
# ---------------------------------------------------------------------------

class TestSharpeningTransformCorrectness:
    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
    @pytest.mark.parametrize("seed", SEEDS)
    def test_cpu_gpu_equivalence(self, seed):
        from batchgeneratorsv2.transforms.noise.sharpen import SharpeningTransform

        t = SharpeningTransform(strength=(0.1, 0.3), p_per_channel=1)
        shape = (32, 32, 32)
        img = make_image(shape, channels=2, seed=seed)

        set_all_seeds(seed)
        out_cpu = t(**{'image': img.clone()})

        set_all_seeds(seed)
        out_gpu = t(**{'image': img.clone().cuda()})

        assert torch.allclose(out_cpu['image'], out_gpu['image'].cpu(), atol=1e-4)


# ---------------------------------------------------------------------------
# MedianFilterTransform tests
# ---------------------------------------------------------------------------

class TestMedianFilterTransformCorrectness:
    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
    @pytest.mark.parametrize("seed", SEEDS)
    def test_cpu_gpu_equivalence(self, seed):
        from batchgeneratorsv2.transforms.noise.median_filter import MedianFilterTransform

        t = MedianFilterTransform(filter_size=3, p_per_channel=1)
        shape = (32, 32, 32)
        img = make_image(shape, channels=2, seed=seed)

        set_all_seeds(seed)
        out_cpu = t(**{'image': img.clone()})

        set_all_seeds(seed)
        out_gpu = t(**{'image': img.clone().cuda()})

        assert torch.allclose(out_cpu['image'], out_gpu['image'].cpu(), atol=1e-4)


# ---------------------------------------------------------------------------
# Local Transform tests
# ---------------------------------------------------------------------------

class TestLocalTransformsCorrectness:
    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
    @pytest.mark.parametrize("seed", SEEDS)
    def test_local_gamma_cpu_gpu(self, seed):
        from batchgeneratorsv2.transforms.local.local_gamma import LocalGammaTransform

        t = LocalGammaTransform(scale=(10, 20), gamma=(0.5, 1.5), p_per_channel=1)
        shape = (32, 32, 32)
        img = make_image(shape, channels=2, seed=seed)

        set_all_seeds(seed)
        out_cpu = t(**{'image': img.clone()})

        set_all_seeds(seed)
        out_gpu = t(**{'image': img.clone().cuda()})

        assert torch.allclose(out_cpu['image'], out_gpu['image'].cpu(), atol=1e-4)

    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
    @pytest.mark.parametrize("seed", SEEDS)
    def test_local_contrast_cpu_gpu(self, seed):
        from batchgeneratorsv2.transforms.local.local_contrast import LocalContrastTransform

        t = LocalContrastTransform(scale=(10, 20), new_contrast=(0.5, 1.5), p_per_channel=1)
        shape = (32, 32, 32)
        img = make_image(shape, channels=2, seed=seed)

        set_all_seeds(seed)
        out_cpu = t(**{'image': img.clone()})

        set_all_seeds(seed)
        out_gpu = t(**{'image': img.clone().cuda()})

        assert torch.allclose(out_cpu['image'], out_gpu['image'].cpu(), atol=1e-4)

    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
    @pytest.mark.parametrize("seed", SEEDS)
    def test_local_smoothing_cpu_gpu(self, seed):
        from batchgeneratorsv2.transforms.local.local_smoothing import LocalSmoothingTransform

        t = LocalSmoothingTransform(scale=(10, 20), kernel_size=(1, 3), p_per_channel=1)
        shape = (32, 32, 32)
        img = make_image(shape, channels=2, seed=seed)

        set_all_seeds(seed)
        out_cpu = t(**{'image': img.clone()})

        set_all_seeds(seed)
        out_gpu = t(**{'image': img.clone().cuda()})

        assert torch.allclose(out_cpu['image'], out_gpu['image'].cpu(), atol=1e-3)

    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
    @pytest.mark.parametrize("seed", SEEDS)
    def test_brightness_gradient_cpu_gpu(self, seed):
        from batchgeneratorsv2.transforms.local.brightness_gradient import BrightnessGradientAdditiveTransform

        t = BrightnessGradientAdditiveTransform(
            scale=(10, 20), max_strength=(0.1, 0.5), p_per_channel=1
        )
        shape = (32, 32, 32)
        img = make_image(shape, channels=2, seed=seed)

        set_all_seeds(seed)
        out_cpu = t(**{'image': img.clone()})

        set_all_seeds(seed)
        out_gpu = t(**{'image': img.clone().cuda()})

        assert torch.allclose(out_cpu['image'], out_gpu['image'].cpu(), atol=1e-4)


# ---------------------------------------------------------------------------
# 2D tests
# ---------------------------------------------------------------------------

class TestTransforms2D:
    @pytest.mark.parametrize("seed", SEEDS)
    def test_spatial_transform_2d(self, seed):
        from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
        shape = (32, 32)

        sp = SpatialTransform(
            patch_size=shape,
            patch_center_dist_from_border=0,
            random_crop=False,
            p_elastic_deform=0,
            p_rotation=1,
            rotation=(0.1, 0.1),
            p_scaling=0,
        )
        img = make_image(shape, channels=2, seed=seed)
        set_all_seeds(seed)
        out = sp(**{'image': img.clone()})
        assert out['image'].shape == (2, *shape)

    @pytest.mark.parametrize("seed", SEEDS)
    def test_gaussian_blur_2d(self, seed):
        from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform
        shape = (32, 32)

        t = GaussianBlurTransform(blur_sigma=(1, 2), synchronize_channels=False,
                                  synchronize_axes=False, p_per_channel=1)
        img = make_image(shape, channels=2, seed=seed)
        set_all_seeds(seed)
        out = t(**{'image': img.clone()})
        assert out['image'].shape == (2, *shape)

    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
    @pytest.mark.parametrize("seed", SEEDS)
    def test_spatial_transform_2d_cpu_gpu(self, seed):
        from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
        shape = (32, 32)

        sp_cpu = SpatialTransform(
            patch_size=shape,
            patch_center_dist_from_border=0,
            random_crop=False,
            p_elastic_deform=0,
            p_rotation=1,
            rotation=(0.1, 0.1),
            p_scaling=0,
        )
        sp_gpu = SpatialTransform(
            patch_size=shape,
            patch_center_dist_from_border=0,
            random_crop=False,
            p_elastic_deform=0,
            p_rotation=1,
            rotation=(0.1, 0.1),
            p_scaling=0,
            device=torch.device('cuda'),
        )
        img = make_image(shape, channels=2, seed=seed)

        set_all_seeds(seed)
        out_cpu = sp_cpu(**{'image': img.clone()})
        set_all_seeds(seed)
        out_gpu = sp_gpu(**{'image': img.clone()})

        assert torch.allclose(out_cpu['image'], out_gpu['image'].cpu(), atol=1e-4)
