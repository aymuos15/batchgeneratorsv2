"""
Reusable benchmarking infrastructure for transform performance measurement.

Uses torch.utils.benchmark.Timer for correct GPU timing (automatic
synchronization, adaptive iteration count, proper warmup).
"""
import json
from dataclasses import dataclass, field, asdict
from typing import Callable, List, Optional, Tuple

import torch
import torch.utils.benchmark as bench


@dataclass
class BenchmarkResult:
    transform_name: str
    device: str
    shape: tuple
    median_ms: float = 0.0
    iqr_ms: float = 0.0
    mean_ms: float = 0.0
    num_threads: int = 1
    gpu_mem_peak_mb: Optional[float] = None


class BenchmarkRunner:
    def __init__(self, min_run_time: float = 2.0, num_threads: int = None):
        """
        Args:
            min_run_time: Minimum total seconds for the timed region.
                          torch.utils.benchmark auto-selects iteration count.
            num_threads: CPU thread count. None = use current default.
        """
        self.min_run_time = min_run_time
        self.num_threads = num_threads or torch.get_num_threads()

    def benchmark_transform(self, transform, data_dict_fn: Callable,
                            device: str, shape: tuple) -> BenchmarkResult:
        """
        Benchmark a single transform using torch.utils.benchmark.Timer.

        Timer handles warmup, adaptive iteration count, and GPU sync.
        """
        is_gpu = device == 'cuda'

        result = BenchmarkResult(
            transform_name=type(transform).__name__,
            device=device,
            shape=shape,
            num_threads=self.num_threads,
        )

        # Pre-create template data; clone per iteration so transforms
        # that modify in-place get a fresh copy without torch.rand overhead
        template_data = data_dict_fn(shape, 42)

        glob = {
            'transform': transform,
            'template_data': template_data,
            'torch': torch,
        }

        stmt = """\
data = {k: v.clone() for k, v in template_data.items()}
with torch.no_grad():
    transform(**data)
"""

        timer = bench.Timer(
            stmt=stmt,
            globals=glob,
            num_threads=self.num_threads,
            label=type(transform).__name__,
            sub_label=f"{device} {'x'.join(str(s) for s in shape)}",
        )

        if is_gpu:
            torch.cuda.reset_peak_memory_stats()

        measurement = timer.blocked_autorange(min_run_time=self.min_run_time)

        result.median_ms = measurement.median * 1000.0
        result.iqr_ms = measurement.iqr * 1000.0
        result.mean_ms = measurement.mean * 1000.0

        if is_gpu:
            result.gpu_mem_peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

        return result

    @staticmethod
    def print_table(all_results: List[BenchmarkResult]):
        """Pretty-print results table."""
        header = f"{'Transform':<35} | {'Shape':<14} | {'Device':<6} | {'Median(ms)':>10} | {'IQR(ms)':>8} | {'Peak GPU(MB)':>12} | {'Speedup':>8}"
        sep = "-" * len(header)
        print(sep)
        print(header)
        print(sep)

        # Group by (transform, shape) to compute speedup
        groups = {}
        for r in all_results:
            key = (r.transform_name, r.shape)
            groups.setdefault(key, {})[r.device] = r

        for (name, shape), devs in groups.items():
            shape_str = ",".join(str(s) for s in shape)
            for dev in ['cpu', 'cuda']:
                if dev not in devs:
                    continue
                r = devs[dev]
                peak = f"{r.gpu_mem_peak_mb:.1f}" if r.gpu_mem_peak_mb is not None else "-"
                if dev == 'cuda' and 'cpu' in devs:
                    speedup = f"{devs['cpu'].median_ms / r.median_ms:.1f}x"
                else:
                    speedup = "-"
                print(f"{name:<35} | {shape_str:<14} | {dev:<6} | {r.median_ms:>10.1f} | {r.iqr_ms:>8.1f} | {peak:>12} | {speedup:>8}")
        print(sep)

    @staticmethod
    def save_json(all_results: List[BenchmarkResult], path: str):
        """Save results as JSON for tracking over time."""
        data = []
        for r in all_results:
            d = asdict(r)
            d['shape'] = list(d['shape'])
            data.append(d)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
