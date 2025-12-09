import itertools
import os
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from typing import Callable, Optional
from tabulate import tabulate
from tqdm import tqdm
import torch
import torch.utils.benchmark as benchmark
from torch._inductor.utils import do_bench_using_profiling
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.nn.functional import scaled_dot_product_attention

# Import Z3 SDPA Verifier
try:
    from torch._inductor.fx_passes.fuse_attention_z3 import (
        get_verifier as get_sdpa_verifier,
    )
    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False
    print("Warning: Z3 SDPA verifier not available")


def benchmark_cuda_function_in_microseconds(func: Callable, *args, **kwargs) -> float:
    """Thin wrapper around do_bench_using_profiling"""
    def no_args():
        func(*args, **kwargs)
    time = do_bench_using_profiling(no_args)
    return time * 1e3


def benchmark_torch_function_in_microseconds(func: Callable, *args, **kwargs) -> float:
    # warmup
    for _ in range(5):
        func(*args, **kwargs)
    t0 = benchmark.Timer(
        stmt="func(*args, **kwargs)",
        globals={"args": args, "kwargs": kwargs, "func": func},
    )
    return t0.adaptive_autorange(min_run_time=0.1).median * 1e6


@dataclass(frozen=True)
class ExperimentConfig:
    batch_size: int
    num_heads: int
    q_seq_len: int
    kv_seq_len: int
    embed_dim: int
    is_causal: bool
    dtype: torch.dtype
    backend: SDPBackend
    device: torch.device = torch.device("cuda")

    @property
    def head_dim(self) -> int:
        return self.embed_dim // self.num_heads

    def asdict(self):
        dict_obj = asdict(self)
        dict_obj["head_dim"] = self.head_dim
        return dict_obj


@dataclass(frozen=True)
class Z3VerificationResult:
    """Z3 verification results for SDPA pattern"""
    verified: bool
    pattern_num: int
    patterns_verified: int
    patterns_total: int
    message: str

    def asdict(self):
        return asdict(self)


@dataclass(frozen=True)
class ExperimentResults:
    forward_time: float  # microseconds
    backward_time: float  # microseconds
    forward_tflops: float
    backward_tflops: float
    z3_verified: bool
    z3_patterns_verified: int
    z3_patterns_total: int

    def asdict(self):
        return asdict(self)


@dataclass(frozen=True)
class Experiment:
    config: ExperimentConfig
    results: ExperimentResults

    def asdict(self):
        dict1 = self.config.asdict()
        dict2 = self.results.asdict()
        return {**dict1, **dict2}


def calculate_tflops(
    config: ExperimentConfig,
    time_us: float,
    is_backward: bool = False,
    sparsity: float = 0.0,
) -> float:
    """
    Calculate TFLOPS for scaled dot product attention.
    """
    B = config.batch_size
    H = config.num_heads
    M = config.q_seq_len
    N = config.kv_seq_len
    D = config.head_dim

    density = 1.0 - sparsity

    qk_flops = M * N * D * 2
    softmax_flops = M * N * 2
    av_flops = M * N * D * 2
    total_flops = B * H * (qk_flops + softmax_flops + av_flops)
    total_flops *= density

    if is_backward:
        total_flops *= 2.5

    tflops = total_flops / (time_us * 1e-6) / 1e12
    return tflops


def verify_sdpa_with_z3(config: ExperimentConfig) -> Z3VerificationResult:
    """
    Verify SDPA pattern using Z3 SMT solver.
    
    Args:
        config: The experiment configuration with Q, K, V shapes
        
    Returns:
        Z3VerificationResult with verification status
    """
    if not Z3_AVAILABLE:
        return Z3VerificationResult(
            verified=False,
            pattern_num=0,
            patterns_verified=0,
            patterns_total=24,
            message="Z3 not available"
        )
    
    try:
        verifier = get_sdpa_verifier()
        
        # Build shapes from config: (batch, heads, seq_len, head_dim)
        q_shape = (config.batch_size, config.num_heads, config.q_seq_len, config.head_dim)
        k_shape = (config.batch_size, config.num_heads, config.kv_seq_len, config.head_dim)
        v_shape = (config.batch_size, config.num_heads, config.kv_seq_len, config.head_dim)
        inv_scale = float(config.head_dim ** 0.5)
        
        # Verify all 24 SDPA patterns
        verified_patterns = []
        total_patterns = 24
        
        for pattern_num in range(1, total_patterns + 1):
            try:
                result = verifier.verify_pattern(
                    pattern_num,
                    q_shape=q_shape,
                    k_shape=k_shape,
                    v_shape=v_shape,
                    inv_scale=inv_scale,
                )
                if result:
                    verified_patterns.append(pattern_num)
            except Exception:
                pass
        
        patterns_verified = len(verified_patterns)
        overall_verified = patterns_verified > 0
        
        # Get the first verified pattern number (Pattern 1 is most common)
        first_pattern = verified_patterns[0] if verified_patterns else 0
        
        message = f"{patterns_verified}/{total_patterns} patterns verified"
        if overall_verified:
            message += f" (Pattern {first_pattern} verified)"
        
        return Z3VerificationResult(
            verified=overall_verified,
            pattern_num=first_pattern,
            patterns_verified=patterns_verified,
            patterns_total=total_patterns,
            message=message
        )
        
    except Exception as e:
        return Z3VerificationResult(
            verified=False,
            pattern_num=0,
            patterns_verified=0,
            patterns_total=24,
            message=f"Error: {str(e)[:50]}"
        )


def get_input(
    config: ExperimentConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q = torch.randn(
        (config.batch_size, config.num_heads, config.q_seq_len, config.head_dim),
        dtype=config.dtype,
        device=config.device,
        requires_grad=True,
    )
    k = torch.randn(
        (config.batch_size, config.num_heads, config.kv_seq_len, config.head_dim),
        dtype=config.dtype,
        device=config.device,
        requires_grad=True,
    )
    v = torch.randn(
        (config.batch_size, config.num_heads, config.kv_seq_len, config.head_dim),
        dtype=config.dtype,
        device=config.device,
        requires_grad=True,
    )
    return q, k, v


def run_single_experiment(config: ExperimentConfig) -> ExperimentResults:
    q, k, v = get_input(config)
    is_causal = config.is_causal
    
    context = (
        sdpa_kernel(config.backend) if config.backend is not None else nullcontext()
    )
    
    with context:
        forward_time = benchmark_cuda_function_in_microseconds(
            scaled_dot_product_attention,
            q,
            k,
            v,
            is_causal=is_causal,
            attn_mask=None,
        )
        out_torch = scaled_dot_product_attention(
            q, k, v, is_causal=is_causal, attn_mask=None
        )
        d_out = torch.randn_like(out_torch)
        backward_time = benchmark_cuda_function_in_microseconds(
            out_torch.backward, d_out, retain_graph=True
        )

    sparsity = 0.5 if is_causal else 0.0
    forward_tflops = calculate_tflops(config, forward_time, sparsity=sparsity)
    backward_tflops = calculate_tflops(
        config, backward_time, is_backward=True, sparsity=sparsity
    )
    
    # Z3 Verification
    z3_result = verify_sdpa_with_z3(config)
    
    return ExperimentResults(
        forward_time=forward_time,
        backward_time=backward_time,
        forward_tflops=forward_tflops,
        backward_tflops=backward_tflops,
        z3_verified=z3_result.verified,
        z3_patterns_verified=z3_result.patterns_verified,
        z3_patterns_total=z3_result.patterns_total,
    )


def print_results(experiments: list[Experiment]):
    table_data = defaultdict(list)
    for experiment in experiments:
        for key, value in experiment.asdict().items():
            table_data[key].append(value)
    
    del table_data["device"]
    if table_data["backend"][0] is None:
        del table_data["backend"]
    
    print(tabulate(table_data, headers="keys", tablefmt="pretty", floatfmt=".3f"))


def print_z3_summary(experiments: list[Experiment]):
    """Print Z3 verification summary"""
    print("\n" + "=" * 70)
    print("Z3 SDPA VERIFICATION SUMMARY")
    print("=" * 70)
    
    total_configs = len(experiments)
    verified_configs = sum(1 for e in experiments if e.results.z3_verified)
    
    # Group by shape
    shape_results = defaultdict(lambda: {'verified': 0, 'total': 0})
    for exp in experiments:
        cfg = exp.config
        shape_key = f"B={cfg.batch_size}, H={cfg.num_heads}, Q={cfg.q_seq_len}, K={cfg.kv_seq_len}, D={cfg.head_dim}"
        shape_results[shape_key]['total'] += 1
        if exp.results.z3_verified:
            shape_results[shape_key]['verified'] += 1
    
    print("\nVerification by shape:")
    for shape, counts in shape_results.items():
        status = "OK" if counts['verified'] == counts['total'] else "NOT OK"
        print(f"  {status} {shape}: {counts['verified']}/{counts['total']} verified")
    
    print("-" * 70)
    print(f"TOTAL: {verified_configs}/{total_configs} configurations verified ({100*verified_configs/total_configs:.1f}%)")
    print("=" * 70)
    

def write_results_to_csv(
    experiments: list[Experiment], output_dir: str = "benchmark_results"
):
    """Write experiment results to CSV with Z3 verification columns"""
    import csv
    from datetime import datetime

    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_dir, f"sdpa_benchmark_z3_{timestamp}.csv")

    if not experiments:
        return

    fieldnames = list(experiments[0].asdict().keys())
    if "device" in fieldnames:
        fieldnames.remove("device")

    with open(filename, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for experiment in experiments:
            row = experiment.asdict()
            if "device" in row:
                del row["device"]
            writer.writerow(row)

    print(f"\nResults written to: {filename}")
    
    # Also write XLSX if pandas available
    try:
        import pandas as pd
        xlsx_filename = os.path.join(output_dir, f"sdpa_benchmark_z3_{timestamp}.xlsx")
        df = pd.DataFrame([e.asdict() for e in experiments])
        if "device" in df.columns:
            df = df.drop(columns=["device"])
        df.to_excel(xlsx_filename, index=False)
        print(f"Results also written to: {xlsx_filename}")
    except ImportError:
        pass


def generate_experiment_configs() -> list[ExperimentConfig]:
    batch_sizes = [1, 8, 16]
    num_heads = [16]
    q_kv_seq_lens = [(128, 128), (256, 256), (512, 512), (1024, 1024), (8192, 8192)]
    embed_dims = [2048]
    backends = [None]
    dtypes = [torch.bfloat16]
    is_causal = [True, False]

    all_configs = []
    for (
        bsz,
        heads,
        (q_seq_len, kv_seq_len),
        embed_dim,
        causal,
        dtype,
        backend,
    ) in itertools.product(
        batch_sizes, num_heads, q_kv_seq_lens, embed_dims, is_causal, dtypes, backends
    ):
        all_configs.append(
            ExperimentConfig(
                batch_size=bsz,
                num_heads=heads,
                q_seq_len=q_seq_len,
                kv_seq_len=kv_seq_len,
                embed_dim=embed_dim,
                is_causal=causal,
                dtype=dtype,
                backend=backend,
            )
        )
    return all_configs


def main():
    seed = 123
    torch.manual_seed(seed)
    
    print("=" * 70)
    print("SDPA BENCHMARK WITH Z3 FORMAL VERIFICATION")
    print("=" * 70)
    print(f"Z3 Verification: {'ENABLED' if Z3_AVAILABLE else 'DISABLED'}")
    print()
    
    results = []
    configs = generate_experiment_configs()
    
    print(f"Running {len(configs)} experiments...")
    for config in tqdm(configs):
        results.append(Experiment(config, run_single_experiment(config)))
    
    print_results(results)
    print_z3_summary(results)
    write_results_to_csv(results, "benchmark_results")


if __name__ == "__main__":
    main()
