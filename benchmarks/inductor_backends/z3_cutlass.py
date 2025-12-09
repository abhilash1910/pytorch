import os
import sys
os.environ["TORCH_LOGS"] = "inductor"
import csv
import itertools
import logging
import time
from abc import abstractmethod
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Optional, List, Dict
from tabulate import tabulate
from tqdm import tqdm
from triton.testing import do_bench

import torch
from torch._inductor import config as inductor_config
from torch.testing._internal.inductor_utils import _quantize_rowwise

try:
    from torch._inductor.fx_passes.pad_mm_z3 import (
        get_verifier as get_pad_mm_verifier,
        DimensionVerifier,
        AlignmentVerifier,
        PaddingLengthVerifier,
    )
    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False
    print("Warning: Z3 pad_mm verifier not available")

log: logging.Logger = logging.getLogger(__name__)

inductor_config.autotune_num_choices_displayed = None
inductor_config.autotune_local_cache = False

USE_FAST_ACCUM = True

UNITS = {
    "name": "",
    "forward_time": " (us)",
    "teraflops": " (TFLOPS)",
    "compilation_time": " (s)",
}

PERF_OVER_ATEN_STR: str = "perf_over_aten (%)"

OP_NAMES = [
    "mm",
]

SHAPES = [
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (8192, 8192, 8192),
]

BATCH_SIZES = [
    8,
]

DTYPES = [
    torch.float16,
    torch.bfloat16,
]

ENABLE_PERSISTENT_TMA_MATMULS = [
    False,
    True,
]

CUTLASS_INSTANTIATION_LEVELS = [
    "0",
    "3332",
]

z3_results: List[Dict[str, Any]] = []


def benchmark_torch_function_in_microseconds(func: Callable, *args, **kwargs) -> float:
    return do_bench(lambda: func(*args, **kwargs), warmup=100, rep=10000) * 1e3


def verify_mm_dimensions_z3(M: int, K: int, N: int, dtype: torch.dtype) -> Dict[str, Any]:
    result = {
        'dims_valid': False,
        'alignment_valid': False,
        'padding_m': 0,
        'padding_k': 0,
        'padding_n': 0,
        'issues': []
    }
    
    if not Z3_AVAILABLE:
        return result
    
    try:
        alignment = 8 if dtype in [torch.float16, torch.bfloat16] else 4
        
        valid, msg = DimensionVerifier.verify_mm_dims(M, K, K, N)
        result['dims_valid'] = valid
        if not valid:
            result['issues'].append(f"Dimension check failed: {msg}")
        
        valid, msg = AlignmentVerifier.verify_alignment_size(dtype, alignment)
        result['alignment_valid'] = valid
        if not valid:
            result['issues'].append(f"Alignment check failed: {msg}")
        
        m_pad = (alignment - (M % alignment)) % alignment
        k_pad = (alignment - (K % alignment)) % alignment
        n_pad = (alignment - (N % alignment)) % alignment
        
        result['padding_m'] = m_pad
        result['padding_k'] = k_pad
        result['padding_n'] = n_pad
        
        if m_pad > 0:
            valid, msg = PaddingLengthVerifier.verify_padded_length(M, alignment, m_pad)
            if not valid:
                result['issues'].append(f"M padding: {msg}")
        
        if k_pad > 0:
            valid, msg = PaddingLengthVerifier.verify_padded_length(K, alignment, k_pad)
            if not valid:
                result['issues'].append(f"K padding: {msg}")
        
        if n_pad > 0:
            valid, msg = PaddingLengthVerifier.verify_padded_length(N, alignment, n_pad)
            if not valid:
                result['issues'].append(f"N padding: {msg}")
        
        verifier = get_pad_mm_verifier()
        mm_verified = verifier.verify_mm_padding(M, K, N, m_pad, k_pad, n_pad, dtype, alignment)
        result['mm_verified'] = mm_verified
        
    except Exception as e:
        result['issues'].append(f"Error: {str(e)}")
    
    return result


@dataclass(frozen=True, kw_only=True)
class ExperimentConfig:
    max_autotune: bool = True
    coordinate_descent_tuning: bool = True
    max_autotune_gemm_backends: str = "ATEN"

    @abstractmethod
    def name(self) -> str:
        pass

    def to_options(self) -> dict[str, Any]:
        return {
            "max_autotune": self.max_autotune,
            "coordinate_descent_tuning": self.coordinate_descent_tuning,
            "max_autotune_gemm_backends": self.max_autotune_gemm_backends,
        }


@dataclass(frozen=True, kw_only=True)
class AtenExperimentConfig(ExperimentConfig):
    def name(self) -> str:
        return "aten"


@dataclass(frozen=True, kw_only=True)
class CutlassExperimentConfig(ExperimentConfig):
    cutlass_instantiation_level: str

    def name(self) -> str:
        level_name = (
            self.cutlass_instantiation_level
            if self.cutlass_instantiation_level != "0"
            else "default"
        )
        return f"cutlass_lvl_{level_name}"

    def to_options(self) -> dict[str, Any]:
        return {
            **super().to_options(),
            "cuda.cutlass_instantiation_level": self.cutlass_instantiation_level,
        }


@dataclass(frozen=True, kw_only=True)
class TritonExperimentConfig(ExperimentConfig):
    enable_persistent_tma_matmul: bool = False

    def name(self) -> str:
        if self.enable_persistent_tma_matmul:
            return "triton_persistent_tma"
        else:
            return "triton"

    def to_options(self) -> dict[str, Any]:
        return {
            **super().to_options(),
            "triton.enable_persistent_tma_matmul": self.enable_persistent_tma_matmul,
        }


@dataclass(frozen=True, kw_only=True)
class ExperimentGroupConfig:
    op_name: str
    shape: tuple[int, int, int]
    dtype: torch.dtype
    batch_size: int
    experiments: list[ExperimentConfig] = field(default_factory=list)

    def name(self) -> str:
        M, N, K = self.shape
        B = self.batch_size
        sizes = (
            f"(BS: {B}, {M}x{K}, {K}x{N})"
            if self.op_name == "bmm"
            else f"({M}x{K}, {K}x{N})"
        )
        return f"{self.op_name} {sizes} {self.dtype}"


@dataclass(frozen=True, kw_only=True)
class ExperimentResults:
    name: str
    forward_time: float
    teraflops: float
    compilation_time: float

    def asdict(self):
        return asdict(self)


@dataclass(frozen=True, kw_only=True)
class ExperimentGroup:
    config: ExperimentGroupConfig
    results: list[ExperimentResults] = field(default_factory=list)


def get_inputs(config: ExperimentGroupConfig) -> tuple[torch.Tensor, ...]:
    op_name = config.op_name
    M, N, K = config.shape
    batch_size = config.batch_size
    dtype = config.dtype
    device = torch.device("cuda")

    if op_name == "mm":
        A = torch.randn(M, K, dtype=dtype, device=device)
        B = torch.randn(N, K, dtype=dtype, device=device).t()
        return A, B
    elif op_name == "addmm":
        A = torch.randn(M, K, dtype=dtype, device=device)
        B = torch.randn(N, K, dtype=dtype, device=device).t()
        C = torch.randn(N, dtype=dtype, device=device)
        return C, A, B
    elif op_name == "bmm":
        A = torch.randn(batch_size, M, K, dtype=dtype, device=device)
        B = torch.randn(batch_size, N, K, dtype=dtype, device=device).permute(0, 2, 1)
        return A, B
    elif op_name == "_scaled_mm":
        if dtype != torch.float8_e4m3fn:
            raise ValueError(f"_scaled_mm only supports fp8e4m3, got {dtype}")
        input_dtype = torch.bfloat16
        x = torch.randn(M, K, dtype=input_dtype, device=device)
        w = torch.randn(N, K, dtype=input_dtype, device=device)
        w_fp8, w_inverse_scale = _quantize_rowwise(w, dtype)
        w_t_fp8 = w_fp8.t()
        w_inverse_scale = w_inverse_scale.t()
        x_fp8, x_inverse_scale = _quantize_rowwise(x, dtype)
        return (
            x_fp8,
            w_t_fp8,
            x_inverse_scale,
            w_inverse_scale,
            None,
            None,
            torch.bfloat16,
            USE_FAST_ACCUM,
        )
    else:
        raise ValueError(f"Unknown op {op_name}")


def run_single_experiment_group(group_config: ExperimentGroupConfig) -> list[ExperimentResults]:
    inputs = get_inputs(group_config)
    op = getattr(torch, group_config.op_name)
    results = []
    
    M, N, K = group_config.shape
    dtype = group_config.dtype
    
    z3_verify = verify_mm_dimensions_z3(M, K, N, dtype)
    
    for config in group_config.experiments:
        torch._dynamo.reset()
        torch._inductor.utils.clear_caches()
        
        compiled_op = torch.compile(op, options=config.to_options())
        
        start_time = time.perf_counter()
        try:
            _ = compiled_op(*inputs)
        except Exception as e:
            import traceback
            log.warning(
                f"Benchmark config {config.name()} failed: {e}, "
                f"traceback: {traceback.format_exc()}"
            )
            results.append(
                ExperimentResults(
                    name=config.name(),
                    forward_time=float("inf"),
                    teraflops=0.0,
                    compilation_time=float("inf"),
                )
            )
            continue
        
        compilation_time = time.perf_counter() - start_time
        
        forward_time = benchmark_torch_function_in_microseconds(compiled_op, *inputs)
        
        flops = calculate_flops(group_config.op_name, group_config.shape, group_config.batch_size)
        teraflops = flops / (forward_time * 1e-6) / 1e12
        
        results.append(
            ExperimentResults(
                name=config.name(),
                forward_time=forward_time,
                teraflops=teraflops,
                compilation_time=compilation_time,
            )
        )
        
        z3_results.append({
            'op': group_config.op_name,
            'backend': config.name(),
            'M': M,
            'N': N,
            'K': K,
            'dtype': str(dtype).replace('torch.', ''),
            'forward_time_us': forward_time,
            'teraflops': teraflops,
            'z3_dims_valid': z3_verify['dims_valid'],
            'z3_alignment_valid': z3_verify['alignment_valid'],
            'z3_mm_verified': z3_verify.get('mm_verified', False),
            'z3_padding_m': z3_verify['padding_m'],
            'z3_padding_k': z3_verify['padding_k'],
            'z3_padding_n': z3_verify['padding_n'],
            'z3_issues': len(z3_verify['issues']),
        })
    
    print(f"[Z3] {group_config.op_name} ({M}x{K})x({K}x{N}) {dtype}: dims={z3_verify['dims_valid']}, align={z3_verify['alignment_valid']}, pad=({z3_verify['padding_m']},{z3_verify['padding_k']},{z3_verify['padding_n']})")
    
    return results


def generate_experiment_groups(
    op_names: list[str],
    shapes: list[tuple[int, int, int]],
    dtypes: list[torch.dtype],
    enable_persistent_tma_matmuls: list[bool],
    cutlass_instantiation_levels: list[str],
    batch_sizes: list[int],
) -> list[ExperimentGroupConfig]:
    groups = []
    for (op_name, shape, dtype, batch_size) in itertools.product(op_names, shapes, dtypes, batch_sizes):
        group = ExperimentGroupConfig(
            op_name=op_name,
            shape=shape,
            dtype=dtype,
            batch_size=batch_size,
        )
        experiments = generate_experiment_configs(enable_persistent_tma_matmuls, cutlass_instantiation_levels)
        group.experiments.extend(experiments)
        groups.append(group)
    return groups


def generate_experiment_configs(
    enable_persistent_tma_matmuls: list[bool], cutlass_instantiation_levels: list[str]
) -> list[ExperimentConfig]:
    configs = []
    
    configs.append(AtenExperimentConfig(max_autotune_gemm_backends="ATEN"))
    
    for enable_persistent_tma_matmul in enable_persistent_tma_matmuls:
        configs.append(
            TritonExperimentConfig(
                max_autotune_gemm_backends="TRITON",
                enable_persistent_tma_matmul=enable_persistent_tma_matmul,
            )
        )
    
    for cutlass_instantiation_level in cutlass_instantiation_levels:
        configs.append(
            CutlassExperimentConfig(
                max_autotune_gemm_backends="CUTLASS",
                cutlass_instantiation_level=cutlass_instantiation_level,
            )
        )
    
    return configs


def calculate_table_data(results: list[ExperimentResults]) -> dict:
    table_data = defaultdict(list)
    aten_perf: Optional[float] = None
    
    for experiment_result in results:
        for key, value in experiment_result.asdict().items():
            assert key in UNITS, f"Unknown key {key}"
            table_data[key + UNITS[key]].append(value)
        if experiment_result.name == "aten":
            aten_perf = experiment_result.forward_time
            table_data[PERF_OVER_ATEN_STR].append("NA")
        elif aten_perf is not None:
            perf_over_aten = (experiment_result.forward_time - aten_perf) / aten_perf * 100
            table_data[PERF_OVER_ATEN_STR].append(perf_over_aten)
        else:
            table_data[PERF_OVER_ATEN_STR].append("NA")
    
    return table_data


def calculate_flops(op_name: str, shape: tuple[int, int, int], batch_size: int) -> int:
    M, N, K = shape
    if op_name == "bmm":
        return 2 * batch_size * M * N * K
    elif op_name == "addmm":
        return 2 * M * N * K + M * N
    elif op_name == "_scaled_mm":
        return 2 * M * N * K
    else:
        return 2 * M * N * K


def get_printable_results(experiment_groups: list[ExperimentGroup]) -> list[str]:
    edge_over_aten = defaultdict(list)
    output = []
    
    for experiment_group in experiment_groups:
        group_config_name = experiment_group.config.name()
        output.append(f"\nExperiment group: {group_config_name}")
        table_data = calculate_table_data(experiment_group.results)
        for name, edge in zip(table_data["name"], table_data[PERF_OVER_ATEN_STR]):
            edge_over_aten[name].append(edge)
        output.append(tabulate(table_data, headers="keys", tablefmt="pretty", floatfmt=".3f"))
    
    if "aten" in edge_over_aten:
        output.append("\nAverage edge over aten (max(-edge, 0), higher is better):")
        for name in edge_over_aten:
            if name != "aten":
                values = [max(-v, 0.0) for v in edge_over_aten[name] if v != float("inf") and v != "NA"]
                valid_count = len(values)
                average_edge = sum(values) / valid_count if values else "No valid data"
                output.append(f"{name}: {average_edge} (from {valid_count} valid values)")
        output.append("\n")
    
    return "\n".join(output)


def output_z3_results(output_file: str = "gemm_benchmark_z3.csv"):
    if z3_results:
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=z3_results[0].keys())
            writer.writeheader()
            writer.writerows(z3_results)
        print(f"\nZ3 verification results saved to: {output_file}")
        
        try:
            import pandas as pd
            df = pd.DataFrame(z3_results)
            xlsx_file = output_file.replace('.csv', '.xlsx')
            df.to_excel(xlsx_file, index=False)
            print(f"Z3 verification results saved to: {xlsx_file}")
        except ImportError:
            pass


def print_z3_summary():
    print("\n" + "=" * 80)
    print("Z3 VERIFICATION SUMMARY")
    print("=" * 80)
    
    total_verified = sum(1 for r in z3_results if r.get('z3_mm_verified', False))
    total_dims_valid = sum(1 for r in z3_results if r.get('z3_dims_valid', False))
    total_align_valid = sum(1 for r in z3_results if r.get('z3_alignment_valid', False))
    total_issues = sum(r.get('z3_issues', 0) for r in z3_results)
    total = len(z3_results)
    
    shapes_checked = set()
    for r in z3_results:
        key = (r['M'], r['N'], r['K'], r['dtype'])
        shapes_checked.add(key)
    
    print(f"\nShapes verified: {len(shapes_checked)}")
    for M, N, K, dtype in sorted(shapes_checked):
        matching = [r for r in z3_results if r['M'] == M and r['N'] == N and r['K'] == K and r['dtype'] == dtype]
        verified = all(r.get('z3_mm_verified', False) for r in matching)
        pad_m = matching[0]['z3_padding_m'] if matching else 0
        pad_k = matching[0]['z3_padding_k'] if matching else 0
        pad_n = matching[0]['z3_padding_n'] if matching else 0
        status = "PASS" if verified else "FAIL"
        print(f"  [{status}] ({M}x{K})x({K}x{N}) {dtype}: padding=({pad_m},{pad_k},{pad_n})")
    
    print("-" * 80)
    print(f"Total experiments: {total}")
    print(f"Dimensions valid: {total_dims_valid}/{total}")
    print(f"Alignment valid: {total_align_valid}/{total}")
    print(f"MM padding verified: {total_verified}/{total}")
    print(f"Issues found: {total_issues}")
    print("=" * 80)
    
    if total_verified == total and total > 0:
        print("\n[SUCCESS] All GEMM operations Z3 verified!")
    elif total_issues > 0:
        print(f"\n[WARNING] {total_issues} Z3 verification issues found")


def main():
    seed = 123
    torch.manual_seed(seed)
    
    print("=" * 80)
    print("GEMM BENCHMARK WITH Z3 VERIFICATION")
    print("=" * 80)
    print(f"Z3 Verification: {'ENABLED' if Z3_AVAILABLE else 'DISABLED'}")
    
    results = []
    log.info("Starting benchmarking...")
    
    configs = list(
        generate_experiment_groups(
            OP_NAMES,
            SHAPES,
            DTYPES,
            ENABLE_PERSISTENT_TMA_MATMULS,
            CUTLASS_INSTANTIATION_LEVELS,
            BATCH_SIZES,
        )
    )
    
    for i, group_config in enumerate(tqdm(configs)):
        group_results = run_single_experiment_group(group_config)
        results.append(ExperimentGroup(config=group_config, results=group_results))
        sys.stderr.write(
            f"\nINTERMEDIATE results: {i + 1}/{len(configs)} \n"
            + get_printable_results(results)
        )
    
    print("\nFINAL results...")
    print(get_printable_results(results))
    
    output_z3_results()
    print_z3_summary()


if __name__ == "__main__":
    main()
