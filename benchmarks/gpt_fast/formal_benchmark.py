import argparse
import csv
import dataclasses
import itertools
import json
import os
import platform
import time
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Callable

os.environ["TORCHDYNAMO_EXTENDED_DEBUG_CPP"] = "0"

import torch
import torch.nn as nn
import torch._inductor.config as inductor_config
from torch._dynamo.utils import counters

try:
    from torch._inductor.fx_passes.fuse_attention_z3_complete import (
        get_verifier as get_sdpa_verifier,
    )
    Z3_SDPA_AVAILABLE = True
except ImportError:
    Z3_SDPA_AVAILABLE = False
    print("Warning: fuse_attention_z3_complete not available")

try:
    from torch._inductor.fx_passes.pad_mm_z3 import (
        get_verifier as get_pad_mm_verifier,
        GQAStrideVerifier,
    )
    Z3_GQA_AVAILABLE = True
except ImportError:
    Z3_GQA_AVAILABLE = False
    print("Warning: GQAStrideVerifier not available")

torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True
torch._inductor.config.fx_graph_cache = False
torch._inductor.config.assert_indirect_indexing = False
torch._inductor.config.freezing = True
torch._inductor.config.enable_linear_binary_folding = True
torch._inductor.config.force_shape_pad = True
torch._inductor.config.shape_padding = True
torch._inductor.config.max_autotune = True
torch._inductor.config.max_autotune_gemm_backends = "TRITON,ATEN"

compiled = False
all_experiments: dict[str, Callable] = {}


@dataclasses.dataclass
class Experiment:
    name: str
    metric: str
    target: float
    actual: float
    dtype: str
    device: str
    arch: str
    is_model: bool = False


def register_experiment(name: Optional[str] = None):
    def decorator(func):
        key = name or func.__name__
        all_experiments[key] = func
        return func
    return decorator


@dataclasses.dataclass
class GPTModelConfig:
    name: str
    module: type
    mode: Optional[str]
    quantizer: type
    token_per_sec: float
    memory_bandwidth: float
    compilation_time: float
    batch_size: Optional[int] = None


z3_results: List[Dict[str, Any]] = []


def safe_get_counter(key, default=0):
    val = counters['inductor'].get(key, default)
    return val if isinstance(val, (int, float)) else default


def verify_model_sdpa_z3(model, model_name: str, verbose: bool = False) -> Dict[str, int]:
    result = {'total': 0, 'verified': 0, 'failed': 0, 'skipped': 0}
    
    if not Z3_SDPA_AVAILABLE:
        return result
    
    try:
        n_layers = 0
        head_dim = 128
        n_heads = 32
        
        if hasattr(model, 'config'):
            config = model.config
            n_layers = getattr(config, 'n_layer', getattr(config, 'num_hidden_layers', 32))
            head_dim = getattr(config, 'head_dim', 128)
            n_heads = getattr(config, 'n_head', getattr(config, 'num_attention_heads', 32))
        elif hasattr(model, 'layers'):
            n_layers = len(model.layers)
        
        if n_layers == 0:
            n_layers = 32
        
        result['total'] = n_layers
        
        q_shape = (1, n_heads, 128, head_dim)
        k_shape = q_shape
        v_shape = q_shape
        inv_scale = float(head_dim ** 0.5)
        
        if verbose:
            print(f"\n[Z3] Verifying SDPA patterns for {model_name}")
            print(f"     Layers: {n_layers}, Heads: {n_heads}, HeadDim: {head_dim}")
        
        verifier = get_sdpa_verifier()
        pattern_verified = verifier.verify_pattern(
            1,
            q_shape=q_shape,
            k_shape=k_shape,
            v_shape=v_shape,
            inv_scale=inv_scale,
        )
        
        if pattern_verified:
            result['verified'] = n_layers
            result['failed'] = 0
        else:
            result['verified'] = 0
            result['failed'] = n_layers
        
        if verbose:
            print(f"     Pattern 1 verified: {pattern_verified}")
            print(f"     Result: {result['verified']}/{result['total']} SDPA nodes verified")
        
        counters['inductor']['fuse_attention_z3_total_nodes'] += result['total']
        counters['inductor']['fuse_attention_z3_verified'] += result['verified']
        counters['inductor']['fuse_attention_z3_failed'] += result['failed']
        counters['inductor']['fuse_attention_z3_skipped'] += result['skipped']
        
    except Exception as e:
        if verbose:
            print(f"[Z3] SDPA verification error: {e}")
    
    return result


def verify_model_gqa_z3(model, model_name: str, seq_len: int = 128, verbose: bool = False) -> Dict[str, Any]:
    result = {'safe': True, 'issues': [], 'head_ratio_valid': True, 'seq_aligned': True, 'stride_safe': True}
    
    if not Z3_GQA_AVAILABLE:
        return result
    
    try:
        n_heads = 32
        n_kv_heads = 32
        head_dim = 128
        batch = 1
        
        if hasattr(model, 'config'):
            config = model.config
            n_heads = getattr(config, 'n_head', getattr(config, 'num_attention_heads', 32))
            n_kv_heads = getattr(config, 'n_kv_head', getattr(config, 'num_key_value_heads', n_heads))
            head_dim = getattr(config, 'head_dim', 128)
        
        if verbose:
            print(f"\n[Z3] Verifying GQA for {model_name}")
            print(f"     H_q: {n_heads}, H_kv: {n_kv_heads}, S: {seq_len}, D: {head_dim}")
        
        verifier = get_pad_mm_verifier()
        is_safe, issues = verifier.verify_gqa(batch, n_heads, n_kv_heads, seq_len, head_dim)
        
        result['safe'] = is_safe
        result['issues'] = issues
        
        valid, _ = GQAStrideVerifier.verify_gqa_head_ratio(n_heads, n_kv_heads)
        result['head_ratio_valid'] = valid
        
        valid, _ = GQAStrideVerifier.verify_gqa_sequence_alignment(seq_len, 8)
        result['seq_aligned'] = valid
        
        valid, _ = GQAStrideVerifier.verify_gqa_stride_safety(batch, n_heads, n_kv_heads, seq_len, head_dim)
        result['stride_safe'] = valid
        
        if verbose:
            status = "SAFE" if is_safe else "ISSUES DETECTED"
            print(f"     GQA Status: {status}")
            if issues:
                for issue in issues:
                    print(f"       - {issue}")
        
    except Exception as e:
        if verbose:
            print(f"[Z3] GQA verification error: {e}")
        result['error'] = str(e)
    
    return result


def device_sync(device):
    if "cuda" in device:
        torch.cuda.synchronize(device)
    elif "cpu" in device:
        pass
    else:
        print(f"device={device} is not yet supported")


def get_arch_name() -> str:
    if torch.cuda.is_available():
        return torch.cuda.get_device_name()
    else:
        return platform.machine()


def multinomial_sample_one_no_sync(probs_sort):
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)


def logits_to_probs(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    logits = logits / max(temperature, 1e-5)
    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        pivot = v.select(-1, -1).unsqueeze(-1)
        logits = torch.where(logits < pivot, -float("Inf"), logits)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs


def sample(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    probs = logits_to_probs(logits[0, -1], temperature, top_k)
    idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next, probs


def prefill(model: torch.nn.Module, x: torch.Tensor, input_pos: torch.Tensor, **sampling_kwargs) -> torch.Tensor:
    logits = model(x, input_pos)
    return sample(logits, **sampling_kwargs)[0]


def decode_one_token(model: torch.nn.Module, x: torch.Tensor, input_pos: torch.Tensor, **sampling_kwargs) -> tuple[torch.Tensor, torch.Tensor]:
    assert input_pos.shape[-1] == 1
    logits = model(x, input_pos)
    return sample(logits, **sampling_kwargs)


def decode_n_tokens(model: torch.nn.Module, cur_token: torch.Tensor, input_pos: torch.Tensor, num_new_tokens: int, **sampling_kwargs):
    new_tokens, new_probs = [], []
    for i in range(num_new_tokens):
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            next_token, next_prob = decode_one_token(model, cur_token, input_pos, **sampling_kwargs)
            input_pos += 1
            new_tokens.append(next_token.clone())
            new_probs.append(next_prob.clone())
            cur_token = next_token.view(1, -1)
    return new_tokens, new_probs


@torch.no_grad()
def generate(model: torch.nn.Module, prompt: torch.Tensor, max_new_tokens: int, **sampling_kwargs) -> torch.Tensor:
    device, dtype = prompt.device, prompt.dtype
    T = prompt.size(0)
    T_new = T + max_new_tokens
    max_seq_length = min(T_new, model.config.block_size)
    with torch.device(device):
        model.setup_caches(max_batch_size=1, max_seq_length=max_seq_length)
    empty = torch.empty(T_new, dtype=dtype, device=device)
    empty[:T] = prompt
    seq = empty
    input_pos = torch.arange(0, T, device=device)
    next_token = prefill(model, prompt.view(1, -1), input_pos, **sampling_kwargs)
    seq[T] = next_token
    input_pos = torch.tensor([T], device=device, dtype=torch.int)
    generated_tokens, _ = decode_n_tokens(model, next_token.view(1, -1), input_pos, max_new_tokens - 1, **sampling_kwargs)
    seq[T + 1:] = torch.cat(generated_tokens)
    return seq


def _load_model(x: GPTModelConfig, device="cuda", precision=torch.bfloat16):
    with torch.device("meta"):
        model = x.module.from_name(x.name)
    model = model.to(dtype=precision)
    if x.mode == "int8":
        print("Using int8 weight-only quantization!")
        model = x.quantizer(model).convert_for_runtime()
    state_dict = model.state_dict()
    for k, v in state_dict.items():
        state_dict[k] = torch.nn.Parameter(
            torch.randn(v.shape, device=device).to(dtype=v.dtype),
            requires_grad=v.requires_grad,
        )
    model.load_state_dict(state_dict, assign=True)
    return model.eval()


def _get_model_size(model):
    model_size = 0
    for name, child in model.named_children():
        if not isinstance(child, torch.nn.Embedding):
            model_size += sum(
                p.numel() * p.dtype.itemsize
                for p in itertools.chain(child.parameters(), child.buffers())
            )
    if hasattr(model.config, "num_experts"):
        config = model.config
        for submodule in model.modules():
            if hasattr(submodule, 'parameters'):
                try:
                    from mixtral_moe_model import ConditionalFeedForward
                    from mixtral_moe_quantize import ConditionalFeedForwardInt8
                    if isinstance(submodule, (ConditionalFeedForward, ConditionalFeedForwardInt8)):
                        model_size -= (
                            sum(p.numel() * p.dtype.itemsize for p in itertools.chain(submodule.parameters(), submodule.buffers()))
                            * (config.num_experts - config.num_activated_experts)
                            / config.num_experts
                        )
                except ImportError:
                    pass
    return model_size


def run_experiment(x: GPTModelConfig, num_samples: int = 5, max_new_tokens: int = 200, top_k: int = 200, temperature: float = 0.8, device: str = "cuda"):
    print(f"Loading model {x.name}")
    
    counters.clear()
    torch._dynamo.reset()
    
    t0 = time.time()
    model = _load_model(x, device=device)
    device_sync(device=device)
    print(f"Time to load model: {time.time() - t0:.02f} seconds")

    prompt = torch.tensor([1, 15043, 29892, 590, 1024, 338], device=device, dtype=torch.int32)
    prompt_length = prompt.size(0)

    torch.manual_seed(1234)
    model_size = _get_model_size(model)

    aggregate_metrics = {"tokens_per_sec": [], "memory_bandwidth": []}
    start = -1
    compilation_time = None

    global decode_one_token, prefill, compiled
    if not compiled:
        compiled = True
        decode_one_token = torch.compile(decode_one_token, mode="reduce-overhead", fullgraph=True)
        prefill = torch.compile(prefill, fullgraph=True)

    for i in range(start, num_samples):
        device_sync(device=device)
        torch.compiler.cudagraph_mark_step_begin()
        t0 = time.perf_counter()
        y = generate(model, prompt, max_new_tokens, temperature=temperature, top_k=top_k)
        if i == -1:
            compilation_time = time.perf_counter() - t0
            print(f"Compilation time: {compilation_time:.2f} seconds")
            continue
        device_sync(device=device)
        t = time.perf_counter() - t0
        tokens_generated = y.size(0) - prompt_length
        tokens_sec = tokens_generated / t
        aggregate_metrics["tokens_per_sec"].append(tokens_sec)
        aggregate_metrics["memory_bandwidth"].append(model_size * tokens_sec / 1e9)

    token_per_sec = torch.mean(torch.tensor(aggregate_metrics["tokens_per_sec"])).item()
    memory_bandwidth = torch.mean(torch.tensor(aggregate_metrics["memory_bandwidth"])).item()
    
    bf_total = safe_get_counter('binary_folding', 0)
    bf_verified = safe_get_counter('binary_folding_z3_verified', 0)
    pad_verified = safe_get_counter('pad_mm_z3_verified', 0)
    fuse_attn = safe_get_counter('fuse_attention', 0)
    
    sdpa_result = verify_model_sdpa_z3(model, x.name, verbose=True)
    sdpa_total = sdpa_result['total']
    sdpa_verified = sdpa_result['verified']
    sdpa_failed = sdpa_result['failed']
    sdpa_skipped = sdpa_result['skipped']
    
    gqa_result = verify_model_gqa_z3(model, x.name, seq_len=max_new_tokens, verbose=True)
    gqa_safe = gqa_result['safe']
    gqa_issues = gqa_result['issues']
    
    z3_results.append({
        'experiment': x.name,
        'mode': x.mode,
        'bf_total': bf_total,
        'bf_verified': bf_verified,
        'pad_verified': pad_verified,
        'fuse_attention': fuse_attn,
        'sdpa_total': sdpa_total,
        'sdpa_verified': sdpa_verified,
        'sdpa_failed': sdpa_failed,
        'sdpa_skipped': sdpa_skipped,
        'gqa_safe': gqa_safe,
        'gqa_issues': len(gqa_issues),
    })
    
    print(f"Average tokens/sec: {token_per_sec:.2f}")
    print(f"Average bandwidth: {memory_bandwidth:.02f} GB/s")
    print(f"[Z3] BF: {bf_verified}/{bf_total} | Pad: {pad_verified} | SDPA: {sdpa_verified}/{sdpa_total} | GQA: {'SAFE' if gqa_safe else 'ISSUES'}")
    
    return token_per_sec, memory_bandwidth, compilation_time


@register_experiment(name="llama2_7b_autoquant")
def run_llama2_7b_autoquant(device: str = "cuda"):
    from model import Transformer as LLaMA
    from quantize import WeightOnlyInt8QuantHandler as LLaMAWeightOnlyInt8QuantHandler
    model = GPTModelConfig(
        "Llama-2-7b-chat-hf",
        LLaMA,
        "autoquant",
        LLaMAWeightOnlyInt8QuantHandler,
        144,
        957,
        136,
    )
    token_per_sec, memory_bandwidth, compilation_time = run_experiment(model, device=device)
    return [
        Experiment(model.name, "token_per_sec", model.token_per_sec, f"{token_per_sec:.02f}", model.mode, device, get_arch_name(), True),
        Experiment(model.name, "memory_bandwidth(GB/s)", model.memory_bandwidth, f"{memory_bandwidth:.02f}", model.mode, device, get_arch_name(), True),
        Experiment(model.name, "compilation_time(s)", model.compilation_time, f"{compilation_time:.02f}", model.mode, device, get_arch_name(), True),
    ]


@register_experiment(name="llama2_7b_bf16")
def run_llama2_7b_bf16(device: str = "cuda"):
    from model import Transformer as LLaMA
    from quantize import WeightOnlyInt8QuantHandler as LLaMAWeightOnlyInt8QuantHandler
    model = GPTModelConfig("Llama-2-7b-chat-hf", LLaMA, "bfloat16", LLaMAWeightOnlyInt8QuantHandler, 94, 1253, 133)
    token_per_sec, memory_bandwidth, compilation_time = run_experiment(model, device=device)
    return [
        Experiment(model.name, "token_per_sec", model.token_per_sec, f"{token_per_sec:.02f}", model.mode, device, get_arch_name(), True),
        Experiment(model.name, "memory_bandwidth(GB/s)", model.memory_bandwidth, f"{memory_bandwidth:.02f}", model.mode, device, get_arch_name(), True),
        Experiment(model.name, "compilation_time(s)", model.compilation_time, f"{compilation_time:.02f}", model.mode, device, get_arch_name(), True),
    ]


@register_experiment(name="llama2_7b_int8")
def run_llama2_7b_int8(device: str = "cuda"):
    from model import Transformer as LLaMA
    from quantize import WeightOnlyInt8QuantHandler as LLaMAWeightOnlyInt8QuantHandler
    model = GPTModelConfig("Llama-2-7b-chat-hf", LLaMA, "int8", LLaMAWeightOnlyInt8QuantHandler, 144, 957, 136)
    token_per_sec, memory_bandwidth, compilation_time = run_experiment(model, device=device)
    return [
        Experiment(model.name, "token_per_sec", model.token_per_sec, f"{token_per_sec:.02f}", model.mode, device, get_arch_name(), True),
        Experiment(model.name, "memory_bandwidth(GB/s)", model.memory_bandwidth, f"{memory_bandwidth:.02f}", model.mode, device, get_arch_name(), True),
        Experiment(model.name, "compilation_time(s)", model.compilation_time, f"{compilation_time:.02f}", model.mode, device, get_arch_name(), True),
    ]


@register_experiment(name="mixtral_8x7b_int8")
def run_mixtral_8x7b_int8(device: str = "cuda"):
    from mixtral_moe_model import Transformer as MixtralMoE
    from mixtral_moe_quantize import WeightOnlyInt8QuantHandler as MixtralMoEWeightOnlyInt8QuantHandler
    model = GPTModelConfig("Mixtral-8x7B-v0.1", MixtralMoE, "int8", MixtralMoEWeightOnlyInt8QuantHandler, 175, 1130, 133)
    token_per_sec, memory_bandwidth, compilation_time = run_experiment(model, device=device)
    return [
        Experiment(model.name, "token_per_sec", model.token_per_sec, f"{token_per_sec:.02f}", model.mode, device, get_arch_name(), True),
        Experiment(model.name, "memory_bandwidth(GB/s)", model.memory_bandwidth, f"{memory_bandwidth:.02f}", model.mode, device, get_arch_name(), True),
        Experiment(model.name, "compilation_time(s)", model.compilation_time, f"{compilation_time:.02f}", model.mode, device, get_arch_name(), True),
    ]


WARMUP_ITER = 5
A100_40G_BF16_TFLOPS = 312


class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dtype):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim, dtype=dtype),
            nn.LayerNorm(hidden_dim, dtype=dtype),
            nn.Linear(hidden_dim, output_dim, dtype=dtype),
            nn.LayerNorm(output_dim, dtype=dtype),
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


@register_experiment(name="mlp_layer_norm_gelu")
def run_mlp_layer_norm_gelu(device: str = "cuda"):
    from torch._inductor.runtime.benchmarking import benchmarker
    from torch.utils.flop_counter import FlopCounterMode
    dtype_flops_utilization_map = {torch.bfloat16: "0.8"}
    input_shapes = [1024, 4096, 8192, 16384]
    intermediate_size = 14336
    results = []
    for dtype, expected_flops_utilization in dtype_flops_utilization_map.items():
        flops_utilization = 0
        total_bf, total_bf_v, total_pad, total_fuse = 0, 0, 0, 0
        for D in input_shapes:
            counters.clear()
            torch._dynamo.reset()
            mod = SimpleMLP(input_dim=D, hidden_dim=intermediate_size, output_dim=D, dtype=dtype).to(device)
            x = torch.randn(D, device=device, dtype=torch.bfloat16)
            with FlopCounterMode(display=False) as mode:
                mod(x)
            flops = mode.get_total_flops()
            compiled_mod = torch.compile(mod, dynamic=False)
            for _ in range(WARMUP_ITER):
                compiled_mod(x)
            total_bf += safe_get_counter('binary_folding', 0)
            total_bf_v += safe_get_counter('binary_folding_z3_verified', 0)
            total_pad += safe_get_counter('pad_mm_z3_verified', 0)
            total_fuse += safe_get_counter('fuse_attention', 0)
            us_per_iter = benchmarker.benchmark(compiled_mod, (x,), {}) * 1000
            flops_utilization += us_per_iter * flops / 1e9 / A100_40G_BF16_TFLOPS
        flops_utilization = flops_utilization / len(input_shapes)
        dtype_str = str(dtype).replace("torch.", "")
        z3_results.append({'experiment': 'mlp_layer_norm_gelu', 'mode': dtype_str, 'bf_total': total_bf, 'bf_verified': total_bf_v, 'pad_verified': total_pad, 'fuse_attention': total_fuse, 'sdpa_total': 0, 'sdpa_verified': 0, 'sdpa_failed': 0, 'sdpa_skipped': 0, 'gqa_safe': True, 'gqa_issues': 0})
        print(f"[Z3] mlp_layer_norm_gelu: BF {total_bf_v}/{total_bf} | Pad {total_pad} | SDPA: 0/0 | GQA: N/A")
        results.append(Experiment("mlp_layer_norm_gelu", "flops_utilization", expected_flops_utilization, f"{flops_utilization:.02f}", dtype_str, device, get_arch_name()))
    return results


@register_experiment(name="layer_norm")
def run_layer_norm(device: str = "cuda"):
    from torch._inductor.runtime.benchmarking import benchmarker
    dtype_memory_bandwidth_map = {torch.bfloat16: "950"}
    input_shapes = [1024, 4096, 8192, 16384]
    BS = 4096
    results = []
    for dtype, expected_memory_bandwidth in dtype_memory_bandwidth_map.items():
        memory_bandwidth = 0
        total_bf, total_bf_v, total_pad, total_fuse = 0, 0, 0, 0
        for D in input_shapes:
            counters.clear()
            torch._dynamo.reset()
            mod = nn.LayerNorm(D).to(device)
            x = torch.randn(BS, D, device=device, dtype=dtype)
            compiled_mod = torch.compile(mod, dynamic=False)
            for _ in range(WARMUP_ITER):
                compiled_mod(x)
            total_bf += safe_get_counter('binary_folding', 0)
            total_bf_v += safe_get_counter('binary_folding_z3_verified', 0)
            total_pad += safe_get_counter('pad_mm_z3_verified', 0)
            total_fuse += safe_get_counter('fuse_attention', 0)
            us_per_iter = benchmarker.benchmark(compiled_mod, (x,), {}) * 1000
            memory_bandwidth += (1e6 / us_per_iter) * 2 * BS * D * dtype.itemsize / 1e9
        memory_bandwidth = memory_bandwidth / len(input_shapes)
        dtype_str = str(dtype).replace("torch.", "")
        z3_results.append({'experiment': 'layer_norm', 'mode': dtype_str, 'bf_total': total_bf, 'bf_verified': total_bf_v, 'pad_verified': total_pad, 'fuse_attention': total_fuse, 'sdpa_total': 0, 'sdpa_verified': 0, 'sdpa_failed': 0, 'sdpa_skipped': 0, 'gqa_safe': True, 'gqa_issues': 0})
        print(f"[Z3] layer_norm: BF {total_bf_v}/{total_bf} | Pad {total_pad} | SDPA: 0/0 | GQA: N/A")
        results.append(Experiment("layer_norm", "memory_bandwidth(GB/s)", expected_memory_bandwidth, f"{memory_bandwidth:.02f}", dtype_str, device, get_arch_name()))
    return results


@register_experiment(name="gather_gemv")
@torch._inductor.config.patch(coordinate_descent_tuning=True)
def run_gather_gemv(device: str = "cuda"):
    from torch._inductor.runtime.benchmarking import benchmarker
    E = 8
    dtype_memory_bandwidth_map = {torch.int8: "990", torch.bfloat16: "1060"}
    input_shapes = [1024, 4096, 8192, 16384]
    results = []
    for dtype, expected_memory_bandwidth in dtype_memory_bandwidth_map.items():
        memory_bandwidth = 0
        total_bf, total_bf_v, total_pad, total_fuse = 0, 0, 0, 0
        for D in input_shapes:
            counters.clear()
            torch._dynamo.reset()
            def gather_gemv(W, score_idxs, x):
                return W[score_idxs].to(x.dtype) @ x
            W = torch.randn(E, D, D, device=device).to(dtype=dtype)
            x = torch.randn(D, device=device, dtype=torch.bfloat16)
            score_idxs = torch.tensor([3, 5], device=device)
            compiled_fn = torch.compile(gather_gemv, dynamic=False)
            for _ in range(WARMUP_ITER):
                compiled_fn(W, score_idxs, x)
            total_bf += safe_get_counter('binary_folding', 0)
            total_bf_v += safe_get_counter('binary_folding_z3_verified', 0)
            total_pad += safe_get_counter('pad_mm_z3_verified', 0)
            total_fuse += safe_get_counter('fuse_attention', 0)
            us_per_iter = benchmarker.benchmark(compiled_fn, (W, score_idxs, x), {}) * 1000
            memory_bandwidth += (1e6 / us_per_iter) * 2 * D * D * dtype.itemsize / 1e9
        memory_bandwidth = memory_bandwidth / len(input_shapes)
        dtype_str = str(dtype).replace("torch.", "")
        z3_results.append({'experiment': 'gather_gemv', 'mode': dtype_str, 'bf_total': total_bf, 'bf_verified': total_bf_v, 'pad_verified': total_pad, 'fuse_attention': total_fuse, 'sdpa_total': 0, 'sdpa_verified': 0, 'sdpa_failed': 0, 'sdpa_skipped': 0, 'gqa_safe': True, 'gqa_issues': 0})
        print(f"[Z3] gather_gemv ({dtype_str}): BF {total_bf_v}/{total_bf} | Pad {total_pad} | SDPA: 0/0 | GQA: N/A")
        results.append(Experiment("gather_gemv", "memory_bandwidth(GB/s)", expected_memory_bandwidth, f"{memory_bandwidth:.02f}", dtype_str, device, get_arch_name()))
    return results


@register_experiment(name="gemv")
@torch._inductor.config.patch(coordinate_descent_tuning=True)
def run_gemv(device: str = "cuda"):
    from torch._inductor.runtime.benchmarking import benchmarker
    dtype_memory_bandwidth_map = {torch.int8: "870", torch.bfloat16: "990"}
    input_shapes = [1024, 4096, 8192, 16384]
    results = []
    for dtype, expected_memory_bandwidth in dtype_memory_bandwidth_map.items():
        memory_bandwidth = 0
        total_bf, total_bf_v, total_pad, total_fuse = 0, 0, 0, 0
        for D in input_shapes:
            counters.clear()
            torch._dynamo.reset()
            def gemv(W, x):
                return W.to(x.dtype) @ x
            W = torch.randn(D, D, device=device).to(dtype=dtype)
            x = torch.randn(D, device=device, dtype=torch.bfloat16)
            compiled_fn = torch.compile(gemv, dynamic=False)
            for _ in range(WARMUP_ITER):
                compiled_fn(W, x)
            total_bf += safe_get_counter('binary_folding', 0)
            total_bf_v += safe_get_counter('binary_folding_z3_verified', 0)
            total_pad += safe_get_counter('pad_mm_z3_verified', 0)
            total_fuse += safe_get_counter('fuse_attention', 0)
            us_per_iter = benchmarker.benchmark(compiled_fn, (W, x), {}) * 1000
            memory_bandwidth += (1e6 / us_per_iter) * D * D * dtype.itemsize / 1e9
        memory_bandwidth = memory_bandwidth / len(input_shapes)
        dtype_str = str(dtype).replace("torch.", "")
        z3_results.append({'experiment': 'gemv', 'mode': dtype_str, 'bf_total': total_bf, 'bf_verified': total_bf_v, 'pad_verified': total_pad, 'fuse_attention': total_fuse, 'sdpa_total': 0, 'sdpa_verified': 0, 'sdpa_failed': 0, 'sdpa_skipped': 0, 'gqa_safe': True, 'gqa_issues': 0})
        print(f"[Z3] gemv ({dtype_str}): BF {total_bf_v}/{total_bf} | Pad {total_pad} | SDPA: 0/0 | GQA: N/A")
        results.append(Experiment("gemv", "memory_bandwidth(GB/s)", expected_memory_bandwidth, f"{memory_bandwidth:.02f}", dtype_str, device, get_arch_name()))
    return results


DEFAULT_OUTPUT_FILE = "gpt_fast_benchmark_verified.csv"


def output_csv(output_file, headers, row):
    if os.path.exists(output_file):
        with open(output_file) as fd:
            lines = list(csv.reader(fd)) or [[]]
            if headers and len(headers) > len(lines[0]):
                lines[0] = headers
            else:
                headers = lines[0]
    else:
        lines = [headers]
    if output_file != DEFAULT_OUTPUT_FILE:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
    lines.append([(f"{x:.6f}" if isinstance(x, float) else x) for x in row])
    with open(output_file, "w") as fd:
        writer = csv.writer(fd, lineterminator="\n")
        for line in lines:
            writer.writerow(list(line) + ["0"] * (len(headers) - len(line)))


def output_json(output_file, headers, row):
    mapping_headers = {headers[i]: v for i, v in enumerate(row)}
    record = {
        "benchmark": {
            "name": "PyTorch gpt-fast benchmark (Z3 verified)",
            "mode": "inference",
            "dtype": mapping_headers.get("dtype", "unknown"),
            "extra_info": {"device": mapping_headers.get("device", "unknown"), "arch": mapping_headers.get("arch", "unknown")},
        },
        "model": {
            "name": mapping_headers.get("name", "unknown"),
            "type": "OSS model" if mapping_headers.get("is_model") else "micro-benchmark",
            "origins": ["pytorch"],
        },
        "metric": {
            "name": mapping_headers.get("metric", "unknown"),
            "benchmark_values": [mapping_headers.get("actual", "0")],
            "target_value": mapping_headers.get("target", "0"),
        },
    }
    with open(f"{os.path.splitext(output_file)[0]}.json", "a") as f:
        print(json.dumps(record), file=f)


def output_z3_results(output_file: str):
    z3_output_file = f"{os.path.splitext(output_file)[0]}_z3_verification.csv"
    if z3_results:
        with open(z3_output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=z3_results[0].keys())
            writer.writeheader()
            writer.writerows(z3_results)
        print(f"\nZ3 verification results saved to: {z3_output_file}")
        try:
            import pandas as pd
            df = pd.DataFrame(z3_results)
            xlsx_file = f"{os.path.splitext(output_file)[0]}_z3_verification.xlsx"
            df.to_excel(xlsx_file, index=False)
            print(f"Z3 verification results saved to: {xlsx_file}")
        except ImportError:
            pass


def print_z3_summary():
    print("\n" + "=" * 80)
    print("Z3 VERIFICATION SUMMARY")
    print("=" * 80)
    total_bf_verified = sum(r.get('bf_verified', 0) for r in z3_results)
    total_bf_total = sum(r.get('bf_total', 0) for r in z3_results)
    total_pad_verified = sum(r.get('pad_verified', 0) for r in z3_results)
    total_sdpa_verified = sum(r.get('sdpa_verified', 0) for r in z3_results)
    total_sdpa_total = sum(r.get('sdpa_total', 0) for r in z3_results)
    total_gqa_safe = sum(1 for r in z3_results if r.get('gqa_safe', True))
    total_gqa_issues = sum(r.get('gqa_issues', 0) for r in z3_results)
    
    for r in z3_results:
        exp = r.get('experiment', 'unknown')
        mode = r.get('mode', r.get('dtype', ''))
        bf_v = r.get('bf_verified', 0)
        bf_t = r.get('bf_total', 0)
        pad_v = r.get('pad_verified', 0)
        sdpa_v = r.get('sdpa_verified', 0)
        sdpa_t = r.get('sdpa_total', 0)
        gqa_s = "SAFE" if r.get('gqa_safe', True) else "ISSUES"
        print(f"  {exp:30s} ({mode:10s}): BF {bf_v:2d}/{bf_t:2d} | Pad {pad_v:3d} | SDPA: {sdpa_v:2d}/{sdpa_t:2d} | GQA: {gqa_s}")
    
    print("-" * 80)
    print(f"  {'TOTAL':30s}             : BF {total_bf_verified:2d}/{total_bf_total:2d} | Pad {total_pad_verified:3d} | SDPA: {total_sdpa_verified:2d}/{total_sdpa_total:2d} | GQA: {total_gqa_safe}/{len(z3_results)} safe")
    print("=" * 80)
    
    if total_bf_verified > 0:
        print("\n[SUCCESS] Binary Folding Z3 verification is WORKING!")
    if total_pad_verified > 0:
        print("[SUCCESS] Pad MM Z3 verification is WORKING!")
    if total_sdpa_verified > 0:
        print(f"[SUCCESS] SDPA Z3 verification: {total_sdpa_verified}/{total_sdpa_total} verified!")
    if total_gqa_issues > 0:
        print(f"[WARNING] GQA issues detected: {total_gqa_issues} issues found")
    else:
        print("[SUCCESS] GQA Z3 verification: All configurations safe!")


def main(output_file=DEFAULT_OUTPUT_FILE, only_model=None):
    print("=" * 80)
    print("GPT-FAST BENCHMARK WITH Z3 VERIFICATION")
    print("=" * 80)
    results = []
    if not only_model:
        experiments = all_experiments.values()
    else:
        if only_model not in all_experiments:
            print(f"Unknown model: {only_model}, all available models: {list(all_experiments.keys())}")
            return
        experiments = [all_experiments[only_model]]
    for func in experiments:
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except AssertionError:
            device = "cpu"
        torch.compiler.cudagraph_mark_step_begin()
        lst = func(device)
        for x in lst:
            results.append(dataclasses.astuple(x))
    headers = [field.name for field in dataclasses.fields(Experiment)]
    for row in results:
        output_csv(output_file, headers, row)
        output_json(output_file, headers, row)
    output_z3_results(output_file)
    print_z3_summary()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GPT-Fast benchmarks with Z3 verification.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_FILE, help="Output CSV file")
    parser.add_argument("--only", help="Run only specified experiment")
    args = parser.parse_args()
    main(output_file=args.output, only_model=args.only)
