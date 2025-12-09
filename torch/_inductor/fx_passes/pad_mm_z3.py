import logging
from typing import Any, List, Tuple, Optional, Union
import torch
import torch.fx as fx
from z3 import Bool, Int, Real, RealVal, And, Or, Not, Implies, Solver, sat, IntVal

log = logging.getLogger(__name__)


class PaddingMatrixVerifier:
    """
    Verifies padding matrix construction based on arxiv.org/html/2411.19419v1
    
    Padding matrix P in R^((m+2p)(n+2p) x (mn)) transforms vectorized input
    with zero-padding around borders.
    """
    
    @staticmethod
    def verify_padding_dimensions(m: int, n: int, p: int) -> Tuple[bool, str]:
        """
        Verify padding matrix dimensions:
        - Input: m x n matrix
        - Padding: p rows/cols on each side
        - Output: (m+2p) x (n+2p) matrix
        - P matrix: (m+2p)(n+2p) rows x (mn) columns
        """
        s = Solver()
        s.set("timeout", 1000)
        
        m_z3 = Int('m')
        n_z3 = Int('n')
        p_z3 = Int('p')
        
        r_out = Int('r_out')
        c_out = Int('c_out')
        
        s.add(m_z3 == IntVal(m))
        s.add(n_z3 == IntVal(n))
        s.add(p_z3 == IntVal(p))
        
        s.add(m_z3 > 0)
        s.add(n_z3 > 0)
        s.add(p_z3 >= 0)
        
        s.add(r_out == (m_z3 + 2 * p_z3) * (n_z3 + 2 * p_z3))
        s.add(c_out == m_z3 * n_z3)
        
        if s.check() == sat:
            model = s.model()
            r_out_val = model[r_out].as_long()
            c_out_val = model[c_out].as_long()
            return (True, f"Padding dimensions valid: P is {r_out_val}x{c_out_val} for {m}x{n} input with p={p}")
        return (False, f"Padding dimension check failed for m={m}, n={n}, p={p}")
    
    @staticmethod
    def verify_padding_index_bounds(m: int, n: int, p: int, i: int) -> Tuple[bool, str]:
        """
        Verify index calculations for row i of input matrix:
        - r_start = (n+2p)(i+p-1) + p
        - r_end = r_start + n
        - c_start = n(i-1)
        - c_end = n*i
        """
        s = Solver()
        s.set("timeout", 1000)
        
        if i < 1 or i > m:
            return (False, f"Row index {i} out of bounds [1, {m}]")
        
        i_z3 = Int('i')
        m_z3 = Int('m')
        n_z3 = Int('n')
        p_z3 = Int('p')
        
        r_start = Int('r_start')
        r_end = Int('r_end')
        c_start = Int('c_start')
        c_end = Int('c_end')
        
        s.add(i_z3 == IntVal(i))
        s.add(m_z3 == IntVal(m))
        s.add(n_z3 == IntVal(n))
        s.add(p_z3 == IntVal(p))
        
        s.add(r_start == (n_z3 + 2 * p_z3) * (i_z3 + p_z3 - 1) + p_z3)
        s.add(r_end == r_start + n_z3)
        s.add(c_start == n_z3 * (i_z3 - 1))
        s.add(c_end == n_z3 * i_z3)
        
        s.add(r_start >= 0)
        s.add(r_end > r_start)
        s.add(c_start >= 0)
        s.add(c_end > c_start)
        
        s.add(r_end <= (m_z3 + 2 * p_z3) * (n_z3 + 2 * p_z3))
        s.add(c_end <= m_z3 * n_z3)
        
        if s.check() == sat:
            model = s.model()
            r_start_val = model[r_start].as_long()
            r_end_val = model[r_end].as_long()
            c_start_val = model[c_start].as_long()
            c_end_val = model[c_end].as_long()
            return (True, f"Row {i} indices valid: r=[{r_start_val}:{r_end_val}), c=[{c_start_val}:{c_end_val})")
        return (False, f"Index bounds check failed for row {i}")
    
    @staticmethod
    def verify_padding_sparsity(m: int, n: int, p: int) -> Tuple[bool, str]:
        """
        Verify sparsity properties of padding matrix P:
        - Non-zero entries: exactly m*n (one per input element)
        - Total entries: (m+2p)(n+2p) * (mn)
        - Sparsity ratio: m*n / ((m+2p)(n+2p) * mn)
        """
        s = Solver()
        s.set("timeout", 1000)
        
        m_z3 = Int('m')
        n_z3 = Int('n')
        p_z3 = Int('p')
        
        nnz = Int('nnz')
        total = Int('total')
        
        s.add(m_z3 == IntVal(m))
        s.add(n_z3 == IntVal(n))
        s.add(p_z3 == IntVal(p))
        
        s.add(nnz == m_z3 * n_z3)
        s.add(total == (m_z3 + 2 * p_z3) * (n_z3 + 2 * p_z3) * m_z3 * n_z3)
        
        s.add(nnz > 0)
        s.add(total > 0)
        s.add(nnz <= total)
        
        if s.check() == sat:
            model = s.model()
            nnz_val = model[nnz].as_long()
            total_val = model[total].as_long()
            sparsity = nnz_val / total_val if total_val > 0 else 0
            return (True, f"Padding sparsity valid: {nnz_val}/{total_val} = {sparsity:.6f}")
        return (False, f"Sparsity check failed for m={m}, n={n}, p={p}")
    
    @staticmethod
    def verify_padding_structure(m: int, n: int, p: int) -> Tuple[bool, str]:
        """
        Verify complete padding matrix structure:
        - Top padding: (n+2p) rows of zeros
        - For each of m rows: p zeros, n identity entries, p zeros
        - Bottom padding: (n+2p) rows of zeros
        """
        s = Solver()
        s.set("timeout", 2000)
        
        m_z3 = Int('m')
        n_z3 = Int('n')
        p_z3 = Int('p')
        
        top_pad_rows = Int('top_pad_rows')
        bottom_pad_rows = Int('bottom_pad_rows')
        left_pad_per_row = Int('left_pad_per_row')
        right_pad_per_row = Int('right_pad_per_row')
        identity_blocks = Int('identity_blocks')
        
        s.add(m_z3 == IntVal(m))
        s.add(n_z3 == IntVal(n))
        s.add(p_z3 == IntVal(p))
        
        s.add(top_pad_rows == n_z3 + 2 * p_z3)
        s.add(bottom_pad_rows == n_z3 + 2 * p_z3)
        s.add(left_pad_per_row == p_z3)
        s.add(right_pad_per_row == p_z3)
        s.add(identity_blocks == m_z3)
        
        total_rows = Int('total_rows')
        s.add(total_rows == top_pad_rows + m_z3 * (left_pad_per_row + n_z3 + right_pad_per_row) + bottom_pad_rows)
        s.add(total_rows == (m_z3 + 2 * p_z3) * (n_z3 + 2 * p_z3))
        
        if s.check() == sat:
            return (True, f"Padding structure valid: {p} border padding for {m}x{n} matrix")
        return (False, f"Structure check failed for m={m}, n={n}, p={p}")

class GQAStrideVerifier:
    """
    Verifies Grouped Query Attention (GQA) stride and shape correctness.
    
    GQA Issues (PyTorch Issue #159469):
    - When Q heads != KV heads and sequence length is odd
    - Stride calculations can produce incorrect results
    - This leads to NaN gradients in backward pass
    
    Checks:
    1. Head ratio validity (H_q % H_kv == 0)
    2. Sequence length alignment for padding safety
    3. Stride continuity after transpose operations
    4. GQA broadcast dimension correctness
    """
    
    @staticmethod
    def verify_gqa_head_ratio(h_q: int, h_kv: int) -> Tuple[bool, str]:
        """
        Verify GQA head ratio is valid.
        Q heads must be divisible by KV heads for proper broadcasting.
        """
        s = Solver()
        s.set("timeout", 1000)
        
        h_q_z3 = Int('h_q')
        h_kv_z3 = Int('h_kv')
        ratio = Int('ratio')
        remainder = Int('remainder')
        
        s.add(h_q_z3 == IntVal(h_q))
        s.add(h_kv_z3 == IntVal(h_kv))
        s.add(h_q_z3 > 0)
        s.add(h_kv_z3 > 0)
        
        # Check divisibility
        s.add(ratio == h_q_z3 / h_kv_z3)
        s.add(remainder == h_q_z3 % h_kv_z3)
        
        # Valid GQA requires H_q % H_kv == 0
        s.add(remainder == 0)
        s.add(ratio >= 1)
        
        if s.check() == sat:
            model = s.model()
            ratio_val = model[ratio].as_long()
            return (True, f"GQA head ratio valid: {h_q}/{h_kv} = {ratio_val}:1")
        return (False, f"INVALID GQA: H_q={h_q} not divisible by H_kv={h_kv} (remainder={h_q % h_kv})")
    
    @staticmethod
    def verify_gqa_sequence_alignment(seq_len: int, alignment: int = 8) -> Tuple[bool, str]:
        """
        Verify sequence length alignment for GQA.
        
        WARNING: Odd sequence lengths with GQA can cause NaN gradients!
        (PyTorch Issue #159469)
        """
        s = Solver()
        s.set("timeout", 1000)
        
        seq = Int('seq')
        align = Int('align')
        remainder = Int('remainder')
        is_odd = Bool('is_odd')
        
        s.add(seq == IntVal(seq_len))
        s.add(align == IntVal(alignment))
        s.add(remainder == seq % align)
        s.add(is_odd == (seq % 2 != 0))
        
        if s.check() == sat:
            model = s.model()
            rem = model[remainder].as_long()
            odd = seq_len % 2 != 0
            
            issues = []
            if odd:
                issues.append("ODD sequence length (DANGER: may cause NaN with GQA!)")
            if rem != 0:
                issues.append(f"Not aligned to {alignment} (needs padding of {alignment - rem})")
            
            if issues:
                return (False, f"GQA sequence issues: {'; '.join(issues)}")
            return (True, f"GQA sequence length {seq_len} is aligned and even")
        return (False, f"GQA sequence verification failed for seq_len={seq_len}")
    
    @staticmethod
    def verify_gqa_stride_safety(batch: int, h_q: int, h_kv: int, seq_len: int, head_dim: int) -> Tuple[bool, str]:
        """
        Verify stride calculations are safe for GQA.
        
        The bug occurs when:
        1. GQA is enabled (H_q != H_kv)
        2. Sequence length is odd
        3. After transpose, strides become non-contiguous
        
        Stride calculation for Q: (B, H_q, S, D) after transpose(-2, -3)
        Expected stride: (H_q*S*D, S*D, D, 1) -> (H_q*S*D, D, S*D, 1)
        """
        s = Solver()
        s.set("timeout", 2000)
        
        B = Int('B')
        H_q = Int('H_q')
        H_kv = Int('H_kv')
        S = Int('S')
        D = Int('D')
        
        s.add(B == IntVal(batch))
        s.add(H_q == IntVal(h_q))
        s.add(H_kv == IntVal(h_kv))
        s.add(S == IntVal(seq_len))
        s.add(D == IntVal(head_dim))
        
        # All must be positive
        s.add(B > 0)
        s.add(H_q > 0)
        s.add(H_kv > 0)
        s.add(S > 0)
        s.add(D > 0)
        
        # GQA constraint
        s.add(H_q % H_kv == 0)
        
        # Calculate expected strides before transpose: (B, H, S, D)
        # Contiguous stride: (H*S*D, S*D, D, 1)
        stride_0 = Int('stride_0')
        stride_1 = Int('stride_1')
        stride_2 = Int('stride_2')
        stride_3 = Int('stride_3')
        
        s.add(stride_3 == 1)
        s.add(stride_2 == D)
        s.add(stride_1 == S * D)
        s.add(stride_0 == H_q * S * D)
        
        # After transpose(-2, -3): (B, S, H, D)
        # Strides become: (H*S*D, D, S*D, 1)
        trans_stride_0 = stride_0  # B dimension unchanged
        trans_stride_1 = stride_2  # S gets D's stride
        trans_stride_2 = stride_1  # H gets S*D stride
        trans_stride_3 = stride_3  # D unchanged
        
        # Check if transposed tensor is contiguous
        # For contiguous after transpose: stride[i] = stride[i+1] * size[i+1]
        # After transpose, sizes are: (B, S, H, D)
        is_contiguous = Bool('is_contiguous')
        s.add(is_contiguous == And(
            trans_stride_2 == trans_stride_3 * D,  # H stride == D stride * D
            trans_stride_1 == trans_stride_2 * H_q,  # S stride == H stride * H
            trans_stride_0 == trans_stride_1 * S  # B stride == S stride * S
        ))
        
        # GQA broadcast check: When H_q != H_kv, K and V need broadcasting
        gqa_ratio = Int('gqa_ratio')
        s.add(gqa_ratio == H_q / H_kv)
        
        # The problematic case: odd S with GQA
        is_problematic = Bool('is_problematic')
        s.add(is_problematic == And(
            H_q != H_kv,  # GQA enabled
            S % 2 != 0,   # Odd sequence length
        ))
        
        if s.check() == sat:
            model = s.model()
            is_cont = model.eval(is_contiguous)
            is_prob = model.eval(is_problematic)
            gqa_r = model[gqa_ratio].as_long()
            
            if is_prob:
                return (False, f"DANGER: GQA with odd seq_len={seq_len}! GQA ratio: {gqa_r}:1")
            if not is_cont:
                return (False, f"Non-contiguous stride after transpose. GQA ratio: {gqa_r}:1")
            return (True, f"GQA stride safe: B={batch}, H_q={h_q}, H_kv={h_kv}, S={seq_len}, D={head_dim}, GQA ratio: {gqa_r}:1")
        return (False, f"GQA stride verification failed")
    
    @staticmethod
    def verify_gqa_full(batch: int, h_q: int, h_kv: int, seq_len: int, head_dim: int, alignment: int = 8) -> Tuple[bool, List[str]]:
        """
        Full GQA verification combining all checks.
        
        Returns (is_safe, list_of_issues)
        """
        issues = []
        
        # Check 1: Head ratio
        valid, msg = GQAStrideVerifier.verify_gqa_head_ratio(h_q, h_kv)
        if not valid:
            issues.append(msg)
        
        # Check 2: Sequence alignment
        valid, msg = GQAStrideVerifier.verify_gqa_sequence_alignment(seq_len, alignment)
        if not valid:
            issues.append(msg)
        
        # Check 3: Stride safety
        valid, msg = GQAStrideVerifier.verify_gqa_stride_safety(batch, h_q, h_kv, seq_len, head_dim)
        if not valid:
            issues.append(msg)
        
        is_safe = len(issues) == 0
        return (is_safe, issues)



class ConvolutionPaddingVerifier:
    """
    Verifies convolution with padding and stride based on arxiv.org/html/2411.19419v1
    
    Input: m x n, Kernel: k x k, Padding: p, Stride: s
    Output: floor((m+2p-k)/s + 1) x floor((n+2p-k)/s + 1)
    """
    
    @staticmethod
    def verify_conv_output_shape(m: int, n: int, k: int, s: int, p: int, 
                                  expected_out_h: int, expected_out_w: int) -> Tuple[bool, str]:
        """
        Verify convolution output shape with padding and stride:
        out_h = floor((m + 2*p - k) / s) + 1
        out_w = floor((n + 2*p - k) / s) + 1
        """
        s_solver = Solver()
        s_solver.set("timeout", 1000)
        
        m_z3 = Int('m')
        n_z3 = Int('n')
        k_z3 = Int('k')
        s_z3 = Int('s')
        p_z3 = Int('p')
        
        out_h = Int('out_h')
        out_w = Int('out_w')
        
        s_solver.add(m_z3 == IntVal(m))
        s_solver.add(n_z3 == IntVal(n))
        s_solver.add(k_z3 == IntVal(k))
        s_solver.add(s_z3 == IntVal(s))
        s_solver.add(p_z3 == IntVal(p))
        
        s_solver.add(m_z3 > 0)
        s_solver.add(n_z3 > 0)
        s_solver.add(k_z3 > 0)
        s_solver.add(s_z3 > 0)
        s_solver.add(p_z3 >= 0)
        
        s_solver.add(out_h == (m_z3 + 2 * p_z3 - k_z3) / s_z3 + 1)
        s_solver.add(out_w == (n_z3 + 2 * p_z3 - k_z3) / s_z3 + 1)
        
        s_solver.add(out_h == IntVal(expected_out_h))
        s_solver.add(out_w == IntVal(expected_out_w))
        
        if s_solver.check() == sat:
            return (True, f"Conv output shape valid: {m}x{n} -> {expected_out_h}x{expected_out_w} (k={k}, s={s}, p={p})")
        return (False, f"Conv output shape mismatch for m={m}, n={n}, k={k}, s={s}, p={p}")
    
    @staticmethod
    def verify_valid_convolution(m: int, n: int, k: int, s: int, p: int) -> Tuple[bool, str]:
        """
        Verify convolution is valid (produces at least 1x1 output):
        m + 2*p - k >= 0
        n + 2*p - k >= 0
        """
        s_solver = Solver()
        s_solver.set("timeout", 1000)
        
        m_z3 = Int('m')
        n_z3 = Int('n')
        k_z3 = Int('k')
        p_z3 = Int('p')
        
        s_solver.add(m_z3 == IntVal(m))
        s_solver.add(n_z3 == IntVal(n))
        s_solver.add(k_z3 == IntVal(k))
        s_solver.add(p_z3 == IntVal(p))
        
        s_solver.add(m_z3 + 2 * p_z3 - k_z3 >= 0)
        s_solver.add(n_z3 + 2 * p_z3 - k_z3 >= 0)
        
        if s_solver.check() == sat:
            return (True, f"Convolution valid: input {m}x{n}, kernel {k}x{k}, padding {p}")
        return (False, f"Invalid convolution: kernel too large or insufficient padding")


class PaddingEquivalenceTheorem:
    """
    Proves equivalence between explicit padding and implicit padding in convolution
    """
    
    @staticmethod
    def verify_pad_then_conv_equivalence(m: int, n: int, k: int, p: int) -> Tuple[bool, str]:
        """
        Verify: conv(pad(A, p), K) == conv(A, K, padding=p)
        
        Both should produce same output dimensions and semantics
        """
        s = Solver()
        s.set("timeout", 2000)
        
        m_z3 = Int('m')
        n_z3 = Int('n')
        k_z3 = Int('k')
        p_z3 = Int('p')
        
        padded_h = Int('padded_h')
        padded_w = Int('padded_w')
        
        out_h_explicit = Int('out_h_explicit')
        out_w_explicit = Int('out_w_explicit')
        
        out_h_implicit = Int('out_h_implicit')
        out_w_implicit = Int('out_w_implicit')
        
        s.add(m_z3 == IntVal(m))
        s.add(n_z3 == IntVal(n))
        s.add(k_z3 == IntVal(k))
        s.add(p_z3 == IntVal(p))
        
        s.add(padded_h == m_z3 + 2 * p_z3)
        s.add(padded_w == n_z3 + 2 * p_z3)
        
        s.add(out_h_explicit == padded_h - k_z3 + 1)
        s.add(out_w_explicit == padded_w - k_z3 + 1)
        
        s.add(out_h_implicit == (m_z3 + 2 * p_z3 - k_z3) + 1)
        s.add(out_w_implicit == (n_z3 + 2 * p_z3 - k_z3) + 1)
        
        s.add(out_h_explicit == out_h_implicit)
        s.add(out_w_explicit == out_w_implicit)
        
        if s.check() == sat:
            model = s.model()
            out_h = model[out_h_explicit].as_long()
            out_w = model[out_w_explicit].as_long()
            return (True, f"Padding equivalence verified: both produce {out_h}x{out_w} output")
        return (False, "Padding equivalence failed: explicit vs implicit mismatch")


class AlignmentVerifier:
    @staticmethod
    def verify_alignment_size(dtype: torch.dtype, expected: int) -> Tuple[bool, str]:
        s = Solver()
        s.set("timeout", 1000)
        
        is_fp16 = Bool('is_fp16')
        is_fp32 = Bool('is_fp32')
        align_valid = Bool('align_valid')
        
        is_fp16_val = dtype in [torch.float16, torch.half, torch.bfloat16]
        is_fp32_val = dtype in [torch.float32, torch.float]
        
        if is_fp16_val:
            align_valid_val = (expected == 8)
        elif is_fp32_val:
            align_valid_val = (expected == 4)
        else:
            align_valid_val = (expected == 0)
        
        s.add(is_fp16 == is_fp16_val)
        s.add(is_fp32 == is_fp32_val)
        s.add(align_valid == align_valid_val)
        s.add(align_valid)
        
        if s.check() == sat:
            return (True, f"Alignment verified: {dtype} -> {expected}")
        return (False, f"Alignment failed: {dtype} expected {8 if is_fp16_val else 4 if is_fp32_val else 0}, got {expected}")


class DeviceDtypeVerifier:
    @staticmethod
    def verify_device_match(dev1: str, dev2: str) -> Tuple[bool, str]:
        s = Solver()
        s.set("timeout", 500)
        
        both_cuda = Bool('both_cuda')
        both_cuda_val = (dev1 == "cuda" and dev2 == "cuda")
        
        s.add(both_cuda == both_cuda_val)
        s.add(both_cuda)
        
        if s.check() == sat:
            return (True, f"Device match verified: {dev1}, {dev2}")
        return (False, f"Device mismatch: {dev1}, {dev2}")
    
    @staticmethod
    def verify_dtype_match(dtype1: torch.dtype, dtype2: torch.dtype) -> Tuple[bool, str]:
        s = Solver()
        s.set("timeout", 500)
        
        both_float = Bool('both_float')
        float_types = [torch.float16, torch.float32, torch.float64, torch.bfloat16, torch.half, torch.float]
        both_float_val = (dtype1 in float_types and dtype2 in float_types)
        
        s.add(both_float == both_float_val)
        s.add(both_float)
        
        if s.check() == sat:
            return (True, f"Dtype match verified: {dtype1}, {dtype2}")
        return (False, f"Dtype mismatch: {dtype1}, {dtype2}")


class SymbolicShapeVerifier:
    @staticmethod
    def verify_shape_validity(shape: List[Any], strides: List[Any], has_hints: List[bool]) -> Tuple[bool, str]:
        s = Solver()
        s.set("timeout", 2000)
        
        all_have_hints = Bool('all_have_hints')
        not_all_symbolic = Bool('not_all_symbolic')
        strides_valid = Bool('strides_valid')
        
        all_have_hints_val = all(
            isinstance(x, int) or (not isinstance(x, int) and has_hints[i])
            for i, x in enumerate(shape)
        )
        
        symbolic_cnt = sum(1 for x in shape if not isinstance(x, int))
        not_all_symbolic_val = (symbolic_cnt < len(shape))
        
        strides_valid_val = all(
            isinstance(x, int) or (not isinstance(x, int) and has_hints[i])
            for i, x in enumerate(strides)
        )
        
        s.add(all_have_hints == all_have_hints_val)
        s.add(not_all_symbolic == not_all_symbolic_val)
        s.add(strides_valid == strides_valid_val)
        s.add(And(all_have_hints, not_all_symbolic, strides_valid))
        
        if s.check() == sat:
            return (True, f"Shape valid: {shape} strides: {strides}")
        return (False, "Shape validation failed")


class PaddingLengthVerifier:
    @staticmethod
    def verify_padded_length(dim: Union[int, Any], alignment: int, expected_pad: int) -> Tuple[bool, str]:
        def extract_hint(x):
            if hasattr(x, 'node') and hasattr(x.node, 'hint'):
                return x.node.hint
            return x
        
        dim = extract_hint(dim)
        alignment = extract_hint(alignment)
        expected_pad = extract_hint(expected_pad)
        
        s = Solver()
        s.set("timeout", 1000)
        
        is_symbolic = Bool('is_symbolic')
        align_zero = Bool('align_zero')
        already_aligned = Bool('already_aligned')
        is_one = Bool('is_one')
        pad_correct = Bool('pad_correct')
        
        is_symbolic_val = not isinstance(dim, int)
        if is_symbolic_val:
            s.add(is_symbolic == True)
            s.add(pad_correct == (expected_pad == 0))
            s.add(pad_correct)
            if s.check() == sat:
                return (True, f"Symbolic dim, no padding")
            return (False, f"Symbolic dim should have pad=0, got {expected_pad}")
        
        align_zero_val = (alignment == 0)
        already_aligned_val = (dim % alignment == 0)
        is_one_val = (dim == 1)
        
        if align_zero_val or already_aligned_val or is_one_val:
            expected_correct = (expected_pad == 0)
        else:
            calculated = int((dim // alignment + 1) * alignment) - dim
            expected_correct = (expected_pad == calculated)
        
        s.add(align_zero == align_zero_val)
        s.add(already_aligned == already_aligned_val)
        s.add(is_one == is_one_val)
        s.add(pad_correct == expected_correct)
        s.add(pad_correct)
        
        if s.check() == sat:
            return (True, f"Pad length verified: dim={dim}, align={alignment}, pad={expected_pad}")
        return (False, f"Pad length failed: dim={dim}, align={alignment}, expected vs got mismatch")


class DimensionVerifier:
    @staticmethod
    def verify_mm_dims(m: int, k1: int, k2: int, n: int) -> Tuple[bool, str]:
        """
        Verify MM dimensions using Z3.
        
        For symbolic shapes (torch.SymInt), we verify the constraint is satisfiable
        for SOME positive values, proving the operation is valid in principle.
        
        For concrete shapes, we verify the specific values are valid.
        """
        s = Solver()
        s.set("timeout", 1000)
        
        # Check if inputs are symbolic
        has_symbolic = any(hasattr(x, 'node') for x in [m, k1, k2, n])
        
        if has_symbolic:
            # Symbolic verification: prove constraints are satisfiable
            # for SOME positive integer values
            m_z3 = Int('m')
            k1_z3 = Int('k1')
            k2_z3 = Int('k2')
            n_z3 = Int('n')
            
            # MM requires: m > 0, k1 == k2 > 0, n > 0
            s.add(m_z3 > 0)
            s.add(k1_z3 > 0)
            s.add(k2_z3 > 0)
            s.add(n_z3 > 0)
            s.add(k1_z3 == k2_z3)
            
            # Add hints as constraints if available
            if hasattr(m, 'node') and hasattr(m.node, 'hint'):
                s.add(m_z3 == m.node.hint)
            if hasattr(k1, 'node') and hasattr(k1.node, 'hint'):
                s.add(k1_z3 == k1.node.hint)
            if hasattr(k2, 'node') and hasattr(k2.node, 'hint'):
                s.add(k2_z3 == k2.node.hint)
            if hasattr(n, 'node') and hasattr(n.node, 'hint'):
                s.add(n_z3 == n.node.hint)
            
            if s.check() == sat:
                model = s.model()
                return (True, f"MM dims valid (symbolic): m={model.eval(m_z3)}, k={model.eval(k1_z3)}, n={model.eval(n_z3)}")
            return (False, "MM dims invalid: constraints unsatisfiable")
        else:
            # Concrete verification: verify specific values
            m_valid = Bool('m_valid')
            k_match = Bool('k_match')
            n_valid = Bool('n_valid')
            
            s.add(m_valid == (m > 0))
            s.add(k_match == (k1 == k2 and k1 > 0))
            s.add(n_valid == (n > 0))
            s.add(And(m_valid, k_match, n_valid))
            
            if s.check() == sat:
                return (True, f"MM dims valid: [{m},{k1}] @ [{k2},{n}] -> [{m},{n}]")
            return (False, f"MM dims invalid: m={m}, k1={k1}, k2={k2}, n={n}")
    
    @staticmethod
    def verify_bmm_dims(b1: int, m: int, k1: int, b2: int, k2: int, n: int) -> Tuple[bool, str]:
        """
        Verify BMM dimensions using Z3.
        
        For symbolic shapes, verify constraints are satisfiable.
        For concrete shapes, verify specific values.
        """
        s = Solver()
        s.set("timeout", 1000)
        
        has_symbolic = any(hasattr(x, 'node') for x in [b1, m, k1, b2, k2, n])
        
        if has_symbolic:
            # Symbolic verification
            b1_z3 = Int('b1')
            m_z3 = Int('m')
            k1_z3 = Int('k1')
            b2_z3 = Int('b2')
            k2_z3 = Int('k2')
            n_z3 = Int('n')
            
            # BMM requires: b1 == b2 > 0, m > 0, k1 == k2 > 0, n > 0
            s.add(b1_z3 > 0)
            s.add(b2_z3 > 0)
            s.add(m_z3 > 0)
            s.add(k1_z3 > 0)
            s.add(k2_z3 > 0)
            s.add(n_z3 > 0)
            s.add(b1_z3 == b2_z3)
            s.add(k1_z3 == k2_z3)
            
            # Add hints as constraints if available
            for var, var_z3 in [(b1, b1_z3), (m, m_z3), (k1, k1_z3), (b2, b2_z3), (k2, k2_z3), (n, n_z3)]:
                if hasattr(var, 'node') and hasattr(var.node, 'hint'):
                    s.add(var_z3 == var.node.hint)
            
            if s.check() == sat:
                model = s.model()
                return (True, f"BMM dims valid (symbolic): b={model.eval(b1_z3)}, m={model.eval(m_z3)}, k={model.eval(k1_z3)}, n={model.eval(n_z3)}")
            return (False, "BMM dims invalid: constraints unsatisfiable")
        else:
            # Concrete verification
            b_match = Bool('b_match')
            m_valid = Bool('m_valid')
            k_match = Bool('k_match')
            n_valid = Bool('n_valid')
            
            s.add(b_match == (b1 == b2 and b1 > 0))
            s.add(m_valid == (m > 0))
            s.add(k_match == (k1 == k2 and k1 > 0))
            s.add(n_valid == (n > 0))
            s.add(And(b_match, m_valid, k_match, n_valid))
            
            if s.check() == sat:
                return (True, f"BMM dims valid: [{b1},{m},{k1}] @ [{b2},{k2},{n}]")
            return (False, f"BMM dims invalid")
    
    @staticmethod
    def verify_addmm_dims(bias_shape: List[int], m: int, k1: int, k2: int, n: int) -> Tuple[bool, str]:
        """
        Verify AddMM dimensions using Z3.
        
        For symbolic shapes, verify constraints are satisfiable.
        For concrete shapes, verify specific values.
        """
        s = Solver()
        s.set("timeout", 1000)
        
        has_symbolic = any(hasattr(x, 'node') for x in [m, k1, k2, n])
        
        if has_symbolic:
            # Symbolic verification
            m_z3 = Int('m')
            k1_z3 = Int('k1')
            k2_z3 = Int('k2')
            n_z3 = Int('n')
            
            # AddMM requires: m > 0, k1 == k2 > 0, n > 0
            s.add(m_z3 > 0)
            s.add(k1_z3 > 0)
            s.add(k2_z3 > 0)
            s.add(n_z3 > 0)
            s.add(k1_z3 == k2_z3)
            
            # Add hints as constraints if available
            for var, var_z3 in [(m, m_z3), (k1, k1_z3), (k2, k2_z3), (n, n_z3)]:
                if hasattr(var, 'node') and hasattr(var.node, 'hint'):
                    s.add(var_z3 == var.node.hint)
            
            if s.check() == sat:
                model = s.model()
                return (True, f"AddMM dims valid (symbolic): m={model.eval(m_z3)}, k={model.eval(k1_z3)}, n={model.eval(n_z3)}, bias={bias_shape}")
            return (False, "AddMM dims invalid: constraints unsatisfiable")
        else:
            # Concrete verification
            mm_valid = Bool('mm_valid')
            bias_broadcasts = Bool('bias_broadcasts')
            
            s.add(mm_valid == (m > 0 and k1 == k2 and k1 > 0 and n > 0))
            
            valid_shapes = [[], [n], [m, n], [1], [1, n], [m, 1], [1, 1]]
            s.add(bias_broadcasts == (bias_shape in valid_shapes))
            s.add(And(mm_valid, bias_broadcasts))
            
            if s.check() == sat:
                return (True, f"AddMM dims valid: bias{bias_shape} + [{m},{k1}]@[{k2},{n}]")
            return (False, f"AddMM dims invalid")


class PaddingEquivalenceVerifier:
    @staticmethod
    def verify_padding_correctness(m: int, k: int, n: int, m_pad: int, k_pad: int, n_pad: int, align: int) -> Tuple[bool, str]:
        s = Solver()
        s.set("timeout", 2000)
        
        pads_nonneg = Bool('pads_nonneg')
        m_aligned = Bool('m_aligned')
        k_aligned = Bool('k_aligned')
        n_aligned = Bool('n_aligned')
        
        s.add(pads_nonneg == (m_pad >= 0 and k_pad >= 0 and n_pad >= 0))
        s.add(m_aligned == ((m + m_pad) % align == 0 if m_pad > 0 else True))
        s.add(k_aligned == ((k + k_pad) % align == 0 if k_pad > 0 else True))
        s.add(n_aligned == ((n + n_pad) % align == 0 if n_pad > 0 else True))
        s.add(And(pads_nonneg, m_aligned, k_aligned, n_aligned))
        
        if s.check() == sat:
            return (True, f"Padding correct: [{m},{k}]@[{k},{n}] + pads({m_pad},{k_pad},{n_pad})")
        return (False, "Padding correctness failed")


class ComputeBoundVerifier:
    @staticmethod
    def verify_compute_bound(M: int, K: int, N: int, dtype: torch.dtype, machine_balance: float, expected: bool) -> Tuple[bool, str]:
        s = Solver()
        s.set("timeout", 1500)
        
        denom = M * K + N * K + M * N
        if denom == 0:
            return (False, "Denominator zero")
        
        arith_intensity = (M * N * K) / denom
        
        is_bf16_special = (
            dtype == torch.bfloat16 and 
            K > M and K > N and 
            N % 2 == 1 and 
            K >= 8388608
        )
        
        if is_bf16_special:
            is_compute_bound = True
        else:
            is_compute_bound = (arith_intensity > machine_balance)
        
        is_valid = Bool('is_valid')
        s.add(is_valid == (is_compute_bound == expected))
        s.add(is_valid)
        
        if s.check() == sat:
            return (True, f"Compute bound verified: M={M}, K={K}, N={N}, AI={arith_intensity:.2f}")
        return (False, f"Compute bound mismatch: expected {expected}, got {is_compute_bound}")


class PaddingExclusionVerifier:
    @staticmethod
    def verify_exclusion_logic(is_contiguous: bool, node_op: str, node_target: str, expected: bool) -> Tuple[bool, str]:
        s = Solver()
        s.set("timeout", 1000)
        
        contiguous_ok = Bool('contiguous_ok')
        not_fixed_output = Bool('not_fixed_output')
        not_placeholder = Bool('not_placeholder')
        
        s.add(contiguous_ok == is_contiguous)
        
        cannot_plan = [
            "aten.mm.default", "aten.convolution.default", 
            "aten.convolution_backward.default", "aten.bmm.default",
            "aten.addmm.default", 
            "aten._scaled_dot_product_flash_attention.default",
            "aten._scaled_dot_product_efficient_attention.default"
        ]
        s.add(not_fixed_output == (node_target not in cannot_plan))
        s.add(not_placeholder == (node_op != "placeholder"))
        
        can_exclude = And(contiguous_ok, not_fixed_output, not_placeholder)
        
        if s.check() == sat:
            model = s.model()
            calculated = (model[contiguous_ok] and model[not_fixed_output] and model[not_placeholder])
            if calculated == expected:
                return (True, f"Exclusion verified: {expected}")
            return (False, f"Exclusion mismatch: expected {expected}, got {calculated}")
        return (False, "Exclusion check failed")


class PaddingDecisionVerifier:
    @staticmethod
    def verify_should_pad(ori_time: float, pad_time: float, multiplier: float, expected: bool) -> Tuple[bool, str]:
        s = Solver()
        s.set("timeout", 1000)
        
        times_valid = Bool('times_valid')
        mult_valid = Bool('mult_valid')
        decision_correct = Bool('decision_correct')
        
        s.add(times_valid == (ori_time > 0 and pad_time > 0))
        s.add(mult_valid == (multiplier >= 1.0))
        
        should_pad_calc = (ori_time > pad_time * multiplier)
        s.add(decision_correct == (should_pad_calc == expected))
        s.add(And(times_valid, mult_valid, decision_correct))
        
        if s.check() == sat:
            speedup = ori_time / pad_time if pad_time > 0 else float('inf')
            return (True, f"Padding decision verified: {ori_time:.6f} vs {pad_time:.6f}*{multiplier} = {should_pad_calc} (speedup={speedup:.2f}x)")
        return (False, f"Decision mismatch: expected {expected}, calc {should_pad_calc}")


class BF16PaddingVerifier:
    @staticmethod
    def verify_bf16_special_case(dtype: torch.dtype, M: int, N: int, K: int, threshold: int, expected: bool) -> Tuple[bool, str]:
        s = Solver()
        s.set("timeout", 1000)
        
        is_bf16 = Bool('is_bf16')
        k_gt_m = Bool('k_gt_m')
        k_gt_n = Bool('k_gt_n')
        n_odd = Bool('n_odd')
        k_large = Bool('k_large')
        
        s.add(is_bf16 == (dtype == torch.bfloat16))
        s.add(k_gt_m == (K > M))
        s.add(k_gt_n == (K > N))
        s.add(n_odd == (N % 2 == 1))
        s.add(k_large == (K >= threshold))
        
        should_pad = And(is_bf16, k_gt_m, k_gt_n, n_odd, k_large)
        
        if s.check() == sat:
            model = s.model()
            calc = all([model[is_bf16], model[k_gt_m], model[k_gt_n], model[n_odd], model[k_large]])
            if calc == expected:
                return (True, f"BF16 special case verified: {expected}")
            return (False, f"BF16 mismatch: expected {expected}, got {calc}")
        return (False, "BF16 check failed")


class BiasPaddingVerifier:
    @staticmethod
    def verify_bias_padding(bias_shape: List[int], bias_dim: int, pad_length: int, dim_to_pad: int, expected_should_pad: bool) -> Tuple[bool, str]:
        s = Solver()
        s.set("timeout", 1000)
        
        pad_nonzero = Bool('pad_nonzero')
        correct_dim = Bool('correct_dim')
        shape_not_one = Bool('shape_not_one')
        should_pad_bias = Bool('should_pad_bias')
        
        s.add(pad_nonzero == (pad_length != 0))
        s.add(correct_dim == (bias_dim == dim_to_pad or (bias_dim == 1 and dim_to_pad == 0)))
        
        if dim_to_pad < len(bias_shape):
            s.add(shape_not_one == (bias_shape[dim_to_pad] != 1))
        else:
            s.add(shape_not_one == False)
        
        s.add(should_pad_bias == And(pad_nonzero, correct_dim, shape_not_one))
        
        if s.check() == sat:
            model = s.model()
            calc = model[should_pad_bias]
            if calc == expected_should_pad:
                return (True, f"Bias padding verified: shape={bias_shape}, dim={bias_dim}, pad={pad_length}, dim_to_pad={dim_to_pad}")
            return (False, f"Bias padding mismatch: expected {expected_should_pad}, got {calc}")
        return (False, "Bias padding check failed")


class PadMMZ3Verifier:
    def __init__(self):
        self.stats = {'total': 0, 'verified': 0, 'failed': 0, 'skipped': 0}
    
    def verify_mm_padding(self, m: int, k: int, n: int, m_pad: int, k_pad: int, n_pad: int, dtype: torch.dtype, align: int) -> bool:
        self.stats['total'] += 1
        
        valid, msg = DimensionVerifier.verify_mm_dims(m, k, k, n)
        if not valid:
            log.warning(f"[Z3] {msg}")
            self.stats['failed'] += 1
            return False
        
        valid, msg = PaddingLengthVerifier.verify_padded_length(m, align, m_pad)
        if not valid:
            log.warning(f"[Z3] {msg}")
            self.stats['failed'] += 1
            return False
        
        valid, msg = PaddingLengthVerifier.verify_padded_length(k, align, k_pad)
        if not valid:
            log.warning(f"[Z3] {msg}")
            self.stats['failed'] += 1
            return False
        
        valid, msg = PaddingLengthVerifier.verify_padded_length(n, align, n_pad)
        if not valid:
            log.warning(f"[Z3] {msg}")
            self.stats['failed'] += 1
            return False
        
        valid, msg = PaddingEquivalenceVerifier.verify_padding_correctness(m, k, n, m_pad, k_pad, n_pad, align)
        if not valid:
            log.warning(f"[Z3] {msg}")
            self.stats['failed'] += 1
            return False
        
        valid, msg = AlignmentVerifier.verify_alignment_size(dtype, align)
        if not valid:
            log.warning(f"[Z3] {msg}")
            self.stats['failed'] += 1
            return False
        
        log.debug(f"[Z3] MM padding verified: [{m},{k}]@[{k},{n}] + pads({m_pad},{k_pad},{n_pad})")
        self.stats['verified'] += 1
        return True
    
    def verify_conv_padding(self, m: int, n: int, k: int, s: int, p: int, 
                           out_h: int, out_w: int) -> bool:
        """
        Verify convolution with padding based on arxiv.org/html/2411.19419v1
        """
        self.stats['total'] += 1
        
        valid, msg = ConvolutionPaddingVerifier.verify_valid_convolution(m, n, k, s, p)
        if not valid:
            log.warning(f"[Z3] {msg}")
            self.stats['failed'] += 1
            return False
        
        valid, msg = ConvolutionPaddingVerifier.verify_conv_output_shape(m, n, k, s, p, out_h, out_w)
        if not valid:
            log.warning(f"[Z3] {msg}")
            self.stats['failed'] += 1
            return False
        
        valid, msg = PaddingMatrixVerifier.verify_padding_dimensions(m, n, p)
        if not valid:
            log.warning(f"[Z3] {msg}")
            self.stats['failed'] += 1
            return False
        
        valid, msg = PaddingMatrixVerifier.verify_padding_structure(m, n, p)
        if not valid:
            log.warning(f"[Z3] {msg}")
            self.stats['failed'] += 1
            return False
        
        log.debug(f"[Z3] Conv padding verified: {m}x{n}, kernel {k}x{k}, stride {s}, pad {p} -> {out_h}x{out_w}")
        self.stats['verified'] += 1
        return True
    def verify_gqa(self, batch: int, h_q: int, h_kv: int, seq_len: int, head_dim: int, 
                   alignment: int = 8) -> Tuple[bool, List[str]]:
        """
        Verify GQA (Grouped Query Attention) configuration for safety.
        
        Detects Issue #159469: GQA with odd sequence lengths can cause NaN gradients.
        
        Args:
            batch: Batch size
            h_q: Number of query heads
            h_kv: Number of key/value heads
            seq_len: Sequence length
            head_dim: Head dimension
            alignment: Memory alignment (default 8 for bfloat16)
            
        Returns:
            (is_safe, list_of_issues)
        """
        self.stats['total'] += 1
        
        is_safe, issues = GQAStrideVerifier.verify_gqa_full(
            batch, h_q, h_kv, seq_len, head_dim, alignment
        )
        
        if is_safe:
            self.stats['verified'] += 1
            log.debug(f"[Z3] GQA verified safe: B={batch}, H_q={h_q}, H_kv={h_kv}, S={seq_len}, D={head_dim}")
        else:
            self.stats['failed'] += 1
            for issue in issues:
                log.warning(f"[Z3] GQA issue: {issue}")
        
        return is_safe, issues

    def get_stats(self):
        return self.stats.copy()
    
    def reset_stats(self):
        self.stats = {'total': 0, 'verified': 0, 'failed': 0, 'skipped': 0}


_verifier_instance = None

def get_verifier() -> PadMMZ3Verifier:
    global _verifier_instance
    if _verifier_instance is None:
        _verifier_instance = PadMMZ3Verifier()
    return _verifier_instance

def get_gqa_verifier():
    return get_verifier()

