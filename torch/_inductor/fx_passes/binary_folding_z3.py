from z3 import *
import numpy as np
import operator as op
import torch
import torch.fx as fx
from typing import Optional, Tuple, Dict, Any, List
import logging

log = logging.getLogger(__name__)

aten = torch.ops.aten
prims = torch.ops.prims


def vec_eq(x, y):
    if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
        return And(np.vectorize(op.eq)(x, y).flatten().tolist())
    return x == y


def NPRealArray(n, prefix=None):
    return np.array([FreshConst(RealSort(), prefix=prefix) for i in range(n)])


def prove(f):
    s = Solver()
    s.add(Not(f))
    result = s.check()
    if result == unsat:
        return True
    elif result == sat:
        return False
    else:
        return None


def prove2(f):
    s = Solver()
    s.add(Not(f))
    if s.check() == unsat:
        return True
    else:
        return False


class ShapeVerifier:
    
    @staticmethod
    def verify_conv_broadcast(weight_shape: List[int], other_shape: List[int]) -> Tuple[bool, str]:
        if len(other_shape) == 0:
            return (True, "other_shape is 0-D tensor (scalar, broadcasts to any shape)")
        
        if len(weight_shape) < len(other_shape):
            return (False, f"weight_shape ({len(weight_shape)}) < other_shape ({len(other_shape)})")
        
        if len(weight_shape) == len(other_shape) + 1:
            for i in reversed(range(len(other_shape))):
                if i == 0 and weight_shape[0] == other_shape[i]:
                    continue
                if other_shape[i] != 1:
                    return (False, f"other_shape[{i}] = {other_shape[i]} != 1")
        else:
            for i in reversed(range(len(other_shape))):
                if i == 1 and weight_shape[0] == other_shape[i]:
                    continue
                if other_shape[i] != 1:
                    return (False, f"other_shape[{i}] = {other_shape[i]} != 1")
        
        return (True, "Broadcasting compatible with conv")
    
    @staticmethod
    def verify_linear_broadcast(weight_shape: List[int], other_shape: List[int], has_reshape: bool) -> Tuple[bool, str]:
        if len(other_shape) == 0:
            return (True, "other_shape is 0-D tensor (scalar, broadcasts to any shape)")
        
        weight_out_features = weight_shape[1]
        
        valid_shapes = [
            tuple([weight_out_features]),
            tuple([1, weight_out_features]),
            tuple([1]),
            tuple([1, 1]),
        ]
        
        if has_reshape:
            valid_shapes.extend([
                tuple([1, 1, weight_out_features]),
                tuple([1, 1, 1]),
            ])
        
        other_tuple = tuple(other_shape)
        if other_tuple in valid_shapes:
            return (True, f"other_shape {other_tuple} is valid for linear")
        else:
            return (False, f"other_shape {other_tuple} not in valid shapes {valid_shapes}")


class ConvolutionFoldingProver:
    
    @staticmethod
    def prove_add_folding(weight_shape: List[int], other_shape: List[int]) -> Tuple[bool, str]:
        valid, msg = ShapeVerifier.verify_conv_broadcast(weight_shape, other_shape)
        if not valid:
            return (False, f"Broadcasting check failed: {msg}")
        
        x = Real('x')
        w = Real('w')
        b = Real('b')
        c = Real('c')
        
        original = (x * w + b) + c
        folded = x * w + (b + c)
        
        claim = (original == folded)
        result = prove(claim)
        
        if result:
            return (True, "Conv add folding verified")
        else:
            return (False, "Conv add folding FAILED")
    
    @staticmethod
    def prove_sub_folding(weight_shape: List[int], other_shape: List[int]) -> Tuple[bool, str]:
        valid, msg = ShapeVerifier.verify_conv_broadcast(weight_shape, other_shape)
        if not valid:
            return (False, f"Broadcasting check failed: {msg}")
        
        x = Real('x')
        w = Real('w')
        b = Real('b')
        c = Real('c')
        
        original = (x * w + b) - c
        folded = x * w + (b - c)
        
        claim = (original == folded)
        result = prove(claim)
        
        if result:
            return (True, "Conv sub folding verified")
        else:
            return (False, "Conv sub folding FAILED")
    
    @staticmethod
    def prove_mul_folding(weight_shape: List[int], other_shape: List[int], has_bias: bool) -> Tuple[bool, str]:
        valid, msg = ShapeVerifier.verify_conv_broadcast(weight_shape, other_shape)
        if not valid:
            return (False, f"Broadcasting check failed: {msg}")
        
        x = Real('x')
        w = Real('w')
        c = Real('c')
        
        if has_bias:
            b = Real('b')
            original = (x * w + b) * c
            folded = x * (w * c) + (b * c)
        else:
            original = (x * w) * c
            folded = x * (w * c)
        
        claim = (original == folded)
        result = prove(claim)
        
        if result:
            return (True, "Conv mul folding verified")
        else:
            return (False, "Conv mul folding FAILED")
    
    @staticmethod
    def prove_div_folding(weight_shape: List[int], other_shape: List[int], has_bias: bool) -> Tuple[bool, str]:
        valid, msg = ShapeVerifier.verify_conv_broadcast(weight_shape, other_shape)
        if not valid:
            return (False, f"Broadcasting check failed: {msg}")
        
        x = Real('x')
        w = Real('w')
        c = Real('c')
        
        if has_bias:
            b = Real('b')
            original = (x * w + b) / c
            folded = x * (w / c) + (b / c)
        else:
            original = (x * w) / c
            folded = x * (w / c)
        
        claim = Implies(c != 0, original == folded)
        result = prove(claim)
        
        if result:
            return (True, "Conv div folding verified")
        else:
            return (False, "Conv div folding FAILED")


class LinearFoldingProver:
    
    @staticmethod
    def prove_add_folding(weight_shape: List[int], other_shape: List[int], has_reshape: bool, 
                         alpha: float = 1.0, beta: float = 1.0) -> Tuple[bool, str]:
        valid, msg = ShapeVerifier.verify_linear_broadcast(weight_shape, other_shape, has_reshape)
        if not valid:
            return (False, f"Broadcasting check failed: {msg}")
        
        x = Real('x')
        w = Real('w')
        b = Real('b')
        c = Real('c')
        alpha_z3 = RealVal(alpha)
        beta_z3 = RealVal(beta)
        
        original = beta_z3 * b + alpha_z3 * (x * w) + c
        
        if beta != 0:
            folded = beta_z3 * (b + c / beta_z3) + alpha_z3 * (x * w)
            claim = (original == folded)
        else:
            folded = c + alpha_z3 * (x * w)
            claim = (original == folded)
        
        result = prove(claim)
        
        if result:
            return (True, f"Linear add folding verified (alpha={alpha}, beta={beta})")
        else:
            return (False, f"Linear add folding FAILED (alpha={alpha}, beta={beta})")
    
    @staticmethod
    def prove_sub_folding(weight_shape: List[int], other_shape: List[int], has_reshape: bool,
                         alpha: float = 1.0, beta: float = 1.0) -> Tuple[bool, str]:
        valid, msg = ShapeVerifier.verify_linear_broadcast(weight_shape, other_shape, has_reshape)
        if not valid:
            return (False, f"Broadcasting check failed: {msg}")
        
        x = Real('x')
        w = Real('w')
        b = Real('b')
        c = Real('c')
        alpha_z3 = RealVal(alpha)
        beta_z3 = RealVal(beta)
        
        original = beta_z3 * b + alpha_z3 * (x * w) - c
        
        if beta != 0:
            folded = beta_z3 * (b - c / beta_z3) + alpha_z3 * (x * w)
            claim = (original == folded)
        else:
            folded = -c + alpha_z3 * (x * w)
            claim = (original == folded)
        
        result = prove(claim)
        
        if result:
            return (True, f"Linear sub folding verified (alpha={alpha}, beta={beta})")
        else:
            return (False, f"Linear sub folding FAILED (alpha={alpha}, beta={beta})")
    
    @staticmethod
    def prove_mul_folding(weight_shape: List[int], other_shape: List[int], has_reshape: bool, has_bias: bool,
                         alpha: float = 1.0, beta: float = 1.0) -> Tuple[bool, str]:
        valid, msg = ShapeVerifier.verify_linear_broadcast(weight_shape, other_shape, has_reshape)
        if not valid:
            return (False, f"Broadcasting check failed: {msg}")
        
        x = Real('x')
        w = Real('w')
        c = Real('c')
        alpha_z3 = RealVal(alpha)
        beta_z3 = RealVal(beta)
        
        if has_bias:
            b = Real('b')
            original = (beta_z3 * b + alpha_z3 * (x * w)) * c
            folded = beta_z3 * (b * c) + (alpha_z3 * c) * (x * w)
        else:
            original = (alpha_z3 * (x * w)) * c
            folded = (alpha_z3 * c) * (x * w)
        
        claim = (original == folded)
        result = prove(claim)
        
        if result:
            return (True, f"Linear mul folding verified (alpha={alpha}, beta={beta})")
        else:
            return (False, f"Linear mul folding FAILED (alpha={alpha}, beta={beta})")
    
    @staticmethod
    def prove_div_folding(weight_shape: List[int], other_shape: List[int], has_reshape: bool, has_bias: bool,
                         alpha: float = 1.0, beta: float = 1.0) -> Tuple[bool, str]:
        valid, msg = ShapeVerifier.verify_linear_broadcast(weight_shape, other_shape, has_reshape)
        if not valid:
            return (False, f"Broadcasting check failed: {msg}")
        
        x = Real('x')
        w = Real('w')
        c = Real('c')
        alpha_z3 = RealVal(alpha)
        beta_z3 = RealVal(beta)
        
        if has_bias:
            b = Real('b')
            original = (beta_z3 * b + alpha_z3 * (x * w)) / c
            folded = beta_z3 * (b / c) + (alpha_z3 / c) * (x * w)
        else:
            original = (alpha_z3 * (x * w)) / c
            folded = (alpha_z3 / c) * (x * w)
        
        claim = Implies(c != 0, original == folded)
        result = prove(claim)
        
        if result:
            return (True, f"Linear div folding verified (alpha={alpha}, beta={beta})")
        else:
            return (False, f"Linear div folding FAILED (alpha={alpha}, beta={beta})")


class MatrixMultiplyFoldingProver:
    
    @staticmethod
    def prove_add_folding(weight_shape: List[int], other_shape: List[int], has_reshape: bool) -> Tuple[bool, str]:
        valid, msg = ShapeVerifier.verify_linear_broadcast(weight_shape, other_shape, has_reshape)
        if not valid:
            return (False, f"Broadcasting check failed: {msg}")
        
        x = Real('x')
        w = Real('w')
        c = Real('c')
        
        original = (x * w) + c
        folded = c + (x * w)
        
        claim = (original == folded)
        result = prove(claim)
        
        if result:
            return (True, "MM add folding verified (converts to addmm)")
        else:
            return (False, "MM add folding FAILED")
    
    @staticmethod
    def prove_sub_folding(weight_shape: List[int], other_shape: List[int], has_reshape: bool) -> Tuple[bool, str]:
        valid, msg = ShapeVerifier.verify_linear_broadcast(weight_shape, other_shape, has_reshape)
        if not valid:
            return (False, f"Broadcasting check failed: {msg}")
        
        x = Real('x')
        w = Real('w')
        c = Real('c')
        
        original = (x * w) - c
        folded = -c + (x * w)
        
        claim = (original == folded)
        result = prove(claim)
        
        if result:
            return (True, "MM sub folding verified (converts to addmm)")
        else:
            return (False, "MM sub folding FAILED")
    
    @staticmethod
    def prove_mul_folding(weight_shape: List[int], other_shape: List[int], has_reshape: bool) -> Tuple[bool, str]:
        valid, msg = ShapeVerifier.verify_linear_broadcast(weight_shape, other_shape, has_reshape)
        if not valid:
            return (False, f"Broadcasting check failed: {msg}")
        
        x = Real('x')
        w = Real('w')
        c = Real('c')
        
        original = (x * w) * c
        folded = x * (w * c)
        
        claim = (original == folded)
        result = prove(claim)
        
        if result:
            return (True, "MM mul folding verified")
        else:
            return (False, "MM mul folding FAILED")
    
    @staticmethod
    def prove_div_folding(weight_shape: List[int], other_shape: List[int], has_reshape: bool) -> Tuple[bool, str]:
        valid, msg = ShapeVerifier.verify_linear_broadcast(weight_shape, other_shape, has_reshape)
        if not valid:
            return (False, f"Broadcasting check failed: {msg}")
        
        x = Real('x')
        w = Real('w')
        c = Real('c')
        
        original = (x * w) / c
        folded = x * (w / c)
        
        claim = Implies(c != 0, original == folded)
        result = prove(claim)
        
        if result:
            return (True, "MM div folding verified")
        else:
            return (False, "MM div folding FAILED")


class TypePromotionVerifier:
    
    @staticmethod
    def verify_no_type_promotion(weight_dtype: torch.dtype, other_dtype: torch.dtype) -> Tuple[bool, str]:
        if weight_dtype not in [torch.float32, torch.float16, torch.bfloat16, torch.float64]:
            return (False, f"Weight dtype {weight_dtype} is not floating point")
        
        if other_dtype not in [torch.float32, torch.float16, torch.bfloat16, torch.float64]:
            return (False, f"Other dtype {other_dtype} is not floating point")
        
        promoted = torch.promote_types(other_dtype, weight_dtype)
        if promoted != weight_dtype:
            return (False, f"Type promotion: {other_dtype} + {weight_dtype} -> {promoted} != {weight_dtype}")
        
        return (True, "No unwanted type promotion")
    
    @staticmethod
    def verify_mixed_precision_allowed(weight_dtype: torch.dtype, other_dtype: torch.dtype) -> Tuple[bool, str]:
        if other_dtype != torch.float32:
            return (False, f"Mixed precision requires other_dtype=float32, got {other_dtype}")
        
        if weight_dtype not in [torch.float16, torch.bfloat16]:
            return (False, f"Mixed precision requires weight_dtype in [float16, bfloat16], got {weight_dtype}")
        
        return (True, "Mixed precision folding allowed")


class VectorFoldingProver:
    
    @staticmethod
    def prove_conv_add_vector(vec_size: int = 4) -> Tuple[bool, str]:
        x = NPRealArray(vec_size, prefix="x")
        w = NPRealArray(vec_size, prefix="w")
        b = NPRealArray(vec_size, prefix="b")
        c = Real('c')
        
        original = (x * w + b) + c
        folded = x * w + (b + c)
        
        claim = vec_eq(original, folded)
        result = prove(claim)
        
        if result:
            return (True, f"Conv add vector folding verified (n={vec_size})")
        else:
            return (False, f"Conv add vector folding FAILED (n={vec_size})")
    
    @staticmethod
    def prove_conv_mul_vector(vec_size: int = 4) -> Tuple[bool, str]:
        x = NPRealArray(vec_size, prefix="x")
        w = NPRealArray(vec_size, prefix="w")
        b = NPRealArray(vec_size, prefix="b")
        c = Real('c')
        
        original = (x * w + b) * c
        folded = x * (w * c) + (b * c)
        
        claim = vec_eq(original, folded)
        result = prove(claim)
        
        if result:
            return (True, f"Conv mul vector folding verified (n={vec_size})")
        else:
            return (False, f"Conv mul vector folding FAILED (n={vec_size})")
    
    @staticmethod
    def prove_distributivity_vector(vec_size: int = 4) -> Tuple[bool, str]:
        x = NPRealArray(vec_size, prefix="x")
        y = NPRealArray(vec_size, prefix="y")
        c = Real('c')
        
        original = (x + y) * c
        folded = (x * c) + (y * c)
        
        claim = vec_eq(original, folded)
        result = prove(claim)
        
        if result:
            return (True, f"Vector distributivity verified (n={vec_size})")
        else:
            return (False, f"Vector distributivity FAILED (n={vec_size})")


class PreconditionChecker:
    
    @staticmethod
    def check_weight_node_z3(weight_node: fx.Node) -> Tuple[bool, str]:
        s = Solver()
        s.set("timeout", 1000)
        
        is_get_attr = Bool('is_get_attr')
        has_one_user = Bool('has_one_user')
        has_metadata = Bool('has_metadata')
        is_float = Bool('is_float')
        
        is_get_attr_val = (weight_node.op == "get_attr")
        has_one_user_val = (len(weight_node.users) == 1)
        
        weight_meta = weight_node.meta.get("val")
        has_metadata_val = (weight_meta is not None)
        is_float_val = (weight_meta.is_floating_point() if weight_meta is not None else False)
        
        s.add(is_get_attr == is_get_attr_val)
        s.add(has_one_user == has_one_user_val)
        s.add(has_metadata == has_metadata_val)
        s.add(is_float == is_float_val)
        
        requirement = And(is_get_attr, has_one_user, has_metadata, is_float)
        
        s.add(requirement)
        result = s.check()
        
        if result == sat:
            return (True, "Weight node satisfies all Z3 constraints")
        else:
            failures = []
            if not is_get_attr_val:
                failures.append("not get_attr")
            if not has_one_user_val:
                failures.append(f"{len(weight_node.users)} users")
            if not has_metadata_val:
                failures.append("no metadata")
            if not is_float_val:
                failures.append(f"not floating point ({weight_meta.dtype if weight_meta else 'None'})")
            
            return (False, f"Weight node Z3 constraint failed: {', '.join(failures)}")
    
    @staticmethod
    def check_bias_node_z3(bias_node: Optional[fx.Node]) -> Tuple[bool, str]:
        s = Solver()
        s.set("timeout", 1000)
        
        is_none = Bool('is_none')
        is_get_attr = Bool('is_get_attr')
        
        is_none_val = (bias_node is None)
        is_get_attr_val = (bias_node.op == "get_attr" if bias_node is not None else False)
        
        s.add(is_none == is_none_val)
        s.add(is_get_attr == is_get_attr_val)
        
        requirement = Or(is_none, is_get_attr)
        s.add(requirement)
        
        result = s.check()
        
        if result == sat:
            return (True, "Bias node satisfies Z3 constraint (None or get_attr)")
        else:
            return (False, f"Bias node Z3 constraint failed: op='{bias_node.op if bias_node else None}'")
    
    @staticmethod
    def check_other_operand_z3(other: Any) -> Tuple[bool, str]:
        s = Solver()
        s.set("timeout", 1000)
        
        is_scalar_constant = Bool('is_scalar_constant')
        is_node = Bool('is_node')
        is_get_attr = Bool('is_get_attr')
        
        is_scalar_constant_val = isinstance(other, (int, float))
        
        is_node_val = isinstance(other, fx.Node)
        is_get_attr_val = (other.op == "get_attr" if is_node_val else False)
        
        s.add(is_scalar_constant == is_scalar_constant_val)
        s.add(is_node == is_node_val)
        s.add(is_get_attr == is_get_attr_val)
        
        requirement = Or(
            is_scalar_constant,
            And(is_node, is_get_attr)
        )
        s.add(requirement)
        
        result = s.check()
        
        if result == sat:
            model = s.model()
            if model[is_scalar_constant]:
                return (True, f"Other is scalar constant: {other}")
            else:
                return (True, "Other is constant tensor (get_attr)")
        else:
            return (False, f"Other has invalid type or op: {type(other)}")
    
    @staticmethod
    def check_type_promotion_z3(weight_dtype: torch.dtype, 
                                other: Any,
                                computation_node: fx.Node) -> Tuple[bool, str]:
        s = Solver()
        s.set("timeout", 1000)
        
        if not (isinstance(other, fx.Node) and other.op == "get_attr"):
            return (True, "Other is scalar (type promotion N/A)")
        
        other_meta = other.meta.get("val")
        if other_meta is None:
            return (False, "Other metadata missing (cannot verify Z3 constraint)")
        
        other_is_float = Bool('other_is_float')
        has_type_promotion = Bool('has_type_promotion')
        mixed_allowed = Bool('mixed_allowed')
        other_is_float32 = Bool('other_is_float32')
        weight_is_low_precision = Bool('weight_is_low_precision')
        
        other_is_float_val = other_meta.is_floating_point()
        
        promoted = torch.promote_types(other_meta.dtype, weight_dtype)
        has_type_promotion_val = (promoted != weight_dtype)
        
        mixed_allowed_raw = computation_node.meta.get("_allow_mixed_dtype_folding", False)
        if isinstance(mixed_allowed_raw, bool):
            mixed_allowed_val = mixed_allowed_raw
        else:
            # If metadata contains unexpected type (like torch.dtype), default to False
            mixed_allowed_val = False
            
        #mixed_allowed_val = computation_node.meta.get("_allow_mixed_dtype_folding", False)
        other_is_float32_val = (other_meta.dtype == torch.float32)
        weight_is_low_precision_val = (weight_dtype in (torch.float16, torch.bfloat16))
        
        s.add(other_is_float == other_is_float_val)
        s.add(has_type_promotion == has_type_promotion_val)
        s.add(mixed_allowed == mixed_allowed_val)
        s.add(other_is_float32 == other_is_float32_val)
        s.add(weight_is_low_precision == weight_is_low_precision_val)
        
        requirement = And(
            other_is_float,
            Or(
                Not(has_type_promotion),
                And(
                    has_type_promotion,
                    mixed_allowed,
                    other_is_float32,
                    weight_is_low_precision
                )
            )
        )
        
        s.add(requirement)
        result = s.check()
        
        if result == sat:
            return (True, f"Type promotion Z3 constraint satisfied ({other_meta.dtype} + {weight_dtype})")
        else:
            if not other_is_float_val:
                return (False, f"Other dtype {other_meta.dtype} is not floating point")
            
            if has_type_promotion_val:
                if not mixed_allowed_val:
                    return (False, f"Type promotion {other_meta.dtype} -> {promoted} not allowed (set _allow_mixed_dtype_folding)")
                if not other_is_float32_val:
                    return (False, f"Mixed dtype requires other=float32, got {other_meta.dtype}")
                if not weight_is_low_precision_val:
                    return (False, f"Mixed dtype requires weight in {{float16,bfloat16}}, got {weight_dtype}")
            
            return (False, "Type promotion Z3 constraint failed (unknown reason)")
    
    @staticmethod
    def check_broadcasting_z3(comp_target: Any,
                             weight_node: fx.Node,
                             other: Any,
                             has_reshape: bool) -> Tuple[bool, str]:
        if isinstance(other, (int, float)):
            return (True, "Scalar constant (broadcasts to any shape)")
        
        weight_meta = weight_node.meta.get("val")
        if weight_meta is None:
            return (False, "Weight metadata missing")
        
        if not isinstance(other, fx.Node) or other.op != "get_attr":
            return (False, "Other is not a constant tensor")
        
        other_meta = other.meta.get("val")
        if other_meta is None:
            return (False, "Other metadata missing")
        
        if other_meta.ndim == 0 or other_meta.numel() == 1:
            return (True, "Other is 0-D tensor or single-element tensor (scalar, broadcasts to any shape)")
        
        
        weight_shape = list(weight_meta.shape)
        other_shape = list(other_meta.shape)
        
        if len(other_shape) == 0:
            return (True, "Other is 0-D tensor (scalar, broadcasts to any shape)")
        
        if comp_target == aten.convolution.default:
            return PreconditionChecker._check_conv_broadcast_z3(weight_shape, other_shape)
        elif comp_target in [aten.addmm.default, aten.mm.default]:
            return PreconditionChecker._check_linear_broadcast_z3(weight_shape, other_shape, has_reshape)
        else:
            return (False, f"Unknown computation target: {comp_target}")
    
    @staticmethod
    def _check_conv_broadcast_z3(weight_shape: List[int], other_shape: List[int]) -> Tuple[bool, str]:
        s = Solver()
        s.set("timeout", 2000)
        
        len_valid = Bool('len_valid')
        case_plus_one = Bool('case_plus_one')
        channel_match = Bool('channel_match')
        others_are_one = Bool('others_are_one')
        
        len_valid_val = (len(weight_shape) >= len(other_shape))
        case_plus_one_val = (len(weight_shape) == len(other_shape) + 1)
        
        if not len_valid_val:
            return (False, f"[Z3] Conv broadcast: len(weight)={len(weight_shape)} < len(other)={len(other_shape)}")
        
        if case_plus_one_val:
            channel_match_val = (other_shape[0] == weight_shape[0] or other_shape[0] == 1)
            others_are_one_val = all(other_shape[i] == 1 for i in range(1, len(other_shape)))
        else:
            if len(other_shape) > 1:
                channel_match_val = (other_shape[1] == weight_shape[0] or other_shape[0] == 1)
                others_are_one_val = (
                    other_shape[0] == 1 and
                    all(other_shape[i] == 1 for i in range(2, len(other_shape)))
                )
            else:
                #channel_match_val = False
                #others_are_one_val = False
                channel_match_val = (other_shape[0] == 1)
                others_are_one_val = True
        
        s.add(len_valid == len_valid_val)
        s.add(case_plus_one == case_plus_one_val)
        s.add(channel_match == channel_match_val)
        s.add(others_are_one == others_are_one_val)
        
        requirement = And(len_valid, channel_match, others_are_one)
        s.add(requirement)
        
        result = s.check()
        
        if result == sat:
            return (True, f"[Z3] Conv broadcast valid: {weight_shape} x {other_shape}")
        else:
            if not channel_match_val:
                expected_dim = 0 if case_plus_one_val else 1
                return (False, f"[Z3] Conv broadcast: other_shape[{expected_dim}]={other_shape[expected_dim] if expected_dim < len(other_shape) else 'N/A'} != {weight_shape[0]}")
            if not others_are_one_val:
                return (False, f"[Z3] Conv broadcast: other dims must be 1, got {other_shape}")
            return (False, "[Z3] Conv broadcast constraint failed")
    
    @staticmethod
    def _check_linear_broadcast_z3(weight_shape: List[int], other_shape: List[int], has_reshape: bool) -> Tuple[bool, str]:
        s = Solver()
        s.set("timeout", 2000)
        if len(other_shape) == 0:
            return (True, "[Z3] Linear broadcast: 0-D tensor (scalar) broadcasts to any shape")
        
        out_features = weight_shape[1] if len(weight_shape) > 1 else weight_shape[0]
        
        valid_shapes = [
            tuple([out_features]),
            tuple([1, out_features]),
            tuple([1]),
            tuple([1, 1]),
        ]
        
        if has_reshape:
            valid_shapes.extend([
                tuple([1, 1, out_features]),
                tuple([1, 1, 1]),
            ])
        
        is_valid = Bool('is_valid')
        
        other_tuple = tuple(other_shape)
        is_valid_val = (other_tuple in valid_shapes)
        
        s.add(is_valid == is_valid_val)
        s.add(is_valid)
        
        result = s.check()
        
        if result == sat:
            return (True, f"[Z3] Linear broadcast valid: {weight_shape} @ {other_shape}")
        else:
            return (False, f"[Z3] Linear broadcast: {other_shape} not in valid shapes {valid_shapes}")
    
    @staticmethod
    def check_all_preconditions(match: Any,
                               binary_node: fx.Node,
                               computation_node: fx.Node,
                               other: Any) -> Tuple[bool, str]:
        comp_target = computation_node.target
        
        has_reshape = False
        if isinstance(binary_node.args[0], fx.Node) and binary_node.args[0].target == aten.reshape.default:
            has_reshape = True
        elif len(binary_node.args) > 1 and isinstance(binary_node.args[1], fx.Node) and binary_node.args[1].target == aten.reshape.default:
            has_reshape = True
        
        if comp_target == aten.convolution.default:
            weight_node = computation_node.args[1]
            bias_node = computation_node.args[2]
        elif comp_target == aten.addmm.default:
            weight_node = computation_node.args[2]
            bias_node = computation_node.args[0]
        elif comp_target == aten.mm.default:
            weight_node = computation_node.args[1]
            bias_node = None
        else:
            return (False, f"Unknown computation target: {comp_target}")
        
        valid, msg = PreconditionChecker.check_weight_node_z3(weight_node)
        if not valid:
            return (False, f"[Z3] Check 1/5 failed - Weight: {msg}")
        
        valid, msg = PreconditionChecker.check_bias_node_z3(bias_node)
        if not valid:
            return (False, f"[Z3] Check 2/5 failed - Bias: {msg}")
        
        valid, msg = PreconditionChecker.check_other_operand_z3(other)
        if not valid:
            return (False, f"[Z3] Check 3/5 failed - Other operand: {msg}")
        
        weight_meta = weight_node.meta.get("val")
        valid, msg = PreconditionChecker.check_type_promotion_z3(
            weight_meta.dtype, other, computation_node
        )
        if not valid:
            return (False, f"[Z3] Check 4/5 failed - Type promotion: {msg}")
        
        valid, msg = PreconditionChecker.check_broadcasting_z3(
            comp_target, weight_node, other, has_reshape
        )
        if not valid:
            return (False, f"[Z3] Check 5/5 failed - Broadcasting: {msg}")
        
        return (True, "[Z3] All 5 precondition checks satisfied via SMT solving")


class BinaryFoldingZ3Verifier:
    
    def __init__(self, timeout: int = 5000, check_preconditions: bool = True):
        self.timeout = timeout
        self.check_preconditions = check_preconditions
        self.stats = {
            'total': 0,
            'verified': 0,
            'failed': 0,
            'skipped': 0,
            'precondition_failed': 0,
        }
    
    def verify_folding_pattern(self,
                               match: Any,
                               binary_node: fx.Node,
                               computation_node: fx.Node,
                               other: Any) -> Tuple[bool, str]:
        self.stats['total'] += 1
        
        if self.check_preconditions:
            valid, msg = PreconditionChecker.check_all_preconditions(
                match, binary_node, computation_node, other
            )
            if not valid:
                self.stats['precondition_failed'] += 1
                return (False, f"Precondition check failed: {msg}")
        
        comp_target = computation_node.target
        binary_target = binary_node.target
        
        if comp_target == aten.convolution.default:
            weight_node = computation_node.args[1]
            bias_node = computation_node.args[2]
        elif comp_target == aten.addmm.default:
            weight_node = computation_node.args[2]
            bias_node = computation_node.args[0]
        elif comp_target == aten.mm.default:
            weight_node = computation_node.args[1]
            bias_node = None
        else:
            self.stats['skipped'] += 1
            return (False, f"Unknown computation op: {comp_target}")
        
        weight_meta = weight_node.meta.get('val')
        if weight_meta is None:
            self.stats['skipped'] += 1
            return (False, "No weight metadata")
        
        weight_shape = list(weight_meta.shape)
        has_bias = bias_node is not None
        
        if isinstance(other, fx.Node):
            other_meta = other.meta.get('val')
            if other_meta is None:
                self.stats['skipped'] += 1
                return (False, "No other metadata")
            other_shape = list(other_meta.shape)
            other_dtype = other_meta.dtype
        else:
            other_shape = []
            other_dtype = torch.float32
        
        has_reshape = False
        if binary_node.args[0].target == aten.reshape.default:
            has_reshape = True
        elif len(binary_node.args) > 1 and isinstance(binary_node.args[1], fx.Node) and binary_node.args[1].target == aten.reshape.default:
            has_reshape = True
        
        if comp_target == aten.convolution.default:
            result = self._verify_conv_folding(
                binary_target, weight_shape, other_shape, has_bias
            )
        elif comp_target in [aten.addmm.default, aten.mm.default]:
            result = self._verify_linear_folding(
                binary_target, comp_target, weight_shape, other_shape, has_bias, has_reshape
            )
        else:
            result = (False, f"Unsupported computation: {comp_target}")
        
        if result[0]:
            self.stats['verified'] += 1
        else:
            self.stats['failed'] += 1
        
        return result
    
    def _verify_conv_folding(self,
                            binary_op,
                            weight_shape: List[int],
                            other_shape: List[int],
                            has_bias: bool) -> Tuple[bool, str]:
        if binary_op == aten.add.Tensor:
            return ConvolutionFoldingProver.prove_add_folding(weight_shape, other_shape)
        elif binary_op == aten.sub.Tensor:
            return ConvolutionFoldingProver.prove_sub_folding(weight_shape, other_shape)
        elif binary_op == aten.mul.Tensor:
            return ConvolutionFoldingProver.prove_mul_folding(weight_shape, other_shape, has_bias)
        elif binary_op == aten.div.Tensor:
            return ConvolutionFoldingProver.prove_div_folding(weight_shape, other_shape, has_bias)
        else:
            return (False, f"Unknown binary op: {binary_op}")
    
    def _verify_linear_folding(self,
                              binary_op,
                              comp_target,
                              weight_shape: List[int],
                              other_shape: List[int],
                              has_bias: bool,
                              has_reshape: bool) -> Tuple[bool, str]:
        is_mm = (comp_target == aten.mm.default)
        is_addmm = (comp_target == aten.addmm.default)
        
        if is_mm:
            if binary_op == aten.add.Tensor:
                return MatrixMultiplyFoldingProver.prove_add_folding(weight_shape, other_shape, has_reshape)
            elif binary_op == aten.sub.Tensor:
                return MatrixMultiplyFoldingProver.prove_sub_folding(weight_shape, other_shape, has_reshape)
            elif binary_op == aten.mul.Tensor:
                return MatrixMultiplyFoldingProver.prove_mul_folding(weight_shape, other_shape, has_reshape)
            elif binary_op == aten.div.Tensor:
                return MatrixMultiplyFoldingProver.prove_div_folding(weight_shape, other_shape, has_reshape)
            else:
                return (False, f"Unknown binary op: {binary_op}")
        
        elif is_addmm:
            alpha, beta = 1.0, 1.0
            
            if binary_op == aten.add.Tensor:
                return LinearFoldingProver.prove_add_folding(weight_shape, other_shape, has_reshape, alpha, beta)
            elif binary_op == aten.sub.Tensor:
                return LinearFoldingProver.prove_sub_folding(weight_shape, other_shape, has_reshape, alpha, beta)
            elif binary_op == aten.mul.Tensor:
                return LinearFoldingProver.prove_mul_folding(weight_shape, other_shape, has_reshape, has_bias, alpha, beta)
            elif binary_op == aten.div.Tensor:
                return LinearFoldingProver.prove_div_folding(weight_shape, other_shape, has_reshape, has_bias, alpha, beta)
            else:
                return (False, f"Unknown binary op: {binary_op}")
        
        else:
            return (False, f"Unknown computation target: {comp_target}")
    
    def get_stats(self) -> Dict[str, int]:
        return self.stats.copy()


_global_verifier: Optional[BinaryFoldingZ3Verifier] = None


def get_verifier() -> BinaryFoldingZ3Verifier:
    global _global_verifier
    if _global_verifier is None:
        _global_verifier = BinaryFoldingZ3Verifier()
    return _global_verifier


def verify_binary_folding_transformation(match, binary_node, computation_node, other) -> bool:
    verifier = get_verifier()
    valid, msg = verifier.verify_folding_pattern(match, binary_node, computation_node, other)
    
    if not valid:
        log.warning(f"[Z3] Binary folding verification FAILED: {msg}")
    else:
        log.debug(f"[Z3] Binary folding verified: {msg}")
    
    return valid


if __name__ == "__main__":
    weight_shape = [64, 3, 3, 3]
    other_shape = [64]
    
    result = ConvolutionFoldingProver.prove_add_folding(weight_shape, other_shape)
    print(f"Conv add: {result[0]}")
    
    result = ConvolutionFoldingProver.prove_mul_folding(weight_shape, other_shape, has_bias=True)
    print(f"Conv mul: {result[0]}")
    
    weight_shape = [512, 1024]
    other_shape = [1024]
    
    result = LinearFoldingProver.prove_add_folding(weight_shape, other_shape, has_reshape=False)
    print(f"Linear add: {result[0]}")
    
    result = MatrixMultiplyFoldingProver.prove_add_folding(weight_shape, other_shape, has_reshape=False)
    print(f"MM add: {result[0]}")
