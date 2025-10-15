import torch
import torch.nn as nn
from torch._dynamo.utils import counters


class TestConvModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3, padding=1, bias=True)
    
    def forward(self, x):
        y1 = self.conv(x) + 0.5
        y2 = self.conv(x) * 2.0
        return y1 + y2


class TestLinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(512, 1024, bias=True)
    
    def forward(self, x):
        y1 = self.linear(x) + 1.0
        y2 = self.linear(x) * 0.5
        return y1 + y2


def test_conv_folding_with_z3():
    model = TestConvModel().eval().cuda()
    x = torch.randn(2, 3, 32, 32, device='cuda')
    
    counters.clear()
    compiled_model = torch.compile(model, backend="inductor")
    
    with torch.no_grad():
        output = compiled_model(x)
    
    binary_folding = counters.get("inductor", {}).get("binary_folding", 0)
    z3_verified = counters.get("inductor", {}).get("binary_folding_z3_verified", 0)
    z3_failed = counters.get("inductor", {}).get("binary_folding_z3_failed", 0)
    
    print(f"Conv: folding={binary_folding} verified={z3_verified} failed={z3_failed}")
    
    assert binary_folding > 0
    assert z3_verified > 0
    assert z3_failed == 0
    return True


def test_linear_folding_with_z3():
    model = TestLinearModel().eval().cuda()
    x = torch.randn(4, 512, device='cuda')
    
    counters.clear()
    compiled_model = torch.compile(model, backend="inductor")
    
    with torch.no_grad():
        output = compiled_model(x)
    
    binary_folding = counters.get("inductor", {}).get("binary_folding", 0)
    z3_verified = counters.get("inductor", {}).get("binary_folding_z3_verified", 0)
    z3_failed = counters.get("inductor", {}).get("binary_folding_z3_failed", 0)
    
    print(f"Linear: folding={binary_folding} verified={z3_verified} failed={z3_failed}")
    
    assert binary_folding > 0
    assert z3_verified > 0
    assert z3_failed == 0
    return True


def test_z3_verifier_directly():
    from torch._inductor.fx_passes.binary_folding_z3 import (
        ConvolutionFoldingProver,
        LinearFoldingProver,
        MatrixMultiplyFoldingProver,
    )
    
    weight_shape = [64, 3, 3, 3]
    other_shape = [64]
    
    result = ConvolutionFoldingProver.prove_add_folding(weight_shape, other_shape)
    assert result[0]
    
    result = ConvolutionFoldingProver.prove_mul_folding(weight_shape, other_shape, has_bias=True)
    assert result[0]
    
    weight_shape = [512, 1024]
    other_shape = [1024]
    
    result = LinearFoldingProver.prove_add_folding(weight_shape, other_shape, has_reshape=False)
    assert result[0]
    
    result = MatrixMultiplyFoldingProver.prove_add_folding(weight_shape, other_shape, has_reshape=False)
    assert result[0]
    
    print("Direct verifier: PASS")
    return True


def main():
    try:
        test_z3_verifier_directly()
        test_conv_folding_with_z3()
        test_linear_folding_with_z3()
        print("ALL TESTS PASSED")
        return True
    except AssertionError as e:
        print(f"FAILED: {e}")
        return False
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)

