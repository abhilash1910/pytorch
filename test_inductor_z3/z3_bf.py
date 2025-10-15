"""
Test binary folding with Z3 verification integration.
Follows pytorch/test/inductor/test_binary_folding.py style.
"""
import torch
from torch import nn
from torch._dynamo.utils import counters
from torch._inductor import config as inductor_config
import itertools


class TestBinaryFoldingZ3:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    @inductor_config.patch({"freezing": True})
    def test_conv_binary_folding_z3(self):
        """Test conv binary folding with Z3 verification"""
        @torch.no_grad()
        def test_conv_fusion(use_bias, module, op, scalar, add_tensor, expect_success, rtol=None, atol=None):
            class ConvOp(nn.Module):
                __constants__ = ["use_scalar"]

                def __init__(self, in_channels, out_channels, device, **kwargs):
                    super().__init__()
                    self.conv = module(
                        in_channels, out_channels, bias=use_bias, **kwargs
                    ).to(device)
                    self.use_scalar = scalar
                    tensor_size = [1 for _ in range(self.conv.weight.ndim)]
                    tensor_size[1] = self.conv.weight.size(0)
                    self.tensor = torch.nn.Parameter(
                        add_tensor if add_tensor is not None else torch.rand(tensor_size).to(device)
                    )
                    self.op = op

                def forward(self, x):
                    x = self.conv(x)
                    if self.use_scalar:
                        return self.op(x, 2.0)
                    else:
                        return self.op(x, self.tensor)

            torch._dynamo.reset()
            counters.clear()
            mod_eager = ConvOp(3, 32, self.device, kernel_size=3, stride=2).eval()
            out_optimized = torch.compile(mod_eager)

            inps = [4, 3, 4]
            if module is nn.Conv2d:
                inps.append(inps[-1])
            if module is nn.Conv3d:
                inps.append(inps[-1])
                inps.append(inps[-1])

            torch.manual_seed(1234)
            inp = torch.rand(inps).to(self.device)
            out_eager = mod_eager(inp)
            out_optimized = out_optimized(inp)
            
            folding_count = counters['inductor']['binary_folding']
            z3_verified_count = counters['inductor'].get('binary_folding_z3_verified', 0)
            z3_failed_count = counters['inductor'].get('binary_folding_z3_failed', 0)
            
            if expect_success:
                print(f"[Conv {module.__name__} bias={use_bias} op={op.__name__} scalar={scalar}] fold={folding_count} z3_ok={z3_verified_count} z3_fail={z3_failed_count}")
            
            assert torch.allclose(out_optimized, out_eager, rtol=rtol or 1e-5, atol=atol or 1e-5)
            if expect_success:
                assert counters["inductor"]["binary_folding"] == 1
            else:
                assert counters["inductor"]["binary_folding"] == 0
            
            return z3_verified_count

        conv_bias = [True, False]
        modules = [nn.Conv1d, nn.Conv2d, nn.Conv3d]
        use_scalar = [True, False]
        ops = [torch.add, torch.sub, torch.mul, torch.div]
        
        total_z3_verified = 0
        for use_bias, module, pytorch_op, scalar in itertools.product(
            conv_bias, modules, ops, use_scalar
        ):
            z3_verified = test_conv_fusion(
                use_bias, module, pytorch_op, scalar,
                add_tensor=None, expect_success=True
            )
            total_z3_verified += z3_verified

        for use_bias, pytorch_op in itertools.product(conv_bias, ops):
            test_conv_fusion(
                use_bias, nn.Conv2d, pytorch_op, False,
                add_tensor=torch.rand(32, 1, 32).to(self.device),
                expect_success=False
            )
            
            z3_verified = test_conv_fusion(
                use_bias, nn.Conv2d, pytorch_op, False,
                add_tensor=torch.rand(1, 1).to(self.device),
                expect_success=True
            )
            total_z3_verified += z3_verified
            
            test_conv_fusion(
                use_bias, nn.Conv2d, pytorch_op, False,
                add_tensor=torch.tensor([2]).to(torch.float64).to(self.device),
                expect_success=False,
                rtol=1.3e-6, atol=1e-5
            )
        
        print(f"\n=== Conv Test Summary ===")
        print(f"Total Z3 verifications: {total_z3_verified}")
        return total_z3_verified

    @inductor_config.patch({"freezing": True})
    def test_conv_bn_folding_z3(self):
        """Test conv+batchnorm folding with Z3 verification"""
        @torch.no_grad()
        def test_conv_fusion(use_bias, module, expect_success):
            class ConvOp(nn.Module):
                def __init__(self, in_channels, out_channels, device, **kwargs):
                    super().__init__()
                    self.conv = module[0](
                        in_channels, out_channels, bias=use_bias, **kwargs
                    ).to(device)
                    self.bn = module[1](out_channels).to(device)

                def forward(self, x):
                    x = self.conv(x)
                    return self.bn(x)

            import functools
            from torch._inductor.compile_fx import compile_fx, compile_fx_inner

            aten = torch.ops.aten
            aten_binary = [
                aten.add.Tensor,
                aten.sub.Tensor,
                aten.mul.Tensor,
                aten.div.Tensor,
            ]
            n_binary_ops = 0

            def my_inner_compile(gm, example_inputs, *args, **kwargs):
                out = compile_fx_inner(gm, example_inputs, *args, **kwargs)
                nonlocal n_binary_ops
                binarry_ops = [n for n in gm.graph.nodes if n.target in aten_binary]
                n_binary_ops += len(binarry_ops)
                return out

            torch._dynamo.reset()
            counters.clear()
            mod_eager = ConvOp(3, 32, self.device, kernel_size=3, stride=2).eval()
            out_optimized = torch.compile(
                mod_eager,
                backend=functools.partial(compile_fx, inner_compile=my_inner_compile),
            )

            inps = [4, 3, 4]
            if module[0] is nn.Conv2d:
                inps.append(inps[-1])
            if module[0] is nn.Conv3d:
                inps.append(inps[-1])
                inps.append(inps[-1])

            inp = torch.rand(inps).to(self.device)
            out_eager = mod_eager(inp)
            out_optimized = out_optimized(inp)
            
            z3_verified_count = counters['inductor'].get('binary_folding_z3_verified', 0)
            
            assert torch.allclose(out_optimized, out_eager, atol=2e-04, rtol=1e-5)
            if expect_success:
                assert n_binary_ops == 0
                print(f"[Conv+BN {module[0].__name__} bias={use_bias}] z3_ok={z3_verified_count}")
            else:
                assert n_binary_ops > 1
            
            return z3_verified_count

        conv_bias = [True, False]
        modules = [
            (nn.Conv1d, nn.BatchNorm1d),
            (nn.Conv2d, nn.BatchNorm2d),
            (nn.Conv3d, nn.BatchNorm3d),
        ]
        
        total_z3_verified = 0
        for use_bias, module in itertools.product(conv_bias, modules):
            z3_verified = test_conv_fusion(use_bias, module, expect_success=True)
            total_z3_verified += z3_verified
        
        print(f"\n=== Conv+BN Test Summary ===")
        print(f"Total Z3 verifications: {total_z3_verified}")
        return total_z3_verified

    @inductor_config.patch({"enable_linear_binary_folding": True, "freezing": True})
    def test_linear_binary_folding_z3(self):
        """Test linear binary folding with Z3 verification"""
        @torch.no_grad()
        def test_linear_fusion(use_bias, op, scalar, add_tensor, expect_success, input_3d=False):
            class LinearOp(nn.Module):
                __constants__ = ["use_scalar"]

                def __init__(self, in_channels, out_channels, device, **kwargs):
                    super().__init__()
                    self.linear = nn.Linear(
                        in_channels, out_channels, bias=use_bias, **kwargs
                    ).to(device)
                    self.use_scalar = scalar
                    tensor_size = [self.linear.weight.size(0),]
                    self.tensor = torch.nn.Parameter(
                        add_tensor if add_tensor is not None else torch.rand(tensor_size).to(device)
                    )
                    self.op = op

                def forward(self, x):
                    x = self.linear(x)
                    if self.use_scalar:
                        return self.op(x, 2.0)
                    else:
                        return self.op(x, self.tensor)

            torch._dynamo.reset()
            counters.clear()
            mod_eager = LinearOp(3, 32, self.device).eval()
            out_optimized = torch.compile(mod_eager)

            torch.manual_seed(1234)
            if input_3d:
                inp = torch.rand([2, 4, 3]).to(self.device)
            else:
                inp = torch.rand([4, 3]).to(self.device)
            out_eager = mod_eager(inp)
            out_optimized = out_optimized(inp)
            
            folding_count = counters['inductor']['binary_folding']
            z3_verified_count = counters['inductor'].get('binary_folding_z3_verified', 0)
            z3_failed_count = counters['inductor'].get('binary_folding_z3_failed', 0)
            
            if expect_success:
                print(f"[Linear bias={use_bias}, op={op.__name__}, scalar={scalar}, shape={add_tensor.shape if add_tensor is not None else 'N/A'}, 3d={input_3d}] fold={folding_count} z3_ok={z3_verified_count} z3_fail={z3_failed_count}")
            
            assert torch.allclose(out_optimized, out_eager, atol=5e-05, rtol=5e-06)
            if expect_success:
                assert counters["inductor"]["binary_folding"] == 1
            else:
                assert counters["inductor"]["binary_folding"] == 0
            
            return z3_verified_count

        linear_bias = [True, False]
        use_scalar = [True, False]
        ops = [torch.add, torch.sub, torch.mul, torch.div]
        add_tensor_size = [[32,], [1, 32], [1,], [1, 1]]
        
        total_z3_verified = 0
        for use_bias, pytorch_op, scalar, tensor_size in itertools.product(
            linear_bias, ops, use_scalar, add_tensor_size
        ):
            z3_verified = test_linear_fusion(
                use_bias,
                pytorch_op,
                scalar,
                add_tensor=torch.rand(tensor_size).to(self.device),
                expect_success=True,
            )
            total_z3_verified += z3_verified

        add_tensor_size.extend([[1, 1, 32], [1, 1, 1]])
        for use_bias, pytorch_op, scalar, tensor_size in itertools.product(
            linear_bias, ops, use_scalar, add_tensor_size
        ):
            z3_verified = test_linear_fusion(
                use_bias,
                pytorch_op,
                scalar,
                add_tensor=torch.rand(tensor_size).to(self.device),
                expect_success=True,
                input_3d=True,
            )
            total_z3_verified += z3_verified

        for use_bias, pytorch_op in itertools.product(linear_bias, ops):
            test_linear_fusion(
                use_bias,
                pytorch_op,
                False,
                add_tensor=torch.rand(4, 32).to(self.device),
                expect_success=False,
            )
            test_linear_fusion(
                use_bias,
                pytorch_op,
                False,
                add_tensor=torch.rand(4, 1).to(self.device),
                expect_success=False,
            )
        
        print(f"\n=== Linear Test Summary ===")
        print(f"Total Z3 verifications: {total_z3_verified}")
        return total_z3_verified


if __name__ == "__main__":
    print("=" * 80)
    print("Testing Binary Folding with Z3 Verification Integration")
    print("=" * 80)
    
    tester = TestBinaryFoldingZ3()
    
    print("\n" + "=" * 80)
    print("TEST 1: Conv Binary Folding + Z3")
    print("=" * 80)
    conv_z3_count = tester.test_conv_binary_folding_z3()
    
    print("\n" + "=" * 80)
    print("TEST 2: Conv+BatchNorm Folding + Z3")
    print("=" * 80)
    conv_bn_z3_count = tester.test_conv_bn_folding_z3()
    
    print("\n" + "=" * 80)
    print("TEST 3: Linear Binary Folding + Z3")
    print("=" * 80)
    linear_z3_count = tester.test_linear_binary_folding_z3()
    
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"Conv Z3 verifications: {conv_z3_count}")
    print(f"Conv+BN Z3 verifications: {conv_bn_z3_count}")
    print(f"Linear Z3 verifications: {linear_z3_count}")
    print(f"Total Z3 verifications: {conv_z3_count + conv_bn_z3_count + linear_z3_count}")
    
    total = conv_z3_count + conv_bn_z3_count + linear_z3_count
    if total > 0:
        print(f"\n? Z3 INTEGRATION IS WORKING! ({total} transformations verified)")
    else:
        print("\n? Z3 integration not triggered (check if binary_folding_z3_advanced.py is in place)")

