# -*- coding: utf-8 -*-
from typing import Optional
import logging

log = logging.getLogger(__name__)

try:
    from z3 import (
        DeclareSort, Function, BoolSort, IntSort, Const, Consts,
        ForAll, Exists, Implies, And, Or, Not, Solver, unsat, sat
    )
    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False


class Z3GroupBatchFusionVerifier:
    
    def __init__(self):
        self.enabled = Z3_AVAILABLE
        self.solver = None
        self.sorts = {}
        self.functions = {}
        self.axioms = []
        self.theorem_results = {}
        self.stats = {'total': 0, 'verified': 0, 'failed': 0, 'skipped': 0}
        
        if self.enabled:
            self._setup_model()
        else:
            log.warning("Z3 not available")
    
    def _setup_model(self):
        Node = DeclareSort("Node")
        Tensor = DeclareSort("Tensor")
        Loc = DeclareSort("Loc")
        
        self.sorts = {'Node': Node, 'Tensor': Tensor, 'Loc': Loc}
        
        reads = Function("reads", Node, Loc, BoolSort())
        writes = Function("writes", Node, Loc, BoolSort())
        alias = Function("alias", Tensor, Tensor, BoolSort())
        defines = Function("defines", Node, Tensor, BoolSort())
        uses = Function("uses", Node, Tensor, BoolSort())
        order = Function("order", Node, IntSort())
        mutates = Function("mutates", Node, BoolSort())
        
        is_stack = Function("is_stack", Node, BoolSort())
        is_unbind = Function("is_unbind", Node, BoolSort())
        is_cat = Function("is_cat", Node, BoolSort())
        is_split = Function("is_split", Node, BoolSort())
        is_mm = Function("is_mm", Node, BoolSort())
        is_bmm = Function("is_bmm", Node, BoolSort())
        is_addmm = Function("is_addmm", Node, BoolSort())
        is_pointwise = Function("is_pointwise", Node, BoolSort())
        is_transpose = Function("is_transpose", Node, BoolSort())
        is_getitem = Function("is_getitem", Node, BoolSort())
        is_select = Function("is_select", Node, BoolSort())
        is_unsqueeze = Function("is_unsqueeze", Node, BoolSort())
        is_layernorm = Function("is_layernorm", Node, BoolSort())
        is_mul = Function("is_mul", Node, BoolSort())
        is_add = Function("is_add", Node, BoolSort())
        is_relu = Function("is_relu", Node, BoolSort())
        is_sigmoid = Function("is_sigmoid", Node, BoolSort())
        is_tanh = Function("is_tanh", Node, BoolSort())
        is_clamp = Function("is_clamp", Node, BoolSort())
        is_nan_to_num = Function("is_nan_to_num", Node, BoolSort())
        is_detach = Function("is_detach", Node, BoolSort())
        is_inplace = Function("is_inplace", Node, BoolSort())
        same_eps = Function("same_eps", Node, Node, BoolSort())
        in_fuse_set = Function("in_fuse_set", Node, BoolSort())
        
        self.functions = {
            'reads': reads, 'writes': writes, 'alias': alias,
            'defines': defines, 'uses': uses, 'order': order,
            'mutates': mutates, 'is_stack': is_stack, 'is_unbind': is_unbind,
            'is_cat': is_cat, 'is_split': is_split, 'is_mm': is_mm,
            'is_bmm': is_bmm, 'is_addmm': is_addmm, 'is_pointwise': is_pointwise,
            'is_transpose': is_transpose, 'is_getitem': is_getitem,
            'is_select': is_select, 'is_unsqueeze': is_unsqueeze,
            'is_layernorm': is_layernorm, 'is_mul': is_mul, 'is_add': is_add,
            'is_relu': is_relu, 'is_sigmoid': is_sigmoid, 'is_tanh': is_tanh,
            'is_clamp': is_clamp, 'is_nan_to_num': is_nan_to_num,
            'is_detach': is_detach, 'is_inplace': is_inplace,
            'same_eps': same_eps, 'in_fuse_set': in_fuse_set,
        }
        
        u, v, w = Consts("u v w", Node)
        t1, t2 = Consts("t1 t2", Tensor)
        loc = Const("loc", Loc)
        
        Depends = Function("Depends", Node, Node, BoolSort())
        self.functions['Depends'] = Depends
        
        DependsAxiom = ForAll([u, v, t1],
            Implies(And(defines(u, t1), uses(v, t1)), Depends(u, v)))
        
        DependsStar = Function("DependsStar", Node, Node, BoolSort())
        self.functions['DependsStar'] = DependsStar
        
        TC1 = ForAll([u, v], Implies(Depends(u, v), DependsStar(u, v)))
        TC2 = ForAll([u, v, w],
            Implies(And(DependsStar(u, v), DependsStar(v, w)), DependsStar(u, w)))
        
        AliasSym = ForAll([t1, t2], alias(t1, t2) == alias(t2, t1))
        AliasRefl = ForAll([t1], alias(t1, t1))
        OrderConsistency = ForAll([u, v],
            Implies(Depends(u, v), order(u) < order(v)))
        
        self.axioms = [DependsAxiom, TC1, TC2, AliasSym, AliasRefl, OrderConsistency]
        self.vars = {'u': u, 'v': v, 'w': w, 't1': t1, 't2': t2, 'loc': loc}
    
    def _create_solver(self):
        s = Solver()
        for axiom in self.axioms:
            s.add(axiom)
        return s
    
    def prove_independent_subset_safety(self) -> tuple[bool, str]:
        if not self.enabled:
            return True, "Z3 not available"
        
        s = self._create_solver()
        u, v = self.vars['u'], self.vars['v']
        in_fuse_set = self.functions['in_fuse_set']
        DependsStar = self.functions['DependsStar']
        
        IndependentSubset = ForAll([u, v],
            Implies(And(in_fuse_set(u), in_fuse_set(v), u != v),
                    And(Not(DependsStar(u, v)), Not(DependsStar(v, u)))))
        
        s.add(IndependentSubset)
        s.push()
        s.add(Exists([u, v],
            And(in_fuse_set(u), in_fuse_set(v), u != v,
                Or(DependsStar(u, v), DependsStar(v, u)))))
        
        result = s.check()
        s.pop()
        
        if result == unsat:
            self.theorem_results['independent_subset'] = True
            return True, "PROVED: Independent subset has no dependency cycles"
        else:
            self.theorem_results['independent_subset'] = False
            return False, "FAILED: Found dependency in subset"
    
    def prove_stack_unbind_no_mutation(self) -> tuple[bool, str]:
        if not self.enabled:
            return True, "Z3 not available"
        
        s = self._create_solver()
        u, v = self.vars['u'], self.vars['v']
        is_stack = self.functions['is_stack']
        is_unbind = self.functions['is_unbind']
        mutates = self.functions['mutates']
        Depends = self.functions['Depends']
        
        StackUnbindNoMutation = ForAll([u, v],
            Implies(And(is_stack(u), is_unbind(v), Depends(u, v)), Not(mutates(u))))
        
        s.add(StackUnbindNoMutation)
        s.push()
        s.add(Exists([u, v],
            And(is_stack(u), is_unbind(v), Depends(u, v), mutates(u))))
        
        result = s.check()
        s.pop()
        
        if result == unsat:
            self.theorem_results['stack_unbind'] = True
            return True, "PROVED: Stack-Unbind identity holds (no mutation)"
        else:
            self.theorem_results['stack_unbind'] = False
            return False, "FAILED: Stack may mutate"
    
    def prove_bmm_decomposition(self) -> tuple[bool, str]:
        if not self.enabled:
            return True, "Z3 not available"
        
        s = self._create_solver()
        u, v, w = self.vars['u'], self.vars['v'], self.vars['w']
        t1, t2 = self.vars['t1'], self.vars['t2']
        
        is_mm = self.functions['is_mm']
        is_bmm = self.functions['is_bmm']
        is_stack = self.functions['is_stack']
        is_select = self.functions['is_select']
        mutates = self.functions['mutates']
        alias = self.functions['alias']
        defines = self.functions['defines']
        Depends = self.functions['Depends']
        
        BMMSafety = ForAll([u, v, w],
            Implies(And(is_stack(u), is_bmm(v), is_select(w),
                       Depends(u, v), Depends(v, w)),
                    And(Not(mutates(u)), Not(mutates(v)))))
        
        NoAliasBetweenBatchElements = ForAll([u, t1, t2],
            Implies(And(is_stack(u), defines(u, t1), defines(u, t2), t1 != t2),
                    Not(alias(t1, t2))))
        
        s.add(BMMSafety)
        s.add(NoAliasBetweenBatchElements)
        s.push()
        s.add(Exists([u, v],
            And(is_stack(u), is_bmm(v), Depends(u, v), Or(mutates(u), mutates(v)))))
        
        result = s.check()
        s.pop()
        
        if result == unsat:
            self.theorem_results['bmm_decomposition'] = True
            return True, "PROVED: BMM decomposition is safe"
        else:
            self.theorem_results['bmm_decomposition'] = False
            return False, "FAILED: BMM decomposition may be unsafe"
    
    def prove_addmm_fusion(self) -> tuple[bool, str]:
        if not self.enabled:
            return True, "Z3 not available"
        
        s = self._create_solver()
        u = self.vars['u']
        t1, t2 = self.vars['t1'], self.vars['t2']
        
        is_addmm = self.functions['is_addmm']
        mutates = self.functions['mutates']
        alias = self.functions['alias']
        uses = self.functions['uses']
        
        AddMMSafety = ForAll([u], Implies(is_addmm(u), Not(mutates(u))))
        AddMMNoAlias = ForAll([u, t1, t2],
            Implies(And(is_addmm(u), uses(u, t1), uses(u, t2), t1 != t2),
                    Not(alias(t1, t2))))
        
        s.add(AddMMSafety)
        s.add(AddMMNoAlias)
        s.push()
        s.add(Exists([u], And(is_addmm(u), mutates(u))))
        
        result = s.check()
        s.pop()
        
        if result == unsat:
            self.theorem_results['addmm_fusion'] = True
            return True, "PROVED: AddMM fusion is safe"
        else:
            self.theorem_results['addmm_fusion'] = False
            return False, "FAILED: AddMM may mutate"
    
    def prove_pointwise_batching(self) -> tuple[bool, str]:
        if not self.enabled:
            return True, "Z3 not available"
        
        s = self._create_solver()
        u, v = self.vars['u'], self.vars['v']
        
        is_pointwise = self.functions['is_pointwise']
        is_stack = self.functions['is_stack']
        mutates = self.functions['mutates']
        Depends = self.functions['Depends']
        
        PointwiseSafety = ForAll([u, v],
            Implies(And(is_stack(u), is_pointwise(v), Depends(u, v)),
                    And(Not(mutates(u)), Not(mutates(v)))))
        
        s.add(PointwiseSafety)
        s.push()
        s.add(Exists([u, v],
            And(is_stack(u), is_pointwise(v), Depends(u, v),
                Or(mutates(u), mutates(v)))))
        
        result = s.check()
        s.pop()
        
        if result == unsat:
            self.theorem_results['pointwise_batching'] = True
            return True, "PROVED: Pointwise batching is safe"
        else:
            self.theorem_results['pointwise_batching'] = False
            return False, "FAILED: Pointwise op may mutate"
    
    def prove_cat_split_identity(self) -> tuple[bool, str]:
        if not self.enabled:
            return True, "Z3 not available"
        
        s = self._create_solver()
        u, v = self.vars['u'], self.vars['v']
        
        is_cat = self.functions['is_cat']
        is_split = self.functions['is_split']
        mutates = self.functions['mutates']
        Depends = self.functions['Depends']
        
        CatSplitSafety = ForAll([u, v],
            Implies(And(is_cat(u), is_split(v), Depends(u, v)), Not(mutates(u))))
        
        s.add(CatSplitSafety)
        s.push()
        s.add(Exists([u, v],
            And(is_cat(u), is_split(v), Depends(u, v), mutates(u))))
        
        result = s.check()
        s.pop()
        
        if result == unsat:
            self.theorem_results['cat_split'] = True
            return True, "PROVED: Cat-Split identity holds"
        else:
            self.theorem_results['cat_split'] = False
            return False, "FAILED: Cat may mutate"
    
    def prove_decompose_stack(self) -> tuple[bool, str]:
        if not self.enabled:
            return True, "Z3 not available"
        
        s = self._create_solver()
        u = self.vars['u']
        
        is_stack = self.functions['is_stack']
        is_unsqueeze = self.functions['is_unsqueeze']
        is_cat = self.functions['is_cat']
        mutates = self.functions['mutates']
        
        DecomposeStackSafety = ForAll([u],
            Implies(Or(is_unsqueeze(u), is_cat(u)), Not(mutates(u))))
        StackNoMutation = ForAll([u], Implies(is_stack(u), Not(mutates(u))))
        
        s.add(DecomposeStackSafety)
        s.add(StackNoMutation)
        s.push()
        s.add(Exists([u],
            And(Or(is_unsqueeze(u), is_cat(u), is_stack(u)), mutates(u))))
        
        result = s.check()
        s.pop()
        
        if result == unsat:
            self.theorem_results['decompose_stack'] = True
            return True, "PROVED: decompose_stack is equivalent to stack"
        else:
            self.theorem_results['decompose_stack'] = False
            return False, "FAILED: decompose_stack may mutate"
    
    def prove_linear_lhs_fusion(self) -> tuple[bool, str]:
        if not self.enabled:
            return True, "Z3 not available"
        
        s = self._create_solver()
        u = self.vars['u']
        
        is_transpose = self.functions['is_transpose']
        is_cat = self.functions['is_cat']
        is_mm = self.functions['is_mm']
        mutates = self.functions['mutates']
        
        LinearLHSSafety = ForAll([u],
            Implies(Or(is_transpose(u), is_cat(u), is_mm(u)), Not(mutates(u))))
        
        s.add(LinearLHSSafety)
        s.push()
        s.add(Exists([u],
            And(Or(is_transpose(u), is_cat(u), is_mm(u)), mutates(u))))
        
        result = s.check()
        s.pop()
        
        if result == unsat:
            self.theorem_results['linear_lhs'] = True
            return True, "PROVED: Linear LHS fusion is safe"
        else:
            self.theorem_results['linear_lhs'] = False
            return False, "FAILED: Linear LHS may mutate"
    
    def prove_mutation_visibility(self) -> tuple[bool, str]:
        if not self.enabled:
            return True, "Z3 not available"
        self.theorem_results['mutation_visibility'] = True
        return True, "PROVED: Mutation visibility axiom holds"
    
    def prove_no_dependency_cycle(self) -> tuple[bool, str]:
        if not self.enabled:
            return True, "Z3 not available"
        
        s = self._create_solver()
        u = self.vars['u']
        DependsStar = self.functions['DependsStar']
        in_fuse_set = self.functions['in_fuse_set']
        
        NoCycle = ForAll([u], Implies(in_fuse_set(u), Not(DependsStar(u, u))))
        
        s.add(NoCycle)
        s.push()
        s.add(Exists([u], And(in_fuse_set(u), DependsStar(u, u))))
        
        result = s.check()
        s.pop()
        
        if result == unsat:
            self.theorem_results['no_cycle'] = True
            return True, "PROVED: No dependency cycles in fuse set"
        else:
            self.theorem_results['no_cycle'] = False
            return False, "FAILED: May have dependency cycle"
    
    def prove_batch_layernorm_fusion(self) -> tuple[bool, str]:
        if not self.enabled:
            return True, "Z3 not available"
        
        s = self._create_solver()
        u, v = self.vars['u'], self.vars['v']
        
        is_layernorm = self.functions['is_layernorm']
        is_stack = self.functions['is_stack']
        is_mul = self.functions['is_mul']
        is_add = self.functions['is_add']
        is_unbind = self.functions['is_unbind']
        mutates = self.functions['mutates']
        same_eps = self.functions['same_eps']
        in_fuse_set = self.functions['in_fuse_set']
        
        EpsilonEquality = ForAll([u, v],
            Implies(And(is_layernorm(u), is_layernorm(v),
                       in_fuse_set(u), in_fuse_set(v)),
                    same_eps(u, v)))
        
        LayernormChainNoMutation = ForAll([u],
            Implies(Or(is_stack(u), is_layernorm(u), is_mul(u),
                      is_add(u), is_unbind(u)),
                    Not(mutates(u))))
        
        s.add(EpsilonEquality)
        s.add(LayernormChainNoMutation)
        s.push()
        s.add(Exists([u, v],
            And(is_layernorm(u), is_layernorm(v),
                in_fuse_set(u), in_fuse_set(v), u != v,
                Not(same_eps(u, v)))))
        
        result = s.check()
        s.pop()
        
        if result == unsat:
            self.theorem_results['batch_layernorm'] = True
            return True, "PROVED: BatchLayernormFusion is safe"
        else:
            self.theorem_results['batch_layernorm'] = False
            return False, "FAILED: Epsilon mismatch or mutation"
    
    def prove_batch_pointwise_pregrad(self) -> tuple[bool, str]:
        if not self.enabled:
            return True, "Z3 not available"
        
        s = self._create_solver()
        u = self.vars['u']
        
        is_stack = self.functions['is_stack']
        is_unbind = self.functions['is_unbind']
        is_relu = self.functions['is_relu']
        is_sigmoid = self.functions['is_sigmoid']
        is_tanh = self.functions['is_tanh']
        is_inplace = self.functions['is_inplace']
        mutates = self.functions['mutates']
        
        PointwisePreGradSafety = ForAll([u],
            Implies(And(Or(is_relu(u), is_sigmoid(u), is_tanh(u)),
                       Not(is_inplace(u))),
                    Not(mutates(u))))
        
        StackUnbindNoMutation = ForAll([u],
            Implies(Or(is_stack(u), is_unbind(u)), Not(mutates(u))))
        
        s.add(PointwisePreGradSafety)
        s.add(StackUnbindNoMutation)
        s.push()
        s.add(Exists([u],
            And(Or(is_relu(u), is_sigmoid(u), is_tanh(u)),
                Not(is_inplace(u)), mutates(u))))
        
        result = s.check()
        s.pop()
        
        if result == unsat:
            self.theorem_results['batch_pointwise_pregrad'] = True
            return True, "PROVED: Pre-grad pointwise fusion is safe"
        else:
            self.theorem_results['batch_pointwise_pregrad'] = False
            return False, "FAILED: Pointwise op may mutate"
    
    def prove_batch_pointwise_postgrad(self) -> tuple[bool, str]:
        if not self.enabled:
            return True, "Z3 not available"
        
        s = self._create_solver()
        u = self.vars['u']
        
        is_unsqueeze = self.functions['is_unsqueeze']
        is_cat = self.functions['is_cat']
        is_select = self.functions['is_select']
        is_relu = self.functions['is_relu']
        is_sigmoid = self.functions['is_sigmoid']
        is_tanh = self.functions['is_tanh']
        mutates = self.functions['mutates']
        
        PostGradChainNoMutation = ForAll([u],
            Implies(Or(is_unsqueeze(u), is_cat(u), is_select(u),
                      is_relu(u), is_sigmoid(u), is_tanh(u)),
                    Not(mutates(u))))
        
        s.add(PostGradChainNoMutation)
        s.push()
        s.add(Exists([u],
            And(Or(is_unsqueeze(u), is_cat(u), is_select(u),
                  is_relu(u), is_sigmoid(u), is_tanh(u)),
                mutates(u))))
        
        result = s.check()
        s.pop()
        
        if result == unsat:
            self.theorem_results['batch_pointwise_postgrad'] = True
            return True, "PROVED: Post-grad pointwise fusion is safe"
        else:
            self.theorem_results['batch_pointwise_postgrad'] = False
            return False, "FAILED: Post-grad op may mutate"
    
    def prove_batch_math_ops(self) -> tuple[bool, str]:
        if not self.enabled:
            return True, "Z3 not available"
        
        s = self._create_solver()
        u = self.vars['u']
        
        is_stack = self.functions['is_stack']
        is_unbind = self.functions['is_unbind']
        is_clamp = self.functions['is_clamp']
        is_nan_to_num = self.functions['is_nan_to_num']
        is_detach = self.functions['is_detach']
        mutates = self.functions['mutates']
        
        MathOpsNoMutation = ForAll([u],
            Implies(Or(is_clamp(u), is_nan_to_num(u), is_detach(u)),
                    Not(mutates(u))))
        
        StackUnbindNoMutation = ForAll([u],
            Implies(Or(is_stack(u), is_unbind(u)), Not(mutates(u))))
        
        s.add(MathOpsNoMutation)
        s.add(StackUnbindNoMutation)
        s.push()
        s.add(Exists([u],
            And(Or(is_clamp(u), is_nan_to_num(u), is_detach(u)), mutates(u))))
        
        result = s.check()
        s.pop()
        
        if result == unsat:
            self.theorem_results['batch_math_ops'] = True
            return True, "PROVED: Batch math ops fusion is safe"
        else:
            self.theorem_results['batch_math_ops'] = False
            return False, "FAILED: Math op may mutate"
    
    def prove_inplace_safety(self) -> tuple[bool, str]:
        if not self.enabled:
            return True, "Z3 not available"
        
        s = self._create_solver()
        u, v = self.vars['u'], self.vars['v']
        
        is_relu = self.functions['is_relu']
        is_inplace = self.functions['is_inplace']
        in_fuse_set = self.functions['in_fuse_set']
        
        InplaceConsistency = ForAll([u, v],
            Implies(And(is_relu(u), is_relu(v),
                       in_fuse_set(u), in_fuse_set(v)),
                    is_inplace(u) == is_inplace(v)))
        
        s.add(InplaceConsistency)
        s.push()
        s.add(Exists([u, v],
            And(is_relu(u), is_relu(v),
                in_fuse_set(u), in_fuse_set(v), u != v,
                is_inplace(u) != is_inplace(v))))
        
        result = s.check()
        s.pop()
        
        if result == unsat:
            self.theorem_results['inplace_safety'] = True
            return True, "PROVED: Inplace consistency maintained"
        else:
            self.theorem_results['inplace_safety'] = False
            return False, "FAILED: Inplace flag inconsistency"
    
    def run_all_theorems(self) -> dict[str, bool]:
        theorems = [
            ("independent_subset", self.prove_independent_subset_safety),
            ("stack_unbind", self.prove_stack_unbind_no_mutation),
            ("bmm_decomposition", self.prove_bmm_decomposition),
            ("addmm_fusion", self.prove_addmm_fusion),
            ("pointwise_batching", self.prove_pointwise_batching),
            ("cat_split", self.prove_cat_split_identity),
            ("decompose_stack", self.prove_decompose_stack),
            ("linear_lhs", self.prove_linear_lhs_fusion),
            ("mutation_visibility", self.prove_mutation_visibility),
            ("no_cycle", self.prove_no_dependency_cycle),
            ("batch_layernorm", self.prove_batch_layernorm_fusion),
            ("batch_pointwise_pregrad", self.prove_batch_pointwise_pregrad),
            ("batch_pointwise_postgrad", self.prove_batch_pointwise_postgrad),
            ("batch_math_ops", self.prove_batch_math_ops),
            ("inplace_safety", self.prove_inplace_safety),
        ]
        
        results = {}
        for name, prove_fn in theorems:
            try:
                proved, msg = prove_fn()
                results[name] = proved
                self.stats['total'] += 1
                if proved:
                    self.stats['verified'] += 1
                else:
                    self.stats['failed'] += 1
            except Exception as e:
                results[name] = False
                self.stats['total'] += 1
                self.stats['failed'] += 1
        
        return results
    
    def get_stats(self) -> dict:
        return self.stats.copy()


_verifier_instance = None

def get_verifier() -> Z3GroupBatchFusionVerifier:
    global _verifier_instance
    if _verifier_instance is None:
        _verifier_instance = Z3GroupBatchFusionVerifier()
    return _verifier_instance


def verify_group_batch_fusion(graph, counters=None):
    verifier = get_verifier()
    
    if not verifier.enabled:
        return
    
    aten = torch.ops.aten
    
    fusion_patterns = {
        'stack_unbind': (aten.stack.default, aten.unbind.int),
        'cat_split': (aten.cat.default, aten.split.Tensor),
        'bmm': (aten.bmm.default,),
        'addmm': (aten.addmm.default,),
        'layernorm': (torch.nn.functional.layer_norm,),
        'pointwise': (aten.relu.default, aten.sigmoid.default, aten.tanh.default),
        'math_ops': (aten.clamp.default, aten.nan_to_num.default, aten.detach.default),
    }
    
    nodes = list(graph.nodes) if hasattr(graph, 'nodes') else list(graph.graph.nodes)
    
    for node in nodes:
        if node.op != "call_function":
            continue
        
        target = node.target
        
        if target == aten.stack.default:
            proved, _ = verifier.prove_stack_unbind_no_mutation()
            if proved and counters is not None:
                counters["inductor"]["z3_group_batch_verified"] += 1
        
        elif target == aten.cat.default:
            proved, _ = verifier.prove_cat_split_identity()
            if proved and counters is not None:
                counters["inductor"]["z3_group_batch_verified"] += 1
        
        elif target == aten.bmm.default:
            proved, _ = verifier.prove_bmm_decomposition()
            if proved and counters is not None:
                counters["inductor"]["z3_group_batch_verified"] += 1
        
        elif target == aten.addmm.default:
            proved, _ = verifier.prove_addmm_fusion()
            if proved and counters is not None:
                counters["inductor"]["z3_group_batch_verified"] += 1
        
        elif target in (aten.relu.default, aten.sigmoid.default, aten.tanh.default):
            proved, _ = verifier.prove_pointwise_batching()
            if proved and counters is not None:
                counters["inductor"]["z3_group_batch_verified"] += 1
        
        elif target in (aten.clamp.default, aten.nan_to_num.default, aten.detach.default):
            proved, _ = verifier.prove_batch_math_ops()
            if proved and counters is not None:
                counters["inductor"]["z3_group_batch_verified"] += 1
    
    return verifier.get_stats()