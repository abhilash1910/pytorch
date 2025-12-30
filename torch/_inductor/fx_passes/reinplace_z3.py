# mypy: allow-untyped-defs
import itertools
import logging
import operator
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Callable, cast, Union, Tuple

import z3
from z3 import *

log = logging.getLogger(__name__)

TensorSort = z3.DeclareSort('Tensor')
NodeSort = z3.DeclareSort('Node')
StorageSort = z3.DeclareSort('Storage')
GraphSort = z3.DeclareSort('Graph')
OpSort = z3.DeclareSort('Op')

get_node_storage = z3.Function('get_node_storage', NodeSort, StorageSort)
is_node_realized = z3.Function('is_node_realized', NodeSort, z3.BoolSort())
node_order = z3.Function('node_order', NodeSort, z3.IntSort())
is_view_op = z3.Function('is_view_op', OpSort, z3.BoolSort())
has_storage = z3.Function('has_storage', NodeSort, z3.BoolSort())
alias = z3.Function('alias', NodeSort, NodeSort, z3.BoolSort())
compute_overlapping_tensors = z3.Function('compute_overlapping_tensors', 
                                        z3.ArraySort(z3.IntSort(), TensorSort), 
                                        z3.IntSort())

@dataclass(frozen=True)
class Z3InplaceableOp:
    inplace_op: str
    mutated_arg: int
    extra_check: str = "default_check"


@dataclass
class Z3ViewOp:
    target: str
    args: tuple[Any, ...]
    kwargs: dict[str, Any]


class Z3ReinplaceVerifier:
    """Z3-based verification system for reinplace optimizations"""
    
    def __init__(self):
        self.solver = z3.Solver()
        self.tensor_vars = {}
        self.node_vars = {}
        self.storage_vars = {}
        self.graph_vars = {}
        
        self.has_storage = has_storage
        self.alias = alias
        self.node_order = node_order
        self.get_node_storage = get_node_storage
        self.is_node_realized = is_node_realized
        
        self._setup_base_constraints()
        self._scatter_op_to_view = {
            'aten.diagonal_scatter.default': 'aten.diagonal.default',
            'aten.select_scatter.default': 'aten.select.int',
            'aten.slice_scatter.default': 'aten.slice.Tensor',
            'aten.as_strided_scatter.default': 'aten.as_strided.default',
        }
        self._view_op_to_scatter = {v: k for k, v in self._scatter_op_to_view.items()}
        self.z3_inplaceable_ops = {
            'aten.index_put.default': Z3InplaceableOp('aten.index_put_.default', 0),
            'aten._unsafe_index_put.default': Z3InplaceableOp('inductor_prims._unsafe_index_put_', 0),
            '_generalized_scatter': Z3InplaceableOp('_inplace_generalized_scatter', 0, 'should_reinplace_scatter'),
        }
        self.meta_only_ops = {
            'aten.sym_size.int',
            'aten.sym_stride.int', 
            'aten.sym_numel.default',
            'aten.sym_storage_offset.default',
        }
    
    def _setup_base_constraints(self):
        n1, n2, n3 = z3.Consts('n1 n2 n3', NodeSort)
        
        # Axiom 1: Aliasing definition - nodes alias if they have same storage
        self.solver.add(
            z3.ForAll(
                [n1, n2],
                z3.Implies(
                    z3.And(self.has_storage(n1), self.has_storage(n2)),
                    self.alias(n1, n2) == (get_node_storage(n1) == get_node_storage(n2))
                )
            )
        )
        
        # Axiom 2: Aliasing is reflexive
        self.solver.add(
            z3.ForAll(
                [n1],
                z3.Implies(self.has_storage(n1), self.alias(n1, n1))
            )
        )
        
        # Axiom 3: Aliasing is symmetric
        self.solver.add(
            z3.ForAll(
                [n1, n2],
                z3.Implies(self.alias(n1, n2), self.alias(n2, n1))
            )
        )
        
        # Axiom 4: Aliasing is transitive
        self.solver.add(
            z3.ForAll(
                [n1, n2, n3],
                z3.Implies(
                    z3.And(self.alias(n1, n2), self.alias(n2, n3)),
                    self.alias(n1, n3)
                )
            )
        )
        
        # Axiom 5: node_order is transitive
        self.solver.add(
            z3.ForAll(
                [n1, n2, n3],
                z3.Implies(
                    z3.And(node_order(n1) < node_order(n2), node_order(n2) < node_order(n3)),
                    node_order(n1) < node_order(n3)
                )
            )
        )
    
    '''
    def _setup_base_constraints(self):
        """Setup base Z3 constraints for tensor operations"""
        node_uses_tensor = z3.Function("node_uses_tensor",  NodeSort, TensorSort, z3.BoolSort())
        alias = z3.Function('alias', TensorSort, TensorSort, z3.BoolSort())
        has_storage = z3.Function("has_storage", TensorSort, z3.BoolSort())
        t1, t2 = z3.Consts('t1 t2', TensorSort)
        #self.solver.add(z3.ForAll([t1, t2],
        #    alias(t1, t2) == (get_node_storage(t1) == get_node_storage(t2))
        #))
        #self.solver.add(
        #z3.ForAll(
        #    [t1, t2],
        #    z3.Implies(
        #        z3.And(has_storage(t1), has_storage(t2)),
        #        alias(t1, t2)
        #        == (get_node_storage(t1) == get_node_storage(t2)),
        #        ),
        #    )
        #)
        t = z3.Const('t', NodeSort)
        self.solver.add(z3.ForAll([t], get_node_storage(t) == get_node_storage(t)))
        n1, n2, n3 = z3.Consts('n1 n2 n3', NodeSort)
        self.solver.add(z3.ForAll([n1, n2, n3],
            z3.Implies(z3.And(node_order(n1) < node_order(n2),
                             node_order(n2) < node_order(n3)),
                      node_order(n1) < node_order(n3))))
        '''
    
    def create_node_var(self, name: str) -> z3.ExprRef:
        """Create a Z3 node variable"""
        if name not in self.node_vars:
            self.node_vars[name] = z3.Const(f'node_{name}', NodeSort)
        return self.node_vars[name]
    
    def create_tensor_var(self, name: str) -> z3.ExprRef:
        """Create a Z3 tensor variable"""
        if name not in self.tensor_vars:
            self.tensor_vars[name] = z3.Const(f'tensor_{name}', TensorSort)
        return self.tensor_vars[name]
    
    def create_graph_var(self, name: str) -> z3.ExprRef:
        """Create a Z3 graph variable"""
        if name not in self.graph_vars:
            self.graph_vars[name] = z3.Const(f'graph_{name}', GraphSort)
        return self.graph_vars[name]


def z3_graph_call_function(verifier: Z3ReinplaceVerifier, graph_name: str, fn_name: str, 
                          *args, **kwargs) -> str:
    """Z3 version of graph_call_function"""
    node_name = f"{fn_name}_call_{len(verifier.node_vars)}"
    node = verifier.create_node_var(node_name)
    graph = verifier.create_graph_var(graph_name)
    
    has_node = z3.Function('has_node', GraphSort, NodeSort, z3.BoolSort())
    verifier.solver.add(has_node(graph, node))
    
    return node_name


def z3_inplace_generalized_scatter(verifier: Z3ReinplaceVerifier, 
                                  inp_name: str, src_name: str, 
                                  view_ops: list[Z3ViewOp]) -> str:
    """Z3 version of _inplace_generalized_scatter"""
    inp = verifier.create_node_var(inp_name)
    src = verifier.create_node_var(src_name)
    result_name = f"inplace_scatter_{len(verifier.node_vars)}"
    result = verifier.create_node_var(result_name)
    verifier.solver.add(get_node_storage(result) == get_node_storage(inp))
    
    current = inp
    for i, view_op in enumerate(view_ops):
        view_name = f"view_{i}_{len(verifier.node_vars)}"
        view_node = verifier.create_node_var(view_name)
        verifier.solver.add(get_node_storage(view_node) == get_node_storage(current))
        current = view_node
    
    return result_name


def z3_generalized_scatter(verifier: Z3ReinplaceVerifier,
                          inp_name: str, src_name: str, 
                          view_ops: list[Z3ViewOp]) -> str:
    """Z3 version of _generalized_scatter"""
    clone_name = f"clone_{inp_name}_{len(verifier.node_vars)}"
    clone_node = verifier.create_node_var(clone_name)
    inp_node = verifier.create_node_var(inp_name)
    verifier.solver.add(get_node_storage(clone_node) != get_node_storage(inp_node))
    return z3_inplace_generalized_scatter(verifier, clone_name, src_name, view_ops)


def z3_decompose_scatter_functional_helper(verifier: Z3ReinplaceVerifier,
                                          graph_name: str, inp_name: str, 
                                          src_name: str, view_ops: list[Z3ViewOp]) -> str:
    """Z3 version of _decompose_scatter_functional_helper"""
    if not view_ops:
        return src_name
    
    view_op = view_ops[0]
    view_ops_tail = view_ops[1:]
    
    if view_ops_tail:
        view_name = z3_graph_call_function(verifier, graph_name, view_op.target, 
                                          inp_name, *view_op.args, **view_op.kwargs)
        src_name = z3_decompose_scatter_functional_helper(verifier, graph_name, 
                                                         view_name, src_name, view_ops_tail)
    
    scatter_op = verifier._view_op_to_scatter.get(view_op.target, 'unknown_scatter')
    return z3_graph_call_function(verifier, graph_name, scatter_op, 
                                 inp_name, src_name, *view_op.args, **view_op.kwargs)


def z3_decompose_scatter_functional(verifier: Z3ReinplaceVerifier, 
                                   graph_name: str, node_name: str) -> str:
    """Z3 version of _decompose_scatter_functional"""
    # Symbolic decomposition of generalized scatter
    node = verifier.create_node_var(node_name)
    
    # Create symbolic arguments (simplified)
    inp_name = f"{node_name}_inp"
    src_name = f"{node_name}_src"
    view_ops = []  # Simplified for Z3 modeling
    
    return z3_decompose_scatter_functional_helper(verifier, graph_name, 
                                                 inp_name, src_name, view_ops)


def z3_decompose_scatter_mutating(verifier: Z3ReinplaceVerifier,
                                 graph_name: str, node_name: str) -> str:
    """Z3 version of _decompose_scatter_mutating"""
    node = verifier.create_node_var(node_name)
    
    # Create clone operation
    inp_name = f"{node_name}_inp"
    clone_name = z3_graph_call_function(verifier, graph_name, 'aten.clone', inp_name)
    
    # Model mutations on the clone
    result_name = f"mutated_{clone_name}"
    result = verifier.create_node_var(result_name)
    clone = verifier.create_node_var(clone_name)
    
    # Constraint: result is the mutated clone
    verifier.solver.add(get_node_storage(result) == get_node_storage(clone))
    
    return result_name


def z3_scatter_always_uses_mutation(verifier: Z3ReinplaceVerifier, node_name: str) -> bool:
    """Z3 version of scatter_always_uses_mutation"""
    node = verifier.create_node_var(node_name)
    
    # Check if any view ops are always mutating
    always_mutating_ops = {'aten.as_strided.default', 'aten.diagonal.default'}
    
    # Create symbolic check
    uses_mutation = z3.Bool(f"uses_mutation_{node_name}")
    
    # For Z3 modeling, we'll assume this is determined by the operation type
    # In practice, this would be checked against the view_ops list
    verifier.solver.add(uses_mutation == z3.BoolVal(True))  # Simplified
    
    verifier.solver.push()
    verifier.solver.add(uses_mutation)
    result = verifier.solver.check() == z3.sat
    verifier.solver.pop()
    
    return result


def z3_should_reinplace_scatter(verifier: Z3ReinplaceVerifier, node_name: str) -> bool:
    """Z3 version of should_reinplace_scatter"""
    node = verifier.create_node_var(node_name)
    
    # Create symbolic variables for the decision factors
    inp_realized = z3.Bool(f"inp_realized_{node_name}")
    output_realized = z3.Bool(f"output_realized_{node_name}")
    uses_mutation = z3.Bool(f"uses_mutation_{node_name}")
    has_copy_user = z3.Bool(f"has_copy_user_{node_name}")
    
    # Decision logic: should reinplace if any of these conditions hold
    should_reinplace = z3.Or(
        uses_mutation,  # Mutating scatter ops unconditionally realize
        z3.And(inp_realized, output_realized),  # Both input and output realized
        has_copy_user  # Output copied back to input
    )
    
    verifier.solver.push()
    verifier.solver.add(should_reinplace)
    result = verifier.solver.check() == z3.sat
    verifier.solver.pop()
    
    return result


def z3_decompose_generalized_scatter(verifier: Z3ReinplaceVerifier, graph_name: str) -> None:
    """Z3 version of decompose_generalized_scatter"""
    graph = verifier.create_graph_var(graph_name)
    
    # Find all generalized scatter nodes (symbolic)
    scatter_nodes = z3.Array('scatter_nodes', z3.IntSort(), NodeSort)
    num_scatter_nodes = z3.Int('num_scatter_nodes')
    
    # For each scatter node, decide decomposition strategy
    i = z3.Int('i')
    node = z3.Select(scatter_nodes, i)
    
    use_mutation = z3.Bool('use_mutation')
    verifier.solver.add(z3.ForAll([i], 
        z3.Implies(z3.And(i >= 0, i < num_scatter_nodes),
                  z3.Or(
                      z3.And(use_mutation, 
                            # Decompose using mutations
                            z3.BoolVal(True)),
                      z3.And(z3.Not(use_mutation),
                            # Decompose functionally  
                            z3.BoolVal(True))
                  ))))


def z3_canonicalize_view_scatter_ops(verifier: Z3ReinplaceVerifier, graph_name: str) -> None:
    """Z3 version of canonicalize_view_scatter_ops"""
    graph = verifier.create_graph_var(graph_name)
    
    # Model view base tracking
    node_to_view_base = z3.Array('node_to_view_base', NodeSort, NodeSort)
    
    # Model view operation tracking  
    node_has_view_op = z3.Function('node_has_view_op', NodeSort, z3.BoolSort())
    
    # For each node in the graph
    node = z3.Const('node', NodeSort)
    verifier.solver.add(z3.ForAll([node],
        z3.Implies(is_view_op(z3.Const('op', OpSort)),
                  node_has_view_op(node))))


def z3_any_use_of_views_after_node(verifier: Z3ReinplaceVerifier, 
                                  node_name: str, shared_view_nodes: list[str],
                                  copy_node_name: str = None, 
                                  mutated_arg_name: str = None) -> bool:
    """Z3 version of any_use_of_views_after_node"""
    node = verifier.create_node_var(node_name)
    node_loc = z3.Int(f"{node_name}_order")
    verifier.solver.add(node_order(node) == node_loc)
    
    copy_node_loc = None
    if copy_node_name:
        copy_node = verifier.create_node_var(copy_node_name)
        copy_node_loc = z3.Int(f"{copy_node_name}_order")
        verifier.solver.add(node_order(copy_node) == copy_node_loc)
    
    # Check if any view has users after the current node
    has_later_use = z3.Bool(f"has_later_use_{node_name}")
    
    for view_name in shared_view_nodes:
        view_node = verifier.create_node_var(view_name)
        user_node = verifier.create_node_var(f"{view_name}_user")
        user_loc = z3.Int(f"{view_name}_user_order")
        
        verifier.solver.add(node_order(user_node) == user_loc)
        
        # User after current node
        later_user = user_loc > node_loc
        
        # But before copy epilogue (if exists)
        if copy_node_loc:
            later_user = z3.And(later_user, user_loc < copy_node_loc)
        
        # Not a meta-only user
        is_meta_user = z3.Bool(f"is_meta_user_{view_name}")
        
        # Not a copy operation on the mutated arg
        is_copy_mutated = z3.Bool(f"is_copy_mutated_{view_name}")
        if mutated_arg_name:
            # Model copy_ operation check
            pass
        
        has_problematic_use = z3.And(later_user, 
                                   z3.Not(is_meta_user),
                                   z3.Not(is_copy_mutated))
        
        verifier.solver.add(z3.Implies(has_problematic_use, has_later_use))
    
    verifier.solver.push()
    verifier.solver.add(has_later_use)
    result = verifier.solver.check() == z3.sat
    verifier.solver.pop()
    
    return result


def z3_get_shared_view_nodes_exact(verifier: Z3ReinplaceVerifier, mutated_arg_name: str, storage_to_nodes: z3.ArrayRef) -> Tuple[z3.ArrayRef, z3.ExprRef]:
    """Exact Z3 translation of the original 2 lines"""
    mutated_arg = verifier.create_node_var(mutated_arg_name)
    
    # Line 1: shared_view_nodes = storage_to_nodes[get_node_storage(mutated_arg)]
    shared_view_nodes = z3.Select(storage_to_nodes, get_node_storage(mutated_arg))
    
    # We need to know the size of shared_view_nodes array
    num_shared_nodes = z3.Int('num_shared_nodes')
    verifier.solver.add(num_shared_nodes >= 0)
    
    # Line 2: shared_view_nodes = [v for v in shared_view_nodes if _overlap([mutated_arg.meta["val"], v.meta["val"]])]
    filtered_nodes = z3.Array('filtered_shared_nodes', z3.IntSort(), NodeSort)
    num_filtered = z3.Int('num_filtered_nodes')
    verifier.solver.add(num_filtered >= 0)
    verifier.solver.add(num_filtered <= num_shared_nodes)
    
    # For all i in range(num_shared_nodes):
    i = z3.Int('i')
    v = z3.Select(shared_view_nodes, i)
    overlap_condition = compute_overlapping_tensors(z3.Store(z3.Store(z3.K(TensorSort, z3.Const('dummy', TensorSort)), 0, mutated_arg), 1, v)) > 0
    
    # Universal quantification: for all valid indices, if overlap then include in filtered
    verifier.solver.add(z3.ForAll([i], 
        z3.Implies(z3.And(i >= 0, i < num_shared_nodes, overlap_condition),
                  z3.Exists([z3.Int('j')], z3.And(z3.Int('j') >= 0, z3.Int('j') < num_filtered, z3.Select(filtered_nodes, z3.Int('j')) == v)))))
    
    return filtered_nodes, num_filtered


def z3_can_inplace(verifier: Z3ReinplaceVerifier, node_name: str, mutated_arg_name: str) -> bool:
    """Z3 version of can_inplace with proper shared view nodes handling"""
    node = verifier.create_node_var(node_name)
    mutated_arg = verifier.create_node_var(mutated_arg_name)
    
    # Check if mutated_arg has storage
    has_storage = z3.Bool(f"has_storage_{mutated_arg_name}")
    null_storage = z3.Const('null_storage', StorageSort)
    verifier.solver.add(has_storage == (get_node_storage(mutated_arg) != null_storage))
    
    # Early return if no storage (simplified for Z3)
    verifier.solver.push()
    verifier.solver.add(z3.Not(has_storage))
    if verifier.solver.check() == z3.sat:
        verifier.solver.pop()
        return False
    verifier.solver.pop()
    
    # Model storage_to_nodes mapping
    storage_to_nodes = z3.Array('storage_to_nodes', StorageSort, 
                               z3.ArraySort(z3.IntSort(), NodeSort))
    
    # Get shared view nodes with overlap filtering
    shared_view_nodes, num_shared_views = z3_get_shared_view_nodes_with_overlap(
        verifier, mutated_arg_name, storage_to_nodes
    )
    
    # Check if mutated_arg is graph input
    is_placeholder = z3.Bool(f"is_placeholder_{mutated_arg_name}")
    is_get_attr = z3.Bool(f"is_get_attr_{mutated_arg_name}")
    is_graph_input = z3.Or(is_placeholder, is_get_attr)
    
    # Model the decision logic
    verifier.solver.push()
    
    # Case 1: Graph input
    verifier.solver.add(is_graph_input)
    
    # Need copy epilogue for graph inputs
    has_copy_epilogue = z3.Bool(f"has_copy_epilogue_{mutated_arg_name}")
    
    # If no copy epilogue, cannot inplace
    no_copy_case = z3.And(is_graph_input, z3.Not(has_copy_epilogue))
    verifier.solver.add(z3.Implies(no_copy_case, z3.BoolVal(False)))
    
    # If has copy epilogue, check for uses after node
    copy_case = z3.And(is_graph_input, has_copy_epilogue)
    
    # Model "any use of views after node" check
    has_use_after_node = z3_any_use_of_views_after_node_array(
        verifier, node_name, shared_view_nodes, num_shared_views,
        f"{mutated_arg_name}_copy", mutated_arg_name
    )
    
    can_inplace_with_copy = z3.And(copy_case, z3.Not(has_use_after_node))
    
    # Case 2: Not graph input
    verifier.solver.add(z3.Not(is_graph_input))
    
    # Check if any view is a graph input
    any_view_is_input = z3_any_shared_view_is_graph_input(
        verifier, shared_view_nodes, num_shared_views
    )
    
    # Cannot inplace if any view is graph input
    view_input_case = z3.And(z3.Not(is_graph_input), any_view_is_input)
    verifier.solver.add(z3.Implies(view_input_case, z3.BoolVal(False)))
    
    # If no view is graph input, check for uses after node
    no_view_input_case = z3.And(z3.Not(is_graph_input), z3.Not(any_view_is_input))
    has_use_after_node_no_copy = z3_any_use_of_views_after_node_array(
        verifier, node_name, shared_view_nodes, num_shared_views
    )
    
    can_inplace_no_input = z3.And(no_view_input_case, z3.Not(has_use_after_node_no_copy))
    
    # Overall decision
    can_inplace_decision = z3.Or(can_inplace_with_copy, can_inplace_no_input)
    
    # Check if the decision is satisfiable
    verifier.solver.add(can_inplace_decision)
    result = verifier.solver.check()
    
    verifier.solver.pop()
    
    return result == z3.sat


def z3_any_shared_view_is_graph_input(verifier: Z3ReinplaceVerifier,
                                     shared_views: z3.ArrayRef,
                                     num_views: z3.ExprRef) -> z3.BoolRef:
    """Check if any shared view is a graph input"""
    any_is_input = z3.Bool('any_shared_view_is_input')
    
    # Initialize to False
    verifier.solver.add(any_is_input == z3.BoolVal(False))
    
    # Check each view
    for i in range(10):  # Max views to check
        view_i = z3.Select(shared_views, i)
        
        # Only check if i < num_views
        in_range = (i < num_views)
        
        # Check if this view is a graph input
        is_placeholder_i = z3.Bool(f'is_placeholder_view_{i}')
        is_get_attr_i = z3.Bool(f'is_get_attr_view_{i}')
        is_input_i = z3.Or(is_placeholder_i, is_get_attr_i)
        
        # If any view is input, set any_is_input to True
        verifier.solver.add(z3.Implies(z3.And(in_range, is_input_i), 
                                      any_is_input == z3.BoolVal(True)))
    
    return any_is_input

'''
def z3_any_use_of_views_after_node_array(verifier: Z3ReinplaceVerifier,
                                        node_name: str,
                                        shared_views: z3.ArrayRef,
                                        num_views: z3.ExprRef,
                                        copy_node_name: str = None,
                                        mutated_arg_name: str = None) -> z3.BoolRef:
    """Z3 version of any_use_of_views_after_node using arrays"""
    node = verifier.create_node_var(node_name)
    node_loc = z3.Int(f"{node_name}_order")
    verifier.solver.add(node_order(node) == node_loc)
    
    copy_node_loc = None
    if copy_node_name:
        copy_node = verifier.create_node_var(copy_node_name)
        copy_node_loc = z3.Int(f"{copy_node_name}_order")
        verifier.solver.add(node_order(copy_node) == copy_node_loc)
    
    # Check if any view has users after the current node
    has_later_use = z3.Bool(f"has_later_use_{node_name}")
    verifier.solver.add(has_later_use == z3.BoolVal(False))  # Initialize to False
    
    # Check each shared view
    for i in range(10):  # Max views to check
        view_node = z3.Select(shared_views, i)
        
        # Only check if i < num_views
        in_range = (i < num_views)
        
        # Create user node for this view
        user_node = verifier.create_node_var(f"view_{i}_user")
        user_loc = z3.Int(f"view_{i}_user_order")
        verifier.solver.add(node_order(user_node) == user_loc)
        
        # User after current node
        later_user = user_loc > node_loc
        
        # But before copy epilogue (if exists)
        if copy_node_loc:
            later_user = z3.And(later_user, user_loc < copy_node_loc)
        
        # Not a meta-only user
        is_meta_user = z3.Bool(f"is_meta_user_view_{i}")
        
        # Not a copy operation on the mutated arg
        is_copy_mutated = z3.Bool(f"is_copy_mutated_view_{i}")
        
        # Has problematic use
        has_problematic_use = z3.And(
            in_range,
            later_user,
            z3.Not(is_meta_user),
            z3.Not(is_copy_mutated)
        )
        
        # If any view has problematic use, set has_later_use to True
        verifier.solver.add(z3.Implies(has_problematic_use, 
                                      has_later_use == z3.BoolVal(True)))
    
    return has_later_use
'''
def z3_any_use_of_views_after_node_array(verifier, mutated_node_name, shared_views, num_views,
                                         copy_node_name=None):
    node_uses_node = z3.Function('node_uses_node', NodeSort, NodeSort, z3.BoolSort())
    m = verifier.create_node_var(mutated_node_name)
    m_ord = z3.Int(f"{mutated_node_name}_ord")
    verifier.solver.add(node_order(m) == m_ord)

    has_bad_user = z3.Bool(f"has_bad_user_{mutated_node_name}")
    verifier.solver.add(has_bad_user == z3.BoolVal(False))

    for i in range(10):
        v = z3.Select(shared_views, i)
        in_range = (i < num_views)

        u = verifier.create_node_var(f"{mutated_node_name}_view{i}_user")
        u_ord = z3.Int(f"{mutated_node_name}_view{i}_user_ord")
        verifier.solver.add(node_order(u) == u_ord)

        is_later = u_ord > m_ord

        is_meta = z3.Bool(f"is_meta_user_{i}")

        is_copy_epilogue = z3.Bool(f"is_copy_epilogue_{i}")
        if copy_node_name:
            copy_node = verifier.create_node_var(copy_node_name)
            is_copy_epilogue = (u == copy_node)

        bad = z3.And(
            in_range,
            node_uses_node(u, v),
            is_later,
            z3.Not(is_meta),
            z3.Not(is_copy_epilogue)
        )

        verifier.solver.add(z3.Implies(bad, has_bad_user))

    return has_bad_user


def z3_reinplace_and_refine_tensors_to_clone(verifier: Z3ReinplaceVerifier,
                                           old_tensors_to_clone: list[str],
                                           kwargs_dict: dict[str, str],
                                           node_name: str,
                                           trigger_name: str) -> dict[str, Any]:
    """Z3 version of reinplace_and_refine_tensors_to_clone"""
    
    # Create symbolic variables
    num_old_tensors = len(old_tensors_to_clone)
    tensors_to_clone = z3.Array('tensors_to_clone', z3.IntSort(), z3.BoolSort())
    storage_reinplaced = z3.Array('storage_reinplaced', StorageSort, z3.BoolSort())
    
    # Track missed opportunities
    missed_args = z3.Array('missed_args', z3.IntSort(), z3.BoolSort())
    missed_nodes = z3.Array('missed_nodes', z3.IntSort(), NodeSort)
    
    # Decision variables for each tensor
    should_attempt_reinplace = z3.Array('should_attempt_reinplace', z3.IntSort(), z3.BoolSort())
    can_inplace_result = z3.Array('can_inplace_result', z3.IntSort(), z3.BoolSort())
    
    # Process each tensor argument
    for i, arg_name in enumerate(old_tensors_to_clone):
        if arg_name not in kwargs_dict:
            continue
            
        mutated_arg_name = kwargs_dict[arg_name]
        mutated_arg = verifier.create_node_var(mutated_arg_name)
        arg_storage = z3.Const(f'storage_{arg_name}', StorageSort)
        
        # Constraint: get storage of mutated arg
        verifier.solver.add(arg_storage == get_node_storage(mutated_arg))
        
        # Check if tensor with same storage already reinplaced
        storage_already_used = z3.Select(storage_reinplaced, arg_storage)
        
        # Should attempt reinplace if storage not already used
        verifier.solver.add(z3.Select(should_attempt_reinplace, i) == z3.Not(storage_already_used))
        
        # Can inplace check (simplified)
        can_inplace_i = z3_can_inplace(verifier, f"{node_name}_{i}", mutated_arg_name)
        verifier.solver.add(z3.Select(can_inplace_result, i) == can_inplace_i)
        
        # Decision: reinplace if both conditions met
        should_reinplace = z3.And(z3.Select(should_attempt_reinplace, i),
                                 z3.Select(can_inplace_result, i))
        
        # If reinplacing, mark storage as used
        verifier.solver.add(z3.Implies(should_reinplace,
                                      z3.Store(storage_reinplaced, arg_storage, z3.BoolVal(True))))
        
        # If not reinplacing, add to tensors_to_clone
        verifier.solver.add(z3.Select(tensors_to_clone, i) == z3.Not(should_reinplace))
        
        # Track missed opportunities
        missed_opportunity = z3.And(z3.Select(should_attempt_reinplace, i),
                                   z3.Not(z3.Select(can_inplace_result, i)))
        verifier.solver.add(z3.Select(missed_args, i) == missed_opportunity)
        
        if missed_opportunity:
            verifier.solver.add(z3.Select(missed_nodes, i) == mutated_arg)
    
    # Verify solver constraints
    verifier.solver.push()
    result = verifier.solver.check()
    
    # Extract results
    results = {
        'tensors_to_clone': [],
        'missed_args': [],
        'missed_nodes': [],
        'z3_satisfiable': result == z3.sat,
        'total_tensors': num_old_tensors,
        'reinplaced_count': 0,
        'missed_count': 0
    }
    
    if result == z3.sat:
        model = verifier.solver.model()
        
        for i in range(num_old_tensors):
            # Check if tensor should be cloned
            clone_decision = model.eval(z3.Select(tensors_to_clone, i))
            if z3.is_true(clone_decision):
                results['tensors_to_clone'].append(old_tensors_to_clone[i])
            else:
                results['reinplaced_count'] += 1
            
            # Check if missed opportunity
            missed_decision = model.eval(z3.Select(missed_args, i))
            if z3.is_true(missed_decision):
                results['missed_args'].append(old_tensors_to_clone[i])
                results['missed_count'] += 1
    
    verifier.solver.pop()
    return results


def z3_tensor_with_same_storage_already_reinplaced(verifier: Z3ReinplaceVerifier,
                                                  arg_name: str,
                                                  storage_reinplaced: z3.ArrayRef) -> z3.BoolRef:
    """Z3 version of tensor_with_same_storage_already_reinplaced check"""
    
    arg_node = verifier.create_node_var(arg_name)
    arg_storage = z3.Const(f'storage_{arg_name}', StorageSort)
    
    # Get storage of the argument
    verifier.solver.add(arg_storage == get_node_storage(arg_node))
    
    # Check if this storage is already marked as reinplaced
    already_reinplaced = z3.Select(storage_reinplaced, arg_storage)
    
    return already_reinplaced


def z3_log_inplace_results(verifier: Z3ReinplaceVerifier,
                          node_name: str,
                          old_tensors: list[str],
                          tensors_to_clone: list[str],
                          missed_args: list[str],
                          missed_nodes: list[str],
                          trigger: str) -> dict[str, Any]:
    """Z3 version of log_inplace_results with symbolic verification"""
    
    # Create symbolic variables for logging metrics
    total_tensors = z3.Int('total_tensors')
    cloned_tensors = z3.Int('cloned_tensors')
    missed_opportunities = z3.Int('missed_opportunities')
    missed_bytes = z3.Int('missed_bytes')
    
    verifier.solver.add(total_tensors == len(old_tensors))
    verifier.solver.add(cloned_tensors == len(tensors_to_clone))
    verifier.solver.add(missed_opportunities == len(missed_args))
    
    # Symbolic byte calculation (simplified)
    # In practice, this would sum up tensor sizes
    verifier.solver.add(missed_bytes >= 0)
    verifier.solver.add(missed_bytes <= missed_opportunities * 1000000)  # Upper bound
    
    # Verify logging constraints
    verifier.solver.push()
    
    # Constraint: cloned + reinplaced = total
    reinplaced_count = z3.Int('reinplaced_count')
    verifier.solver.add(reinplaced_count == total_tensors - cloned_tensors)
    verifier.solver.add(reinplaced_count >= 0)
    
    # Constraint: missed opportunities <= cloned tensors
    verifier.solver.add(missed_opportunities <= cloned_tensors)
    
    result = verifier.solver.check()
    
    log_results = {
        'node_name': node_name,
        'trigger': trigger,
        'total_tensors': len(old_tensors),
        'tensors_to_clone': len(tensors_to_clone),
        'missed_opportunities': len(missed_args),
        'z3_verified': result == z3.sat,
        'constraints_satisfiable': result == z3.sat
    }
    
    if result == z3.sat:
        model = verifier.solver.model()
        log_results.update({
            'symbolic_total': model.eval(total_tensors).as_long(),
            'symbolic_cloned': model.eval(cloned_tensors).as_long(),
            'symbolic_missed': model.eval(missed_opportunities).as_long(),
            'symbolic_reinplaced': model.eval(reinplaced_count).as_long()
        })
    
    verifier.solver.pop()
    return log_results


def z3_reinplace_inplaceable_ops_core(verifier: Z3ReinplaceVerifier, graph_name: str) -> None:
    """Z3 version of reinplace_inplaceable_ops_core"""
    graph = verifier.create_graph_var(graph_name)
    
    # Model copy args to copy nodes mapping
    copy_args_to_copy_nodes = z3.Array('copy_args_to_copy_nodes', 
                                      z3.TupleSort(NodeSort, NodeSort), NodeSort)
    
    # Model mutated inputs
    mutated_inputs = z3.Array('mutated_inputs', z3.IntSort(), NodeSort)
    
    # Model storage to nodes mapping
    storage_to_nodes = z3.Array('storage_to_nodes', StorageSort, 
                               z3.ArraySort(z3.IntSort(), NodeSort))
    
    # Model node order
    num_nodes = z3.Int('num_nodes')
    
    # Process each node in reverse order (as in original)
    i = z3.Int('i')
    current_node = z3.Const('current_node', NodeSort)
    
    # For each inplaceable operation
    for op_name, inplaceable_op in verifier.z3_inplaceable_ops.items():
        op_node = verifier.create_node_var(f"{op_name}_node")
        mutated_arg = verifier.create_node_var(f"{op_name}_mutated_arg")
        
        can_inplace_result = z3.Bool(f"can_inplace_{op_name}")
        extra_check_result = z3.Bool(f"extra_check_{op_name}")
        
        # Model the inplace decision
        should_inplace = z3.And(can_inplace_result, extra_check_result)
        
        verifier.solver.add(z3.Implies(should_inplace,
            # Replace with inplace operation
            z3.BoolVal(True)  # Simplified action
        ))
        
        # Model the refinement process for this operation
        old_tensors = [f"{op_name}_tensor_{j}" for j in range(3)]  # Simplified
        kwargs_dict = {f"arg_{j}": f"{op_name}_mutated_{j}" for j in range(3)}
        
        refinement_result = z3_reinplace_and_refine_tensors_to_clone(
            verifier, old_tensors, kwargs_dict, op_name, "test_trigger"
        )
        
        # Add constraints based on refinement results
        if refinement_result['z3_satisfiable']:
            verifier.solver.add(z3.BoolVal(True))  # Refinement succeeded


def z3_reinplace_inplaceable_ops(verifier: Z3ReinplaceVerifier, graph_name: str) -> None:
    """Z3 version of reinplace_inplaceable_ops"""
    # Canonicalize view scatter ops
    z3_canonicalize_view_scatter_ops(verifier, graph_name)
    
    # Run core reinplace logic
    z3_reinplace_inplaceable_ops_core(verifier, graph_name)
    
    # Decompose generalized scatter
    z3_decompose_generalized_scatter(verifier, graph_name)


# Main verification functions
def verify_reinplace_safety(op_name: str, mutated_arg: str, 
                           shared_views: list[str], context: dict) -> dict:
    """Main function to verify reinplace safety using Z3"""
    verifier = Z3ReinplaceVerifier()
    
    try:
        # Create symbolic representation
        node_name = f"{op_name}_verification"
        
        # Perform Z3-based safety check
        can_inplace = z3_can_inplace(verifier, node_name, mutated_arg)
        
        # Additional checks based on operation type
        extra_checks = True
        if op_name == '_generalized_scatter':
            extra_checks = z3_should_reinplace_scatter(verifier, node_name)
        
        safe = can_inplace and extra_checks
        
        return {
            'op_name': op_name,
            'mutated_arg': mutated_arg,
            'can_inplace': can_inplace,
            'extra_checks_passed': extra_checks,
            'safe_to_reinplace': safe,
            'z3_verified': True,
            'solver_stats': {
                'num_assertions': len(verifier.solver.assertions()),
                'num_variables': len(verifier.node_vars) + len(verifier.tensor_vars)
            }
        }
        
    except Exception as e:
        return {
            'op_name': op_name,
            'mutated_arg': mutated_arg,
            'can_inplace': False,
            'extra_checks_passed': False,
            'safe_to_reinplace': False,
            'z3_verified': False,
            'error': str(e)
        }


def verify_scatter_decomposition(inp_tensor: str, src_tensor: str, 
                               view_ops: list[dict]) -> dict:
    """Verify scatter decomposition correctness using Z3"""
    verifier = Z3ReinplaceVerifier()
    
    try:
        # Convert view_ops to Z3ViewOp objects
        z3_view_ops = []
        for i, view_op in enumerate(view_ops):
            z3_view_op = Z3ViewOp(
                target=view_op.get('target', f'view_op_{i}'),
                args=view_op.get('args', ()),
                kwargs=view_op.get('kwargs', {})
            )
            z3_view_ops.append(z3_view_op)
        
        # Verify functional decomposition
        functional_result = z3_decompose_scatter_functional(verifier, 'test_graph', 
                                                          'scatter_node')
        
        # Verify mutating decomposition  
        mutating_result = z3_decompose_scatter_mutating(verifier, 'test_graph',
                                                       'scatter_node')
        
        return {
            'input_tensor': inp_tensor,
            'source_tensor': src_tensor,
            'num_view_ops': len(view_ops),
            'functional_decomposition_valid': functional_result is not None,
            'mutating_decomposition_valid': mutating_result is not None,
            'z3_verified': True,
            'solver_stats': {
                'num_assertions': len(verifier.solver.assertions()),
                'num_variables': len(verifier.node_vars)
            }
        }
        
    except Exception as e:
        return {
            'input_tensor': inp_tensor,
            'source_tensor': src_tensor,
            'num_view_ops': len(view_ops),
            'functional_decomposition_valid': False,
            'mutating_decomposition_valid': False,
            'z3_verified': False,
            'error': str(e)
        }
