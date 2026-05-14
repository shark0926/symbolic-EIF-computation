"""
graph.py — ComputationGraph and GraphNode base class.

Every statistical primitive and raw-variable node is a GraphNode.
The graph orchestrates:

  1. Forward pass  — computes symbolic values (ScalarValue / L2Value)
                     bottom-up in topological order.

  2. Backward pass — propagates scalar adjoints / L2 adjoints top-down,
                     accumulating the EIF contribution from each node
                     (Algorithm 2 of Luedtke 2025).

  3. eif()         — runs both passes and returns the total symbolic EIF
                     as a single Expr.

Usage example (after implementing primitives)::

    P = Measure("P", ["Y", "X"])
    graph = ComputationGraph()

    Y_node = graph.add(RandomVariable(P, "Y"))
    ...
    psi   = graph.add(DifferentiableFunction(...))

    eif_expr = graph.eif(psi)
    print(eif_expr)
"""

from __future__ import annotations
from typing import Optional, TypeVar
from .core import (
    Measure,
    NodeValue, ScalarValue, L2Value,
    AdjointValue, ScalarAdjoint, L2Adjoint,
)
from .expr import Expr, Const, Sum

_T = TypeVar("_T", bound="GraphNode")

# Module-level counter for auto-generating node IDs
_node_counter = 0


def _next_id() -> int:
    global _node_counter
    _node_counter += 1
    return _node_counter


# ---------------------------------------------------------------------------
# GraphNode
# ---------------------------------------------------------------------------

class GraphNode:
    """A node in the EIF computation graph.

    Each node represents either a raw random variable (leaf) or a
    statistical primitive (internal node).  Nodes are connected through
    ``parents``: the edge ``parent → child`` means "child depends on parent".

    Lifecycle
    ---------
    1. Construction: set ``measure``, ``parents``, optional ``name``.
    2. ``forward()``:  implement to return the symbolic NodeValue.
    3. ``eif_contribution()``: implement to return this node's EIF term
       (using ``self._adjoint`` which has been accumulated by then).
    4. ``propagate_adjoint()``: implement to push this node's adjoint
       onto its parents via ``parent._accumulate_adjoint(...)``.

    Subclasses
    ----------
    See ``primitives.py`` for concrete implementations.
    """

    def __init__(
        self,
        measure: Measure,
        *parents: "GraphNode",
        name: Optional[str] = None,
    ):
        self.id: int = _next_id()
        self.measure: Measure = measure
        self.parents: list[GraphNode] = list(parents)
        self.name: str = name or f"node_{self.id}"

        # Set during forward pass by ComputationGraph.forward()
        self._value: Optional[NodeValue] = None

        # Accumulated during backward pass by _accumulate_adjoint()
        self._adjoint: Optional[AdjointValue] = None

        # EIF terms accumulated during backward pass
        self._eif_terms: list[Expr] = []

    # ---- forward pass interface -------------------------------------------

    def forward(self) -> NodeValue:
        """Compute and return the symbolic value of this node.

        Called by ``ComputationGraph.forward()`` after all parents have
        had their ``forward()`` called.  Implementations should read
        ``parent.value`` for parent values.
        """
        raise NotImplementedError(f"{type(self).__name__}.forward()")

    # ---- backward pass interface ------------------------------------------

    def eif_contribution(self) -> Expr:
        """Return this node's contribution to the total EIF.

        Called during the backward pass after ``self._adjoint`` has been
        fully accumulated.  Should use ``self._adjoint`` and
        ``self._value`` to build the EIF term from Table 1 of Luedtke 2025.

        Returns ``Const(0)`` if this node has no direct EIF contribution
        (e.g. DifferentiableFunction only propagates its adjoint onward).
        """
        raise NotImplementedError(f"{type(self).__name__}.eif_contribution()")

    def propagate_adjoint(self) -> None:
        """Propagate this node's adjoint to its parents.

        Called during the backward pass immediately after
        ``eif_contribution()``.  Implementations should call
        ``parent._accumulate_adjoint(...)`` for each parent.

        Leaf nodes (no parents) should implement this as a no-op.
        """
        raise NotImplementedError(f"{type(self).__name__}.propagate_adjoint()")

    # ---- adjoint accumulation (called by children) -----------------------

    def _accumulate_adjoint(self, adj: AdjointValue) -> None:
        """Add ``adj`` to this node's accumulated adjoint.

        Handles both first assignment and subsequent accumulation.
        Scalar and L2 adjoints are handled separately; mixing them raises.
        """
        if self._adjoint is None:
            self._adjoint = adj
        elif isinstance(self._adjoint, ScalarAdjoint) and isinstance(adj, ScalarAdjoint):
            self._adjoint = self._adjoint.add(adj)
        elif isinstance(self._adjoint, L2Adjoint) and isinstance(adj, L2Adjoint):
            self._adjoint = self._adjoint.add(adj)
        else:
            raise TypeError(
                f"Adjoint type mismatch at node {self.name!r}: "
                f"existing {type(self._adjoint).__name__}, "
                f"incoming {type(adj).__name__}"
            )

    # ---- value property ---------------------------------------------------

    @property
    def value(self) -> NodeValue:
        """The symbolic value computed during the forward pass."""
        if self._value is None:
            raise RuntimeError(
                f"Node {self.name!r}: forward pass has not been run yet. "
                "Call ComputationGraph.forward() first."
            )
        return self._value

    # ---- reset ------------------------------------------------------------

    def reset(self) -> None:
        """Clear forward/backward state so the graph can be re-run."""
        self._value = None
        self._adjoint = None
        self._eif_terms = []

    # ---- display ----------------------------------------------------------

    def __repr__(self) -> str:
        cls = type(self).__name__
        val = repr(self._value) if self._value is not None else "?"
        return f"{cls}({self.name!r}, value={val})"


# ---------------------------------------------------------------------------
# ComputationGraph
# ---------------------------------------------------------------------------

class ComputationGraph:
    """Manages a DAG of GraphNodes and runs forward/backward passes.

    Typical workflow::

        graph = ComputationGraph()
        Y = graph.add(RandomVariable(P, "Y"))
        mu = graph.add(ConditionalMean(P, Y, given=["X"]))
        ...
        psi = graph.add(DifferentiableFunction(...))

        eif_expr = graph.eif(psi)

    The graph auto-discovers all reachable nodes via DFS from the root
    (see ``eif()``), so explicit registration is optional when using
    ``add()``.

    Alternatively, pass nodes directly to ``eif()`` without calling
    ``add()`` — the graph will collect them automatically.
    """

    def __init__(self) -> None:
        self._nodes: list[GraphNode] = []
        self._sorted: Optional[list[GraphNode]] = None  # cached topo order

    # ---- node registration ------------------------------------------------

    def add(self, node: _T) -> _T:
        """Register a node with this graph and return it (fluent API)."""
        self._nodes.append(node)
        self._sorted = None  # invalidate cache
        return node

    # ---- topological sort -------------------------------------------------

    def topological_sort(
        self,
        root: Optional[GraphNode] = None,
    ) -> list[GraphNode]:
        """Return nodes in topological order (leaves first, root last).

        If ``root`` is given, only nodes reachable from it are included.
        Otherwise all registered nodes are sorted.
        """
        seed = [root] if root is not None else self._nodes
        visited: set[int] = set()
        order: list[GraphNode] = []

        def dfs(node: GraphNode) -> None:
            if id(node) in visited:
                return
            visited.add(id(node))
            for parent in node.parents:
                dfs(parent)
            order.append(node)

        for start in seed:
            dfs(start)

        self._sorted = order
        return order

    # ---- forward pass -----------------------------------------------------

    def forward(self, root: Optional[GraphNode] = None) -> None:
        """Run the symbolic forward pass.

        Computes ``node._value`` for every node in topological order.
        ``root`` restricts computation to the subgraph reachable from it.
        """
        order = self.topological_sort(root)
        for node in order:
            node._value = node.forward()

    # ---- backward pass ----------------------------------------------------

    def backward(self, root: GraphNode) -> None:
        """Run the symbolic backward pass (Algorithm 2 of Luedtke 2025).

        Seeds ``root`` with adjoint = ScalarAdjoint(1) and propagates
        adjoints in reverse topological order, accumulating EIF
        contributions in ``node._eif_terms``.

        ``root`` must be a scalar-valued node.
        """
        if self._sorted is None:
            self.topological_sort(root)

        # Validate root is scalar
        if not isinstance(root.value, ScalarValue):
            raise ValueError(
                f"Backward pass requires a scalar-valued root node, "
                f"but {root.name!r} has value type {type(root.value).__name__}"
            )

        # Seed the root
        root._accumulate_adjoint(ScalarAdjoint.one())

        # Reverse topological order: from root back to leaves
        for node in reversed(self._sorted):
            if node._adjoint is None:
                # This node was never reached by adjoint propagation
                continue

            # Step 1: collect EIF contribution from this node
            contrib = node.eif_contribution()
            simplified = contrib.simplify()
            if not (isinstance(simplified, Const) and simplified.value == 0):
                node._eif_terms.append(simplified)

            # Step 2: propagate adjoint to parents
            if node.parents:
                node.propagate_adjoint()

    # ---- combined eif() entry point --------------------------------------

    def eif(self, root: GraphNode) -> Expr:
        """Compute and return the symbolic EIF for the functional at ``root``.

        Runs forward pass, then backward pass, then collects all EIF
        contribution terms into a single ``Sum`` expression.

        Args:
            root: The root node representing the scalar functional ψ(P).

        Returns:
            A symbolic ``Expr`` representing φ(v; P), the efficient
            influence function evaluated at a generic observation v.
        """
        self.forward(root)
        self.backward(root)

        all_terms: list[Expr] = []
        for node in self._sorted or []:
            all_terms.extend(node._eif_terms)

        return Sum(all_terms).simplify()

    # ---- utilities --------------------------------------------------------

    def reset(self) -> None:
        """Reset all node states so forward/backward can be re-run."""
        for node in (self._sorted or self._nodes):
            node.reset()
        self._sorted = None

    def summary(self) -> str:
        """Return a human-readable summary of the graph structure."""
        order = self._sorted or self.topological_sort()
        lines = ["ComputationGraph:"]
        for node in order:
            parent_names = [p.name for p in node.parents]
            val_type = type(node._value).__name__ if node._value else "?"
            lines.append(
                f"  {node.name:<30s}  type={type(node).__name__:<25s}"
                f"  parents={parent_names}  value_type={val_type}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Convenience: collect graph from root without explicit graph object
# ---------------------------------------------------------------------------

def compute_eif(root: GraphNode) -> Expr:
    """Compute the symbolic EIF for a scalar functional rooted at ``root``.

    Auto-collects all nodes reachable from ``root`` via DFS.

    Args:
        root: A scalar-valued GraphNode representing ψ(P).

    Returns:
        Symbolic EIF expression φ(v; P).

    Example::

        eif_expr = compute_eif(psi_node)
        print(eif_expr)
    """
    graph = ComputationGraph()
    # Register all reachable nodes
    visited: set[int] = set()

    def collect(node: GraphNode) -> None:
        if id(node) in visited:
            return
        visited.add(id(node))
        for parent in node.parents:
            collect(parent)
        graph.add(node)

    collect(root)
    return graph.eif(root)
