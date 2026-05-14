"""
core.py — Measure, and the value/adjoint type system.

The type system distinguishes two kinds of values that a computation-graph
node can produce:

  ScalarValue  — a real number ψ(P) ∈ ℝ
  L2Value      — an element of L²(P), i.e. a function of an observation v

and two corresponding adjoint types used in the backward pass:

  ScalarAdjoint — a real-valued multiplier λ ∈ ℝ
  L2Adjoint     — a function a ∈ L²(P) of the current observation

All symbolic arithmetic on adjoint values (accumulation, scaling) is
performed on the underlying Expr trees and simplified eagerly.
"""

from __future__ import annotations
from typing import Optional
from .expr import Expr, Const, ObsVar, CondExpect, MargExpect, MargVar


# ---------------------------------------------------------------------------
# Measure
# ---------------------------------------------------------------------------

class Measure:
    """A probability measure P with a named set of random variables.

    Args:
        name: Short name used in symbolic output (e.g. "P").
        variables: Names of all random variables in the domain of P.

    Example::

        P = Measure("P", ["Y", "X"])
        Y = P.obs("Y")   # → ObsVar("Y")
        expr = P.E(Y)    # → MargExpect("P", ObsVar("Y"))
    """

    def __init__(self, name: str, variables: list[str]):
        self.name = name
        self.variables = list(variables)
        self._obs: dict[str, ObsVar] = {v: ObsVar(v) for v in variables}

    # ---- convenience expression builders ----------------------------------

    def obs(self, name: str) -> ObsVar:
        """Return the ObsVar for a named variable."""
        if name not in self._obs:
            raise ValueError(
                f"Variable {name!r} is not in measure {self.name!r}. "
                f"Known variables: {self.variables}"
            )
        return self._obs[name]

    def E(self, dep: Expr, given: Optional[list[str]] = None) -> Expr:
        """Build E_P[dep] or E_P[dep | given₁=·, ...].

        Args:
            dep: The expression under the expectation.
            given: If provided, list of variable names to condition on.
        """
        if given:
            return CondExpect(self.name, dep, given)
        return MargExpect(self.name, dep)

    def Var(self, dep: Expr) -> Expr:
        """Build Var_P[dep]."""
        return MargVar(self.name, dep)

    def __repr__(self) -> str:
        return self.name


# ---------------------------------------------------------------------------
# Node value types
# ---------------------------------------------------------------------------

class NodeValue:
    """Abstract base: the value type produced by a computation-graph node."""


class ScalarValue(NodeValue):
    """A scalar functional ψ(P) ∈ ℝ, represented as a symbolic Expr.

    The expr may contain MargExpect / MargVar atoms (distributional
    constants) but must not contain ObsVar atoms (it does not depend on
    a particular observation).
    """

    def __init__(self, expr: Expr):
        self.expr = expr

    def __repr__(self) -> str:
        return repr(self.expr)


class L2Value(NodeValue):
    """An element of L²(P), represented as a function of an observation v.

    The expr may contain ObsVar atoms (values at the current observation)
    as well as CondExpect / MargExpect / MargVar atoms (distributional
    quantities).

    Args:
        expr: Symbolic expression representing h(v).
        measure: The underlying measure P.
        domain_vars: Which observed variables h depends on.
    """

    def __init__(self, expr: Expr, measure: Measure, domain_vars: list[str]):
        self.expr = expr
        self.measure = measure
        self.domain_vars = list(domain_vars)

    def __repr__(self) -> str:
        return repr(self.expr)


# ---------------------------------------------------------------------------
# Adjoint value types
# ---------------------------------------------------------------------------

class AdjointValue:
    """Abstract base: the type of an adjoint in the backward pass."""


class ScalarAdjoint(AdjointValue):
    """A scalar adjoint λ ∈ ℝ for a scalar-valued node.

    Represents d(ψ)/d(node_value) — how much the functional ψ changes
    per unit change in this node's scalar output.
    """

    def __init__(self, expr: Expr):
        self.expr = expr

    # ---- factory methods --------------------------------------------------

    @classmethod
    def zero(cls) -> "ScalarAdjoint":
        return cls(Const(0))

    @classmethod
    def one(cls) -> "ScalarAdjoint":
        return cls(Const(1))

    # ---- arithmetic -------------------------------------------------------

    def add(self, other: "ScalarAdjoint") -> "ScalarAdjoint":
        """Accumulate another scalar adjoint into this one."""
        return ScalarAdjoint((self.expr + other.expr).simplify())

    def scale(self, factor: Expr) -> "ScalarAdjoint":
        """Multiply this adjoint by a scalar Expr factor."""
        return ScalarAdjoint((factor * self.expr).simplify())

    def is_zero(self) -> bool:
        return isinstance(self.expr, Const) and self.expr.value == 0

    def __repr__(self) -> str:
        return f"ScalarAdj({self.expr})"


class L2Adjoint(AdjointValue):
    """An L²(P) adjoint a ∈ L²(P) for an L²-valued node.

    Represents d(ψ)/d(node_function) — how much the functional ψ changes
    per unit perturbation of this node's L² output in a given direction.
    The expr is a function of the current observation v (may contain ObsVars).
    """

    def __init__(self, expr: Expr, measure: Measure):
        self.expr = expr
        self.measure = measure

    # ---- factory methods --------------------------------------------------

    @classmethod
    def zero(cls, measure: Measure) -> "L2Adjoint":
        return cls(Const(0), measure)

    # ---- arithmetic -------------------------------------------------------

    def add(self, other: "L2Adjoint") -> "L2Adjoint":
        """Accumulate another L² adjoint (must share the same measure)."""
        if self.measure is not other.measure:
            raise ValueError(
                f"Cannot accumulate L2Adjoints from different measures: "
                f"{self.measure.name!r} vs {other.measure.name!r}"
            )
        return L2Adjoint((self.expr + other.expr).simplify(), self.measure)

    def scale(self, factor: Expr) -> "L2Adjoint":
        """Pointwise multiply this L² adjoint by an Expr (may depend on v)."""
        return L2Adjoint((factor * self.expr).simplify(), self.measure)

    def is_zero(self) -> bool:
        return isinstance(self.expr, Const) and self.expr.value == 0

    def __repr__(self) -> str:
        return f"L2Adj({self.expr})"
