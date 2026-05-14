"""
api.py — High-level fluent API for building EIF computation graphs.

Adds two things:

1.  **Operator overloading on GraphNode** so that arithmetic between nodes
    auto-creates PointwiseOperation (L² inputs) or DifferentiableFunction
    (scalar inputs) with the correct partial derivatives.

2.  **Factory methods on Measure** so statistical primitives can be created
    without naming every node explicitly.

After importing this module the R² example reduces from 18 lines to 6::

    P = Measure("P", ["Y", "X"])
    Y     = P.rv("Y")
    mu    = P.mean(Y, given=["X"])      # ConditionalMean
    mu_z  = P.lift(mu)                  # LiftToDomain
    mse   = P.mean((Y - mu_z) ** 2)    # chain of PointwiseOperations + MarginalMean
    var_y = P.variance(Y)              # Variance
    psi   = 1 - mse / var_y            # DifferentiableFunction chain

    eif = compute_eif(psi)

This module is imported automatically by ``pydimple.symbolic.__init__``.
"""

from __future__ import annotations
from .expr import Const
from .core import Measure, ScalarValue, L2Value
from .graph import GraphNode
from .primitives import (
    RandomVariable,
    ConstantMap,
    Variance,
    ConditionalMean,
    RFoldConditionalMean,
    ConditionalVariance,
    ConditionalCovariance,
    LiftToDomain,
    PointwiseOperation,
    MarginalMean,
    InnerProduct,
    SquaredNorm,
    KernelEmbedding,
    BoundedAffineMap,
    ChangeOfMeasure,
    OptimalValue,
    OptimalSolution,
    PathwiseDiffParameter,
    RootDensity,
    ConditionalDensity,
    DoseResponseFunction,
    CounterfactualDensity,
    DifferentiableFunction,
    FixBinaryArgument,
)

# ---------------------------------------------------------------------------
# Output-type registry
# ---------------------------------------------------------------------------
# Each primitive class advertises whether it produces a scalar or L² value.
# Operators use this to decide which node class to create.

RandomVariable._output_type        = "l2"
ConstantMap._output_type           = "l2"
Variance._output_type              = "scalar"
ConditionalMean._output_type       = "l2"
RFoldConditionalMean._output_type  = "l2"
ConditionalVariance._output_type   = "l2"
ConditionalCovariance._output_type = "l2"
LiftToDomain._output_type          = "l2"
PointwiseOperation._output_type    = "l2"
MarginalMean._output_type          = "scalar"
InnerProduct._output_type          = "scalar"
SquaredNorm._output_type           = "scalar"
KernelEmbedding._output_type       = "l2"
BoundedAffineMap._output_type      = "l2"
ChangeOfMeasure._output_type       = "l2"
OptimalValue._output_type          = "scalar"
OptimalSolution._output_type       = "scalar"
PathwiseDiffParameter._output_type = "l2"
RootDensity._output_type           = "l2"
ConditionalDensity._output_type    = "l2"
DoseResponseFunction._output_type  = "l2"
CounterfactualDensity._output_type = "l2"
DifferentiableFunction._output_type = "scalar"
FixBinaryArgument._output_type     = "l2"


def _otype(node: GraphNode) -> str:
    return getattr(node, "_output_type", "l2")


# ---------------------------------------------------------------------------
# Helpers: create the right node class for an operation
# ---------------------------------------------------------------------------

def _pw(measure, func, partials, *inputs):
    """Create PointwiseOperation (L²) or DifferentiableFunction (scalar)."""
    inputs = list(inputs)
    if any(_otype(n) == "l2" for n in inputs):
        return PointwiseOperation(measure, func, partials, inputs)
    else:
        return DifferentiableFunction(measure, func, partials, inputs)


# ---------------------------------------------------------------------------
# Node arithmetic operators
# ---------------------------------------------------------------------------
# Each operator handles two cases:
#   • node OP node   — two GraphNode operands
#   • node OP number — embed the constant in the lambda (no wrapper node)

def _add(self, other):
    if isinstance(other, (int, float)):
        c = other
        return _pw(self.measure,
                   lambda a: a + Const(c),
                   [lambda a: Const(1)],
                   self)
    return _pw(self.measure,
               lambda a, b: a + b,
               [lambda a, b: Const(1), lambda a, b: Const(1)],
               self, other)

def _radd(self, other):
    return _add(self, other)   # addition is commutative

def _sub(self, other):
    if isinstance(other, (int, float)):
        c = other
        return _pw(self.measure,
                   lambda a: a - Const(c),
                   [lambda a: Const(1)],
                   self)
    return _pw(self.measure,
               lambda a, b: a - b,
               [lambda a, b: Const(1), lambda a, b: Const(-1)],
               self, other)

def _rsub(self, other):
    # other - self  (called when `other` is a plain number, e.g. 1 - node)
    if isinstance(other, (int, float)):
        c = other
        return _pw(self.measure,
                   lambda a: Const(c) - a,
                   [lambda a: Const(-1)],
                   self)
    return _pw(self.measure,
               lambda a, b: a - b,
               [lambda a, b: Const(1), lambda a, b: Const(-1)],
               other, self)

def _mul(self, other):
    if isinstance(other, (int, float)):
        c = other
        return _pw(self.measure,
                   lambda a: a * Const(c),
                   [lambda a: Const(c)],
                   self)
    return _pw(self.measure,
               lambda a, b: a * b,
               [lambda a, b: b, lambda a, b: a],
               self, other)

def _rmul(self, other):
    return _mul(self, other)   # multiplication is commutative

def _truediv(self, other):
    if isinstance(other, (int, float)):
        c = other
        return _pw(self.measure,
                   lambda a: a / Const(c),
                   [lambda a: Const(1) / Const(c)],
                   self)
    return _pw(self.measure,
               lambda a, b: a / b,
               [lambda a, b: Const(1) / b,
                lambda a, b: -(a / (b ** Const(2)))],
               self, other)

def _rtruediv(self, other):
    # other / self  (called when `other` is a plain number)
    if isinstance(other, (int, float)):
        c = other
        return _pw(self.measure,
                   lambda a: Const(c) / a,
                   [lambda a: -(Const(c) / (a ** Const(2)))],
                   self)
    return _pw(self.measure,
               lambda a, b: a / b,
               [lambda a, b: Const(1) / b,
                lambda a, b: -(a / (b ** Const(2)))],
               other, self)

def _neg(self):
    return _pw(self.measure,
               lambda a: -a,
               [lambda a: Const(-1)],
               self)

def _pow(self, n):
    if not isinstance(n, (int, float)):
        raise TypeError(
            f"GraphNode ** n requires a numeric exponent, got {type(n).__name__}"
        )
    _n = n  # capture for closure
    return _pw(self.measure,
               lambda a: a ** Const(_n),
               [lambda a: Const(_n) * a ** Const(_n - 1)],
               self)


# Attach operators to GraphNode
GraphNode.__add__      = _add
GraphNode.__radd__     = _radd
GraphNode.__sub__      = _sub
GraphNode.__rsub__     = _rsub
GraphNode.__mul__      = _mul
GraphNode.__rmul__     = _rmul
GraphNode.__truediv__  = _truediv
GraphNode.__rtruediv__ = _rtruediv
GraphNode.__neg__      = _neg
GraphNode.__pow__      = _pow


# ---------------------------------------------------------------------------
# Measure factory methods
# ---------------------------------------------------------------------------

def _rv(self, name: str) -> RandomVariable:
    """Create a RandomVariable leaf node for the named variable."""
    return RandomVariable(self, name)

def _mean(self, dep: GraphNode, given: list[str] | None = None) -> GraphNode:
    """E_P[dep]  or  E_P[dep | given₁=·, ...]

    Returns a MarginalMean (scalar) if ``given`` is None,
    or a ConditionalMean (L²) if ``given`` is provided.
    """
    if given:
        return ConditionalMean(self, dep, given)
    return MarginalMean(self, dep)

def _variance(self, dep: GraphNode) -> Variance:
    """Var_P[dep(Z)] as a scalar node."""
    return Variance(self, dep)

def _lift(self, node: GraphNode) -> LiftToDomain:
    """Lift an L²(P_X) node to L²(P): z ↦ node(x)."""
    return LiftToDomain(self, node)

def _inner(self, left: GraphNode, right: GraphNode) -> InnerProduct:
    """Inner product E_P[left(Z) * right(Z)]."""
    return InnerProduct(self, left, right)

def _squared_norm(self, dep: GraphNode) -> SquaredNorm:
    """Squared L² norm E_P[dep(Z)^2]."""
    return SquaredNorm(self, dep)

def _fix(self, dep: GraphNode, binary_var: str, value: int) -> FixBinaryArgument:
    """Fix a binary argument, e.g. μ(A,X) -> μ(1,X)."""
    return FixBinaryArgument(self, dep, binary_var, value)

Measure.rv       = _rv
Measure.mean     = _mean
Measure.variance = _variance
Measure.lift     = _lift
Measure.inner    = _inner
Measure.squared_norm = _squared_norm
Measure.fix      = _fix
