"""
primitives.py — The seven Table 1 primitives plus RandomVariable leaf.

Each primitive maps to a row in Table 1 of Luedtke (2025, JRSSB).

Primitive                | Node class               | Value type
-------------------------|--------------------------|------------
(leaf)                   | RandomVariable           | L2Value
Constant map             | ConstantMap              | L2Value
Variance                 | Variance                 | ScalarValue
Conditional mean         | ConditionalMean          | L2Value
Lifting (coordinate proj)| LiftToDomain             | L2Value
Pointwise operations     | PointwiseOperation       | L2Value
Mean (special inner prod)| MarginalMean             | ScalarValue
Inner product            | InnerProduct             | ScalarValue
Squared norm             | SquaredNorm              | ScalarValue
Differentiable function  | DifferentiableFunction   | ScalarValue
Fix binary argument      | FixBinaryArgument        | L2Value


Design convention for partials
-------------------------------
``PointwiseOperation`` and ``DifferentiableFunction`` both accept a
``func`` callable and a ``partials`` list of callables:

    func(*exprs)           → result Expr
    partials[i](*exprs)    → ∂func/∂(i-th input) as Expr

All callables operate on symbolic ``Expr`` objects, not numbers.  For
example, for func = lambda a, b: a - b:

    partials = [lambda a, b: Const(1),   # ∂/∂a
                lambda a, b: Const(-1)]  # ∂/∂b
"""

from __future__ import annotations
from typing import Callable, Optional
from .core import (
    Measure,
    NodeValue, ScalarValue, L2Value,
    AdjointValue, ScalarAdjoint, L2Adjoint,
)
from .expr import (
    Expr, Const, ObsVar, CondExpect, MargExpect, MargVar, SymbolicAtom,
)
from .graph import GraphNode


# ---------------------------------------------------------------------------
# Leaf node: RandomVariable
# ---------------------------------------------------------------------------

class RandomVariable(GraphNode):
    """A raw observed random variable — a leaf in the computation graph.

    Produces an L2Value whose expression is simply ObsVar(var_name).
    Has no EIF contribution and no parents to propagate to.

    This node represents the identity map v ↦ V(v), not a statistical
    functional; it cannot receive an adjoint from the backward pass.

    Args:
        measure:  The measure P whose domain contains this variable.
        var_name: Name of the random variable (must be in P.variables).

    Example::

        P = Measure("P", ["Y", "X"])
        Y = RandomVariable(P, "Y")   # represents v ↦ y
    """

    def __init__(self, measure: Measure, var_name: str, name: Optional[str] = None):
        if var_name not in measure.variables:
            raise ValueError(
                f"Variable {var_name!r} not in measure {measure.name!r}. "
                f"Known variables: {measure.variables}"
            )
        super().__init__(measure, name=name or var_name)
        self.var_name = var_name

    def forward(self) -> L2Value:
        return L2Value(ObsVar(self.var_name), self.measure, [self.var_name])

    def eif_contribution(self) -> Expr:
        # Raw variables are not statistical functionals; no EIF contribution.
        return Const(0)

    def propagate_adjoint(self) -> None:
        # Leaf: no parents to receive an adjoint.
        pass


# ---------------------------------------------------------------------------
# Abstract primitive base
# ---------------------------------------------------------------------------

class Primitive(GraphNode):
    """Abstract base class for all Table 1 primitives.

    Every concrete primitive must implement:
    - ``forward()``           → NodeValue  (symbolic value)
    - ``eif_contribution()``  → Expr        (Table 1 EIF formula)
    - ``propagate_adjoint()`` → None        (Table 1 chain-rule formula)
    """
    _TABLE1_ROW: str = "(unspecified)"  # overridden by subclasses for doc


# ---------------------------------------------------------------------------
# 1. ConstantMap
# ---------------------------------------------------------------------------

class ConstantMap(Primitive):
    """Primitive θ: constant map.

    Maps any observation v to the same L² function:
        θ(P) = (z ↦ dep(z))  ∈ L²(P)

    where ``dep`` is a fixed random variable (not a functional of P).
    This is used in the R² example as θ₁ and θ₃:
        θ₁(P) = (z=(x,y) ↦ y)
        θ₃(P) = (z=(x,y) ↦ y)

    Table 1 row: "Constant map density"
      Adjoint formula: θ̇*_{P,u}(w) = (ν̇_P*(w), 0)
      EIF contribution to f₀: ν̇_P*(w)   [depends on the density ν]
      Adjoint propagated to parents: none (no functional parents)

    Implementation note
    -------------------
    For the Y-identity constant map in the R² example, ν is the
    marginal density of Y under P, and ν̇_P*(w) involves the score
    of P.  In the nonparametric model the contribution is zero
    (since constants have zero pathwise derivative when there are
    no functional parent nodes).

    Args:
        measure:  The measure P.
        dep:      A RandomVariable node representing the variable to wrap.
    """

    _TABLE1_ROW = "Constant map density"

    def __init__(
        self,
        measure: Measure,
        dep: RandomVariable,
        name: Optional[str] = None,
    ):
        # ConstantMap has no functional parents in the graph
        super().__init__(measure, name=name or f"const({dep.var_name})")
        self.dep = dep

    def forward(self) -> L2Value:
        dep_val = self.dep.forward()  # call directly: dep is not a graph node
        return L2Value(dep_val.expr, self.measure, dep_val.domain_vars)

    def eif_contribution(self) -> Expr:
        # Table 1 ("Constant map density"): contribution is ν̇_P*(w), where ν
        # is the density of u(Z) under P.  In the nonparametric model every
        # constant map has a *zero* pathwise derivative — confirmed by Table 2
        # footnote §: "θ₃ and θ₁ are constant maps and so have derivative zero."
        return Const(0)

    def propagate_adjoint(self) -> None:
        # No functional parents → nothing to propagate.
        pass


# ---------------------------------------------------------------------------
# 2. Variance
# ---------------------------------------------------------------------------

class Variance(Primitive):
    """Primitive θ: marginal variance.

    θ(P, u) = Var_P[u(Z)]  ∈ ℝ

    Used in the R² example as θ₂: Var_P[h₁(Z)] = Var_P[Y] = σ²_P.

    Table 1 row: "Conditional variance" (marginal case: no conditioning)
      Forward:   Var_P[u(Z)]
      EIF contribution to f₀:
          f₀ += f_j * Π_P(u)(z)
                 = f_j * (u(z) − E_P[u(Z)])²  − f_j * Var_P[u(Z)]
                 = f_j * ((u(z) − E_P[u(Z)])² − Var_P[u(Z)])
      Adjoint propagated to u:  none in the standard graph
          (u is a ConstantMap / RandomVariable, not a functional parent)

    Args:
        measure: The measure P.
        dep:     An L²-valued node u ∈ L²(P) to take the variance of.
    """

    _TABLE1_ROW = "Conditional variance (marginal)"

    def __init__(
        self,
        measure: Measure,
        dep: GraphNode,
        name: Optional[str] = None,
    ):
        super().__init__(measure, dep, name=name or f"Var[{dep.name}]")
        self.dep = dep

    def forward(self) -> ScalarValue:
        dep_val = dep_value(self.dep)
        return ScalarValue(MargVar(self.measure.name, dep_val.expr))

    def eif_contribution(self) -> Expr:
        # Table 1 ("Conditional variance", marginal case):
        #   f₀ += adj * Π_P(u)(z)
        #       = adj * ((u(z) − E_P[u(Z)])² − Var_P[u(Z)])
        adj = self._adjoint  # ScalarAdjoint
        u_expr  = dep_value(self.dep).expr               # u(z)
        mean_expr = MargExpect(self.measure.name, u_expr) # E_P[u(Z)]
        var_expr  = self.value.expr                       # Var_P[u(Z)]
        residual_sq = (u_expr - mean_expr) ** 2
        return (adj.expr * (residual_sq - var_expr)).simplify()

    def propagate_adjoint(self) -> None:
        # Table 1 ("Conditional variance"):
        #   f_u += z ↦ f_j * 2 * (u(z) − E_P[u(Z)])
        # When dep is a ConstantMap or RandomVariable this adjoint is ignored
        # (their propagate_adjoint is a no-op), so the R² example is unaffected.
        # When dep is a functional node (e.g. LiftToDomain of ConditionalMean),
        # the adjoint correctly flows onward to complete the EIF chain.
        adj = self._adjoint  # ScalarAdjoint
        u_expr = dep_value(self.dep).expr
        mean_expr = MargExpect(self.measure.name, u_expr)
        deriv = (Const(2) * (u_expr - mean_expr)).simplify()
        new_expr = (adj.expr * deriv).simplify()
        self.dep._accumulate_adjoint(L2Adjoint(new_expr, self.measure))


# ---------------------------------------------------------------------------
# 3. ConditionalMean
# ---------------------------------------------------------------------------

class ConditionalMean(Primitive):
    """Primitive θ: conditional mean.

    θ(P, u)(·) = E_P[u(Z) | X = ·]  ∈ L²(P_X)

    Used in the R² example as θ₄: E_P[h₃(Z)|X=·] = E_P[Y|X=·] = μ_P.

    Table 1 row: "Conditional mean"
      Forward:   E_P[u(Z) | X = x]  (as a function of the current x)
      EIF contribution to f₀:
          f₀ += (p_{Z|X} / q_X) * w
              = w(x) * (u(z) − E_P[u(Z)|X=x])
          where w = f_j is the L² adjoint of this node.
      Adjoint propagated to u: none (u is a ConstantMap/leaf)

    Args:
        measure:  The measure P.
        dep:      An L²-valued node u(Z) to condition.
        given:    Variable names to condition on (X).
    """

    _TABLE1_ROW = "Conditional mean"

    def __init__(
        self,
        measure: Measure,
        dep: GraphNode,
        given: list[str],
        name: Optional[str] = None,
    ):
        super().__init__(measure, dep, name=name or f"E[{dep.name}|{','.join(given)}]")
        self.dep = dep
        self.given = list(given)

    def forward(self) -> L2Value:
        dep_val = dep_value(self.dep)
        cond_expr = CondExpect(self.measure.name, dep_val.expr, self.given)
        return L2Value(cond_expr, self.measure, self.given)

    def eif_contribution(self) -> Expr:
        # Table 1 ("Conditional mean"):
        #   f₀ += w(x) * (u(z) − E_P[u(Z)|X=x])
        # where w = f_j is the L² adjoint evaluated at the current observation.
        adj = self._adjoint  # L2Adjoint: w(x) as a function of the observation
        u_expr        = dep_value(self.dep).expr  # u(z)
        cond_mean_expr = self.value.expr           # E_P[u(Z)|X=x]
        return (adj.expr * (u_expr - cond_mean_expr)).simplify()

    def propagate_adjoint(self) -> None:
        # The parent u enters theta(P, u) = E_P[u | X] linearly. Its adjoint
        # is the downstream L2 adjoint viewed on the parent domain. This is
        # essential for nested conditional-mean graphs such as longitudinal
        # G-formula decompositions, where the dependency is itself a graph node
        # rather than a ConstantMap leaf.
        adj = self._adjoint
        self.dep._accumulate_adjoint(L2Adjoint(adj.expr, self.measure))


# ---------------------------------------------------------------------------
# 3b. RFoldConditionalMean
# ---------------------------------------------------------------------------

class RFoldConditionalMean(ConditionalMean):
    """Symbolic r-fold conditional mean.

    In the symbolic engine this has the same forward value and adjoint formula
    as ``ConditionalMean``; the r-fold/sample-splitting distinction matters for
    numerical estimation, not for the closed-form expression tree.
    """

    _TABLE1_ROW = "r-fold conditional mean"


# ---------------------------------------------------------------------------
# 3c. ConditionalVariance
# ---------------------------------------------------------------------------

class ConditionalVariance(Primitive):
    """Primitive θ: conditional variance.

    θ(P, u)(x) = Var_P[u(Z) | X=x]
               = E_P[(u(Z) - E_P[u(Z)|X=x])^2 | X=x]
    """

    _TABLE1_ROW = "Conditional variance"

    def __init__(
        self,
        measure: Measure,
        dep: GraphNode,
        given: list[str],
        name: Optional[str] = None,
    ):
        super().__init__(measure, dep, name=name or f"Var[{dep.name}|{','.join(given)}]")
        self.dep = dep
        self.given = list(given)

    def forward(self) -> L2Value:
        u_expr = dep_value(self.dep).expr
        mean_expr = CondExpect(self.measure.name, u_expr, self.given)
        var_expr = CondExpect(
            self.measure.name,
            ((u_expr - mean_expr) ** Const(2)).simplify(),
            self.given,
        )
        return L2Value(var_expr, self.measure, self.given)

    def eif_contribution(self) -> Expr:
        adj = self._adjoint
        u_expr = dep_value(self.dep).expr
        mean_expr = CondExpect(self.measure.name, u_expr, self.given)
        centered_sq = ((u_expr - mean_expr) ** Const(2)).simplify()
        return (adj.expr * (centered_sq - self.value.expr)).simplify()

    def propagate_adjoint(self) -> None:
        adj = self._adjoint
        u_expr = dep_value(self.dep).expr
        mean_expr = CondExpect(self.measure.name, u_expr, self.given)
        deriv = (Const(2) * (u_expr - mean_expr) * adj.expr).simplify()
        self.dep._accumulate_adjoint(L2Adjoint(deriv, self.measure))


# ---------------------------------------------------------------------------
# 3d. ConditionalCovariance
# ---------------------------------------------------------------------------

class ConditionalCovariance(Primitive):
    """Primitive θ: conditional covariance."""

    _TABLE1_ROW = "Conditional covariance"

    def __init__(
        self,
        measure: Measure,
        left: GraphNode,
        right: GraphNode,
        given: list[str],
        name: Optional[str] = None,
    ):
        super().__init__(measure, left, right, name=name or f"Cov[{left.name},{right.name}|{','.join(given)}]")
        self.left = left
        self.right = right
        self.given = list(given)

    def forward(self) -> L2Value:
        left_expr = dep_value(self.left).expr
        right_expr = dep_value(self.right).expr
        left_mean = CondExpect(self.measure.name, left_expr, self.given)
        right_mean = CondExpect(self.measure.name, right_expr, self.given)
        cov_expr = CondExpect(
            self.measure.name,
            ((left_expr - left_mean) * (right_expr - right_mean)).simplify(),
            self.given,
        )
        return L2Value(cov_expr, self.measure, self.given)

    def eif_contribution(self) -> Expr:
        adj = self._adjoint
        left_expr = dep_value(self.left).expr
        right_expr = dep_value(self.right).expr
        left_mean = CondExpect(self.measure.name, left_expr, self.given)
        right_mean = CondExpect(self.measure.name, right_expr, self.given)
        centered_prod = ((left_expr - left_mean) * (right_expr - right_mean)).simplify()
        return (adj.expr * (centered_prod - self.value.expr)).simplify()

    def propagate_adjoint(self) -> None:
        adj = self._adjoint
        left_expr = dep_value(self.left).expr
        right_expr = dep_value(self.right).expr
        left_mean = CondExpect(self.measure.name, left_expr, self.given)
        right_mean = CondExpect(self.measure.name, right_expr, self.given)
        self.left._accumulate_adjoint(
            L2Adjoint((adj.expr * (right_expr - right_mean)).simplify(), self.measure)
        )
        self.right._accumulate_adjoint(
            L2Adjoint((adj.expr * (left_expr - left_mean)).simplify(), self.measure)
        )


# ---------------------------------------------------------------------------
# 4. LiftToDomain
# ---------------------------------------------------------------------------

class LiftToDomain(Primitive):
    """Primitive θ: lift an L²(P_X) function to L²(P).

    θ(P, u)(z) = u(x)   where z = (x, y, ...)

    Used in the R² example as θ₅: (z ↦ h₄(x)) = (z ↦ μ_P(x)).

    Table 1 row: "Coordinate projection" (κ = marginal projection)
      Forward:   z ↦ u(x)  (the function indexed by x, evaluated at x(z))
      EIF contribution to f₀:  0  (the lifting itself has no EIF)
      Adjoint propagated to u (L²(P_X)):
          f_u += E_P[f_j(Z) | X = ·]
          (i.e. project the L²(P) adjoint down to L²(P_X))

    Args:
        measure:    The measure P.
        l2x_node:   An L²(P_X)-valued node (the function of X to lift).
    """

    _TABLE1_ROW = "Coordinate projection (lifting)"

    def __init__(
        self,
        measure: Measure,
        l2x_node: GraphNode,
        name: Optional[str] = None,
    ):
        super().__init__(measure, l2x_node, name=name or f"lift({l2x_node.name})")
        self.l2x_node = l2x_node

    def forward(self) -> L2Value:
        l2x_val = dep_value(self.l2x_node)
        # Same expression, now viewed as a function of all of z (not just x)
        return L2Value(l2x_val.expr, self.measure, l2x_val.domain_vars)

    def eif_contribution(self) -> Expr:
        # Lifting itself contributes nothing to the EIF.
        return Const(0)

    def propagate_adjoint(self) -> None:
        # Table 1 ("Coordinate projection"):
        #   f_u += E_P[f_j(Z) | X = ·]
        # Project the L²(P) adjoint down to L²(P_X) by conditioning on the
        # variables that u (the lifted function) depends on.
        adj = self._adjoint  # L2Adjoint with expr a(z)
        given = dep_value(self.l2x_node).domain_vars  # e.g. ["X"]
        projected = CondExpect(self.measure.name, adj.expr, given)
        self.l2x_node._accumulate_adjoint(L2Adjoint(projected, self.measure))


# ---------------------------------------------------------------------------
# 5. PointwiseOperation
# ---------------------------------------------------------------------------

class PointwiseOperation(Primitive):
    """Primitive θ: apply a differentiable map pointwise to L² inputs.

    θ(P, ū)(z) = f_z(u₁(z), ..., u_d(z))   for  uᵢ ∈ L²(P)

    Used in R² as θ₆: z ↦ (y − h₅(z))² = (y − μ_P(x))².

    Table 1 row: "Pointwise operations"
      Forward:   z ↦ f_z(u₁(z), ..., u_d(z))
      EIF contribution to f₀:  0
      Adjoint propagated to uᵢ:
          f_{uᵢ} += z ↦ w(z) * (∂f_z/∂uᵢ)(u₁(z), ..., u_d(z))
          where w = f_j is the L² adjoint of this node.

    Args:
        measure:    The measure P.
        func:       Callable (*exprs) → Expr for the forward pass.
        partials:   List of callables; partials[i](*exprs) → ∂func/∂(i-th) as Expr.
        l2_inputs:  The L²-valued parent nodes u₁, ..., u_d.

    Example (z ↦ (y − h)²)::

        sq_res = PointwiseOperation(
            P,
            func=lambda r: r ** 2,
            partials=[lambda r: Const(2) * r],
            l2_inputs=[residual_node],
        )
    """

    _TABLE1_ROW = "Pointwise operations"

    def __init__(
        self,
        measure: Measure,
        func: Callable[..., Expr],
        partials: list[Callable[..., Expr]],
        l2_inputs: list[GraphNode],
        name: Optional[str] = None,
    ):
        if len(partials) != len(l2_inputs):
            raise ValueError(
                f"PointwiseOperation: len(partials)={len(partials)} "
                f"must equal len(l2_inputs)={len(l2_inputs)}"
            )
        super().__init__(
            measure,
            *l2_inputs,
            name=name or f"pw({', '.join(n.name for n in l2_inputs)})",
        )
        self.func = func
        self.partials = partials
        self.l2_inputs = list(l2_inputs)

    def forward(self) -> L2Value:
        input_exprs = []
        domain_vars: list[str] = []
        for inp in self.l2_inputs:
            val = dep_value(inp)
            input_exprs.append(val.expr)
            for v in val.domain_vars:
                if v not in domain_vars:
                    domain_vars.append(v)
        result = self.func(*input_exprs).simplify()
        return L2Value(result, self.measure, domain_vars)

    def eif_contribution(self) -> Expr:
        # Pointwise operations have no direct EIF contribution.
        return Const(0)

    def propagate_adjoint(self) -> None:
        # Table 1 ("Pointwise operations"):
        #   f_{uᵢ} += z ↦ w(z) * (∂f_z/∂uᵢ)(u₁(z), ..., u_d(z))
        adj = self._adjoint  # L2Adjoint with expr w(z)
        input_exprs = [dep_value(inp).expr for inp in self.l2_inputs]
        for inp, partial_fn in zip(self.l2_inputs, self.partials):
            dfdui = partial_fn(*input_exprs)
            new_expr = (adj.expr * dfdui).simplify()
            inp._accumulate_adjoint(L2Adjoint(new_expr, self.measure))


# ---------------------------------------------------------------------------
# 6. MarginalMean
# ---------------------------------------------------------------------------

class MarginalMean(Primitive):
    """Primitive θ: marginal mean (inner product with constant 1).

    θ(P, u) = E_P[u(Z)]  ∈ ℝ

    Used in R² as θ₇: E_P[h₆(Z)] = E_P[(Y−μ_P(X))²] = γ_P.

    Table 1 row: "Inner product" (with the constant-1 function)
      Forward:   E_P[u(Z)]
      EIF contribution to f₀:
          f₀ += w * (u(z) − E_P[u(Z)])
          where w = f_j is the scalar adjoint of this node.
      Adjoint propagated to u:
          f_u += z ↦ w  (constant L² adjoint equal to the scalar adjoint)

    Args:
        measure: The measure P.
        dep:     An L²-valued node u(Z).
    """

    _TABLE1_ROW = "Inner product / marginal mean"

    def __init__(
        self,
        measure: Measure,
        dep: GraphNode,
        name: Optional[str] = None,
    ):
        super().__init__(measure, dep, name=name or f"E[{dep.name}]")
        self.dep = dep

    def forward(self) -> ScalarValue:
        dep_val = dep_value(self.dep)
        return ScalarValue(MargExpect(self.measure.name, dep_val.expr))

    def eif_contribution(self) -> Expr:
        # Table 1 ("Inner product"):
        #   f₀ += w * (u(z) − E_P[u(Z)])
        adj = self._adjoint  # ScalarAdjoint
        u_expr   = dep_value(self.dep).expr            # u(z)
        mean_expr = self.value.expr                    # E_P[u(Z)]
        return (adj.expr * (u_expr - mean_expr)).simplify()

    def propagate_adjoint(self) -> None:
        # Table 1 ("Inner product"):
        #   f_u += z ↦ w   (constant L² adjoint equal to the scalar adjoint)
        adj = self._adjoint  # ScalarAdjoint: scalar w
        self.dep._accumulate_adjoint(L2Adjoint(adj.expr, self.measure))


# ---------------------------------------------------------------------------
# 7. InnerProduct
# ---------------------------------------------------------------------------

class InnerProduct(Primitive):
    """Primitive θ: inner product between two L²(P) functions.

    θ(P, u₁, u₂) = E_P[u₁(Z) u₂(Z)]  ∈ ℝ

    This is the general Table 1 inner-product primitive. ``MarginalMean`` is
    the special case where ``u₂ ≡ 1``.

    Table 1 row: "Inner product"
      Forward:   E_P[u₁(Z) u₂(Z)]
      EIF contribution to f₀:
          f₀ += w * (u₁(z)u₂(z) − E_P[u₁(Z)u₂(Z)])
      Adjoint propagated to u₁:
          f_{u₁} += z ↦ w * u₂(z)
      Adjoint propagated to u₂:
          f_{u₂} += z ↦ w * u₁(z)

    Args:
        measure: The measure P.
        left:    First L²-valued node.
        right:   Second L²-valued node.
    """

    _TABLE1_ROW = "Inner product"

    def __init__(
        self,
        measure: Measure,
        left: GraphNode,
        right: GraphNode,
        name: Optional[str] = None,
    ):
        super().__init__(measure, left, right, name=name or f"<{left.name},{right.name}>")
        self.left = left
        self.right = right

    def forward(self) -> ScalarValue:
        left_expr = dep_value(self.left).expr
        right_expr = dep_value(self.right).expr
        return ScalarValue(MargExpect(self.measure.name, (left_expr * right_expr).simplify()))

    def eif_contribution(self) -> Expr:
        adj = self._adjoint
        left_expr = dep_value(self.left).expr
        right_expr = dep_value(self.right).expr
        pointwise = (left_expr * right_expr).simplify()
        return (adj.expr * (pointwise - self.value.expr)).simplify()

    def propagate_adjoint(self) -> None:
        adj = self._adjoint
        left_expr = dep_value(self.left).expr
        right_expr = dep_value(self.right).expr
        self.left._accumulate_adjoint(L2Adjoint((adj.expr * right_expr).simplify(), self.measure))
        self.right._accumulate_adjoint(L2Adjoint((adj.expr * left_expr).simplify(), self.measure))


# ---------------------------------------------------------------------------
# 8. SquaredNorm
# ---------------------------------------------------------------------------

class SquaredNorm(Primitive):
    """Primitive θ: squared L²(P) norm.

    θ(P, u) = ||u||²_P = E_P[u(Z)²]  ∈ ℝ

    Table 1 row: "Squared norm"
      Forward:   E_P[u(Z)²]
      EIF contribution to f₀:
          f₀ += w * (u(z)² − E_P[u(Z)²])
      Adjoint propagated to u:
          f_u += z ↦ 2 w u(z)

    Args:
        measure: The measure P.
        dep:     L²-valued node u.
    """

    _TABLE1_ROW = "Squared norm"

    def __init__(
        self,
        measure: Measure,
        dep: GraphNode,
        name: Optional[str] = None,
    ):
        super().__init__(measure, dep, name=name or f"||{dep.name}||^2")
        self.dep = dep

    def forward(self) -> ScalarValue:
        u_expr = dep_value(self.dep).expr
        return ScalarValue(MargExpect(self.measure.name, (u_expr ** Const(2)).simplify()))

    def eif_contribution(self) -> Expr:
        adj = self._adjoint
        u_expr = dep_value(self.dep).expr
        pointwise = (u_expr ** Const(2)).simplify()
        return (adj.expr * (pointwise - self.value.expr)).simplify()

    def propagate_adjoint(self) -> None:
        adj = self._adjoint
        u_expr = dep_value(self.dep).expr
        self.dep._accumulate_adjoint(
            L2Adjoint((Const(2) * adj.expr * u_expr).simplify(), self.measure)
        )


# ---------------------------------------------------------------------------
# 8b. KernelEmbedding
# ---------------------------------------------------------------------------

class KernelEmbedding(Primitive):
    """Symbolic kernel embedding primitive.

    The output is represented as an RKHS-valued symbolic atom.  The backward
    pass uses symbolic RKHS dual-pairing notation rather than expanding kernel
    derivatives numerically.  This keeps the primitive algebraically honest:
    the graph records the correct chain-rule shape, while any kernel-specific
    simplification can be supplied by a downstream simplifier.
    """

    _TABLE1_ROW = "Kernel embedding"

    def __init__(
        self,
        measure: Measure,
        dep: GraphNode,
        given: list[str] | None = None,
        kernel: str = "K",
        derivative_label: str | None = None,
        name: Optional[str] = None,
    ):
        super().__init__(measure, dep, name=name or f"kernel({dep.name})")
        self.dep = dep
        self.given = [] if given is None else list(given)
        self.kernel = kernel
        self.derivative_label = derivative_label or f"D_{dep.name}Kernel[{self.name}]"

    def forward(self) -> L2Value:
        dep_expr = dep_value(self.dep).expr
        free = ", ".join(f"{v}={v.lower()}" for v in self.given) if self.given else "·"
        expr = SymbolicAtom(
            f"∫ {self.kernel}({free}, ·) {dep_expr} d{self.measure.name}_{','.join(self.given) or 'Z'}"
        )
        return L2Value(expr, self.measure, self.given or dep_value(self.dep).domain_vars)

    def eif_contribution(self) -> Expr:
        adj = self._adjoint
        centered_feature = SymbolicAtom(
            f"{self.kernel}({dep_value(self.dep).expr}, .) - {self.value.expr}"
        )
        return SymbolicAtom(f"<{adj.expr}, {centered_feature}>_H")

    def propagate_adjoint(self) -> None:
        adj = self._adjoint
        self.dep._accumulate_adjoint(
            L2Adjoint(
                SymbolicAtom(f"{self.derivative_label}^*({adj.expr})"),
                self.measure,
            )
        )


# ---------------------------------------------------------------------------
# 8c. BoundedAffineMap
# ---------------------------------------------------------------------------

class BoundedAffineMap(Primitive):
    """Bounded affine map u ↦ a*u + b.

    The map preserves the value type of its parent.
    """

    _TABLE1_ROW = "Bounded affine map"

    def __init__(
        self,
        measure: Measure,
        dep: GraphNode,
        scale: Expr | int | float = 1,
        shift: Expr | int | float = 0,
        name: Optional[str] = None,
    ):
        super().__init__(measure, dep, name=name or f"affine({dep.name})")
        self.dep = dep
        self.scale = scale if isinstance(scale, Expr) else Const(scale)
        self.shift = shift if isinstance(shift, Expr) else Const(shift)

    def forward(self) -> NodeValue:
        dep_val = self.dep.value
        out_expr = (self.scale * dep_val.expr + self.shift).simplify()
        if isinstance(dep_val, ScalarValue):
            return ScalarValue(out_expr)
        return L2Value(out_expr, dep_val.measure, dep_val.domain_vars)

    def eif_contribution(self) -> Expr:
        return Const(0)

    def propagate_adjoint(self) -> None:
        adj = self._adjoint
        if isinstance(adj, ScalarAdjoint):
            self.dep._accumulate_adjoint(ScalarAdjoint((adj.expr * self.scale).simplify()))
        else:
            self.dep._accumulate_adjoint(L2Adjoint((adj.expr * self.scale).simplify(), self.measure))


# ---------------------------------------------------------------------------
# 8d. ChangeOfMeasure
# ---------------------------------------------------------------------------

class ChangeOfMeasure(Primitive):
    """Change of measure u ↦ (dλ₂/dλ₁) u."""

    _TABLE1_ROW = "Change of measure"

    def __init__(
        self,
        measure: Measure,
        dep: GraphNode,
        density_ratio: Expr | int | float,
        name: Optional[str] = None,
    ):
        super().__init__(measure, dep, name=name or f"com({dep.name})")
        self.dep = dep
        self.density_ratio = density_ratio if isinstance(density_ratio, Expr) else Const(density_ratio)

    def forward(self) -> L2Value:
        dep_val = dep_value(self.dep)
        return L2Value((self.density_ratio * dep_val.expr).simplify(), dep_val.measure, dep_val.domain_vars)

    def eif_contribution(self) -> Expr:
        return Const(0)

    def propagate_adjoint(self) -> None:
        adj = self._adjoint
        self.dep._accumulate_adjoint(
            L2Adjoint((adj.expr * self.density_ratio).simplify(), self.measure)
        )


# ---------------------------------------------------------------------------
# 8e. OptimalValue / OptimalSolution
# ---------------------------------------------------------------------------

class OptimalValue(Primitive):
    """Symbolic optimal value primitive."""

    _TABLE1_ROW = "Optimal value"

    def __init__(
        self,
        measure: Measure,
        dep: GraphNode,
        objective_label: str = "F",
        arg_label: str = "y",
        derivative_label: str | None = None,
        influence_label: str | None = None,
        name: Optional[str] = None,
    ):
        super().__init__(measure, dep, name=name or f"optval({dep.name})")
        self.dep = dep
        self.objective_label = objective_label
        self.arg_label = arg_label
        self.derivative_label = derivative_label or f"D_{dep.name}OptVal[{self.name}]"
        self.influence_label = influence_label or f"IF_optval[{self.name}]"

    def forward(self) -> ScalarValue:
        dep_expr = getattr(self.dep.value, "expr", SymbolicAtom(self.dep.name))
        return ScalarValue(SymbolicAtom(f"min_{self.arg_label} {self.objective_label}({self.arg_label}; {dep_expr})"))

    def eif_contribution(self) -> Expr:
        adj = self._adjoint
        return (adj.expr * SymbolicAtom(self.influence_label)).simplify()

    def propagate_adjoint(self) -> None:
        adj = self._adjoint
        parent_adj = SymbolicAtom(f"{self.derivative_label}^*({adj.expr})")
        dep_val = self.dep.value
        if isinstance(dep_val, ScalarValue):
            self.dep._accumulate_adjoint(ScalarAdjoint(parent_adj))
        else:
            self.dep._accumulate_adjoint(L2Adjoint(parent_adj, self.measure))


class OptimalSolution(Primitive):
    """Symbolic optimal solution primitive.

    The current engine represents the optimizer symbolically as a scalar-like
    atom so it can participate in display and scalar maps.
    """

    _TABLE1_ROW = "Optimal solution"

    def __init__(
        self,
        measure: Measure,
        dep: GraphNode,
        objective_label: str = "F",
        arg_label: str = "y",
        derivative_label: str | None = None,
        influence_label: str | None = None,
        name: Optional[str] = None,
    ):
        super().__init__(measure, dep, name=name or f"optsol({dep.name})")
        self.dep = dep
        self.objective_label = objective_label
        self.arg_label = arg_label
        self.derivative_label = derivative_label or f"D_{dep.name}OptSol[{self.name}]"
        self.influence_label = influence_label or f"IF_optsol[{self.name}]"

    def forward(self) -> ScalarValue:
        dep_expr = getattr(self.dep.value, "expr", SymbolicAtom(self.dep.name))
        return ScalarValue(SymbolicAtom(f"argmin_{self.arg_label} {self.objective_label}({self.arg_label}; {dep_expr})"))

    def eif_contribution(self) -> Expr:
        adj = self._adjoint
        return (adj.expr * SymbolicAtom(self.influence_label)).simplify()

    def propagate_adjoint(self) -> None:
        adj = self._adjoint
        parent_adj = SymbolicAtom(f"{self.derivative_label}^*({adj.expr})")
        dep_val = self.dep.value
        if isinstance(dep_val, ScalarValue):
            self.dep._accumulate_adjoint(ScalarAdjoint(parent_adj))
        else:
            self.dep._accumulate_adjoint(L2Adjoint(parent_adj, self.measure))


# ---------------------------------------------------------------------------
# 8f. Pathwise-differentiable parameter family
# ---------------------------------------------------------------------------

class PathwiseDiffParameter(Primitive):
    """Generic symbolic pathwise-differentiable parameter primitive."""

    _TABLE1_ROW = "Pathwise diff. parameter"

    def __init__(
        self,
        measure: Measure,
        label: str,
        domain_vars: list[str] | None = None,
        influence_label: str | None = None,
        name: Optional[str] = None,
    ):
        super().__init__(measure, name=name or label)
        self.label = label
        self.domain_vars = [] if domain_vars is None else list(domain_vars)
        self.influence_label = influence_label or f"IF[{self.label}]"

    def forward(self) -> L2Value:
        return L2Value(SymbolicAtom(self.label), self.measure, self.domain_vars)

    def eif_contribution(self) -> Expr:
        adj = self._adjoint
        if isinstance(adj, L2Adjoint):
            return (adj.expr * SymbolicAtom(self.influence_label)).simplify()
        return SymbolicAtom(self.influence_label)

    def propagate_adjoint(self) -> None:
        pass


class RootDensity(PathwiseDiffParameter):
    _TABLE1_ROW = "Root-density"

    def __init__(self, measure: Measure, density_label: str = "p", name: Optional[str] = None):
        super().__init__(measure, label=f"({density_label}_{measure.name})^1/2", domain_vars=measure.variables, name=name)


class ConditionalDensity(PathwiseDiffParameter):
    _TABLE1_ROW = "Conditional density"

    def __init__(
        self,
        measure: Measure,
        dep_vars: list[str],
        given: list[str],
        name: Optional[str] = None,
    ):
        dep = ", ".join(dep_vars)
        giv = ", ".join(given)
        super().__init__(measure, label=f"p_{measure.name}({dep} | {giv})", domain_vars=list(dep_vars) + list(given), name=name)


class DoseResponseFunction(PathwiseDiffParameter):
    _TABLE1_ROW = "Dose-response function"

    def __init__(
        self,
        measure: Measure,
        outcome_var: str,
        treatment_var: str,
        given: list[str],
        name: Optional[str] = None,
    ):
        giv = ", ".join(given)
        super().__init__(
            measure,
            label=f"E_{measure.name}[{outcome_var} | {treatment_var}=a, {giv}]",
            domain_vars=["a"] + list(given),
            name=name,
        )


class CounterfactualDensity(PathwiseDiffParameter):
    _TABLE1_ROW = "Counterfactual density"

    def __init__(
        self,
        measure: Measure,
        outcome_var: str,
        treatment_var: str,
        value: int = 1,
        given: list[str] | None = None,
        name: Optional[str] = None,
    ):
        giv = "" if not given else f", {', '.join(given)}"
        super().__init__(
            measure,
            label=f"y ↦ E_{measure.name}[p(y | {treatment_var}={value}{giv})]",
            domain_vars=[outcome_var],
            name=name,
        )


# ---------------------------------------------------------------------------
# 9. DifferentiableFunction
# ---------------------------------------------------------------------------

class DifferentiableFunction(Primitive):
    """Primitive θ: apply a differentiable function to scalar inputs.

    θ(P, t₁, ..., t_d) = f(t₁, ..., t_d)  ∈ ℝ

    Used in R² as θ₈: f(h₂, h₇) = 1 − h₇/h₂.

    Table 1 row: "Hadamard differentiable map"
      Forward:   f(t₁, ..., t_d)
      EIF contribution to f₀:  0
      Adjoint propagated to tᵢ (scalar):
          f_{tᵢ} += w * (∂f/∂tᵢ)(t₁, ..., t_d)
          where w = f_j is the scalar adjoint of this node.

    Args:
        measure:       The measure P.
        func:          Callable (*scalar_exprs) → Expr for the forward pass.
        partials:      List of callables; partials[i](*exprs) → ∂f/∂(i-th).
        scalar_inputs: The scalar-valued parent nodes t₁, ..., t_d.

    Example (1 − h₇/h₂)::

        psi = DifferentiableFunction(
            P,
            func=lambda h2, h7: Const(1) - h7 / h2,
            partials=[
                lambda h2, h7: h7 / (h2 ** 2),   # ∂/∂h₂ =  h₇/h₂²
                lambda h2, h7: Const(-1) / h2,    # ∂/∂h₇ = −1/h₂
            ],
            scalar_inputs=[var_node, mse_node],
        )
    """

    _TABLE1_ROW = "Hadamard differentiable map"

    def __init__(
        self,
        measure: Measure,
        func: Callable[..., Expr],
        partials: list[Callable[..., Expr]],
        scalar_inputs: list[GraphNode],
        name: Optional[str] = None,
    ):
        if len(partials) != len(scalar_inputs):
            raise ValueError(
                f"DifferentiableFunction: len(partials)={len(partials)} "
                f"must equal len(scalar_inputs)={len(scalar_inputs)}"
            )
        super().__init__(
            measure,
            *scalar_inputs,
            name=name or f"f({', '.join(n.name for n in scalar_inputs)})",
        )
        self.func = func
        self.partials = partials
        self.scalar_inputs = list(scalar_inputs)

    def forward(self) -> ScalarValue:
        input_exprs = [scalar_value(inp).expr for inp in self.scalar_inputs]
        result = self.func(*input_exprs).simplify()
        return ScalarValue(result)

    def eif_contribution(self) -> Expr:
        # Differentiable functions have no direct EIF contribution.
        return Const(0)

    def propagate_adjoint(self) -> None:
        # Table 1 ("Hadamard differentiable map"):
        #   f_{tᵢ} += w * (∂f/∂tᵢ)(t₁, ..., t_d)
        adj = self._adjoint  # ScalarAdjoint: scalar w
        input_exprs = [scalar_value(inp).expr for inp in self.scalar_inputs]
        for inp, partial_fn in zip(self.scalar_inputs, self.partials):
            dfdti = partial_fn(*input_exprs)
            new_expr = (adj.expr * dfdti).simplify()
            inp._accumulate_adjoint(ScalarAdjoint(new_expr))


# ---------------------------------------------------------------------------
# 10. Aliases for Table 1 naming
# ---------------------------------------------------------------------------

HadamardDiffMap = DifferentiableFunction
CoordinateProjection = LiftToDomain
LiftToNewDomain = LiftToDomain


# ---------------------------------------------------------------------------
# 11. FixBinaryArgument
# ---------------------------------------------------------------------------

class FixBinaryArgument(Primitive):
    """Fix a binary argument of an L²(P_{A,X}) function to 0 or 1.

    θ(P, u)(x) = u(a₀, x)   where a₀ ∈ {0, 1}

    Primary use: turn E_P[Y | A=·, X=·] ∈ L²(P_{A,X}) into
    E_P[Y | A=a₀, X=·] ∈ L²(P_X) without the E[AY|X]/π workaround.

    Not in Table 1 of Luedtke (2025) — this is a binary-treatment extension.

    EIF contribution: 0

    Adjoint propagated to dep  (standard IPW formula):
      a₀ = 1 :  (a, x) ↦  a / E_P[A | X=x]         · w(x)
      a₀ = 0 :  (a, x) ↦  (1−a) / (1 − E_P[A|X=x]) · w(x)

    where w(x) is the L² adjoint at the output of this node.

    Args:
        measure:     The measure P (must contain binary_var and free_vars).
        dep:         An L²(P_{A,X})-valued node represented by a conditional
                     expectation. This is usually a ConditionalMean, but may
                     also be a previous FixBinaryArgument whose domain_vars
                     still include binary_var.
        binary_var:  Name of the binary variable to fix (e.g. "A").
        value:       The value to fix it to: 0 or 1.
    """

    _TABLE1_ROW = "FixBinaryArgument (binary-treatment extension)"

    def __init__(
        self,
        measure: Measure,
        dep: GraphNode,
        binary_var: str,
        value: int,
        name: str | None = None,
    ):
        if value not in (0, 1):
            raise ValueError(
                f"FixBinaryArgument: value must be 0 or 1, got {value!r}"
            )
        if binary_var not in measure.variables:
            raise ValueError(
                f"FixBinaryArgument: {binary_var!r} not in measure variables "
                f"{measure.variables}"
            )
        super().__init__(
            measure, dep,
            name=name or f"fix({dep.name}, {binary_var}={value})",
        )
        self.dep = dep
        self.binary_var = binary_var
        self.fixed_value = value

    def forward(self) -> L2Value:
        dep_val = dep_value(self.dep)
        dep_expr = dep_val.expr

        # Build the output expression: u(a₀, x)
        # If dep is a ConditionalMean, its expr is a CondExpect — we can
        # directly move binary_var from 'given' into 'fixed_vals'.
        if isinstance(dep_expr, CondExpect):
            new_given  = [v for v in dep_expr.given if v != self.binary_var]
            new_fixed  = dict(dep_expr.fixed_vals)
            new_fixed[self.binary_var] = self.fixed_value
            out_expr = CondExpect(
                dep_expr.measure_name, dep_expr.dep,
                new_given, fixed_vals=new_fixed,
            )
        else:
            raise NotImplementedError(
                f"FixBinaryArgument.forward(): dep expression is "
                f"{type(dep_expr).__name__!r}, expected CondExpect. "
                f"Only ConditionalMean nodes are supported as dep."
            )

        new_domain = [v for v in dep_val.domain_vars if v != self.binary_var]
        return L2Value(out_expr, self.measure, new_domain)

    def eif_contribution(self) -> Expr:
        # Fixing a binary argument introduces no direct EIF term —
        # the contribution flows entirely through the dep's adjoint.
        return Const(0)

    def propagate_adjoint(self) -> None:
        # Standard IPW adjoint (Table 1 extension):
        #
        #   a₀ = 1:  f_dep += (a,x) ↦  a       / π(x)      · w(x)
        #   a₀ = 0:  f_dep += (a,x) ↦  (1−a)   / (1−π(x))  · w(x)
        #
        # where π(x) = E_P[A | X=x] and w(x) is self._adjoint.expr.
        adj = self._adjoint  # L2Adjoint: w(x)

        # Identify free variables (everything except the binary one)
        dep_val  = dep_value(self.dep)
        free_vars = [v for v in dep_val.domain_vars if v != self.binary_var]

        a_expr  = ObsVar(self.binary_var)
        pi_expr = CondExpect(self.measure.name, a_expr, free_vars)

        if self.fixed_value == 1:
            weight = a_expr / pi_expr
        else:  # value == 0
            weight = (Const(1) - a_expr) / (Const(1) - pi_expr)

        adj_expr = (weight * adj.expr).simplify()
        self.dep._accumulate_adjoint(L2Adjoint(adj_expr, self.measure))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def dep_value(node: GraphNode) -> L2Value:
    """Extract the L2Value from a node; raise clearly if wrong type."""
    val = node.value
    if not isinstance(val, L2Value):
        raise TypeError(
            f"Node {node.name!r} was expected to be L²-valued, "
            f"but has value type {type(val).__name__}"
        )
    return val


def scalar_value(node: GraphNode) -> ScalarValue:
    """Extract the ScalarValue from a node; raise clearly if wrong type."""
    val = node.value
    if not isinstance(val, ScalarValue):
        raise TypeError(
            f"Node {node.name!r} was expected to be scalar-valued, "
            f"but has value type {type(val).__name__}"
        )
    return val
