"""
verify.py — Numerical verification of computation graph decompositions.

Verifies that the symbolic ψ(P) produced by a computation graph matches
the intended target parameter, evaluated on a concrete discrete distribution
where all expectations are exact finite sums.

Usage::

    from pydimple.symbolic.verify import DiscretePopulation, verify_graph

    # 1. Define a discrete joint distribution
    pop = DiscretePopulation(
        atoms=[
            {"Y": 3.0, "X": 0},
            {"Y": 1.0, "X": 0},
            {"Y": 5.0, "X": 1},
            {"Y": 7.0, "X": 1},
        ],
        weights=[0.2, 0.3, 0.1, 0.4],
    )

    # 2. Define your target parameter as a function of the population
    def my_target(pop):
        var_y = pop.var("Y")
        # E[(Y - E[Y|X])^2]
        resid_var = pop.expect(
            lambda z: (z["Y"] - pop.cond_expect("Y", {"X": z["X"]})) ** 2
        )
        return 1 - resid_var / var_y

    # 3. Verify the graph computes the same thing
    result = verify_graph("r2_graph.json", my_target, pop)
    # result = {"graph_psi": 0.123, "target_psi": 0.123, "match": True, ...}
"""

from __future__ import annotations

import json
from typing import Callable, Union

from .expr import (
    Expr, Const, ObsVar, CondExpect, MargExpect, MargVar, SymbolicAtom,
    Add, Sub, Mul, Div, Neg, Pow, Sum,
)


# ---------------------------------------------------------------------------
# DiscretePopulation
# ---------------------------------------------------------------------------

class DiscretePopulation:
    """A finite discrete joint distribution for exact numerical evaluation.

    Each atom is a dict mapping variable names to values.
    Weights default to uniform if not provided.

    Args:
        atoms:   List of observations, each a ``{var_name: value}`` dict.
        weights: Probability weights (need not sum to 1; will be normalised).
    """

    def __init__(
        self,
        atoms: list[dict[str, float]],
        weights: list[float] | None = None,
    ):
        if not atoms:
            raise ValueError("Population must have at least one atom")
        self.atoms = atoms
        if weights is None:
            n = len(atoms)
            self.weights = [1.0 / n] * n
        else:
            if len(weights) != len(atoms):
                raise ValueError(
                    f"weights length ({len(weights)}) != atoms length ({len(atoms)})"
                )
            total = sum(weights)
            self.weights = [w / total for w in weights]

    @property
    def variables(self) -> list[str]:
        """Variable names present in the first atom."""
        return list(self.atoms[0].keys())

    # ---- convenience methods for writing target functions --------------------

    def expect(self, fn: Callable[[dict], float]) -> float:
        """E_P[fn(Z)] = Σ_i w_i · fn(z_i)."""
        return sum(w * fn(z) for w, z in zip(self.weights, self.atoms))

    def var(self, var_name: str) -> float:
        """Var_P[var_name] = E[X²] - E[X]²."""
        mean = self.expect(lambda z: z[var_name])
        return self.expect(lambda z: (z[var_name] - mean) ** 2)

    def cond_expect(
        self,
        var_or_fn: Union[str, Callable[[dict], float]],
        given: dict[str, float],
    ) -> float:
        """E_P[var_or_fn | given₁=v₁, ...].

        Filters atoms to those matching ``given`` exactly, then averages.

        Args:
            var_or_fn: Variable name or callable(atom) -> float.
            given:     Dict of {var_name: value} to condition on.
        """
        fn = (lambda z: z[var_or_fn]) if isinstance(var_or_fn, str) else var_or_fn
        matching_vals = []
        matching_wts = []
        for w, z in zip(self.weights, self.atoms):
            if all(z[k] == v for k, v in given.items()):
                matching_vals.append(fn(z))
                matching_wts.append(w)
        if not matching_wts:
            raise ValueError(
                f"No atoms match conditioning set {given}. "
                f"Check that the population contains these values."
            )
        total = sum(matching_wts)
        return sum(v * w / total for v, w in zip(matching_vals, matching_wts))


# ---------------------------------------------------------------------------
# Expression evaluator
# ---------------------------------------------------------------------------

def eval_expr(
    expr: Expr,
    pop: DiscretePopulation,
    obs: dict[str, float] | None = None,
) -> float:
    """Recursively evaluate a symbolic Expr on a discrete population.

    Args:
        expr: Symbolic expression to evaluate.
        pop:  The discrete population P.
        obs:  Current observation (for L² context). ``None`` for scalar context.

    Returns:
        The numerical value of the expression.

    Two evaluation modes:
      - **Scalar context** (obs=None): for ψ(P)-type expressions.
        ObsVar will raise; MargExpect/MargVar compute by iterating atoms.
      - **L² context** (obs given): for per-observation expressions.
        ObsVar returns obs[name]; CondExpect conditions on obs values.
    """
    # -- Const ---------------------------------------------------------------
    if isinstance(expr, Const):
        return expr.value

    # -- ObsVar --------------------------------------------------------------
    if isinstance(expr, ObsVar):
        if obs is None:
            raise ValueError(
                f"ObsVar({expr.name!r}) encountered in scalar context "
                f"(no current observation). This likely means the expression "
                f"is L² but was evaluated without an observation."
            )
        return obs[expr.name]

    # -- MargExpect: E_P[dep] ------------------------------------------------
    if isinstance(expr, MargExpect):
        return sum(
            w * eval_expr(expr.dep, pop, obs=z)
            for w, z in zip(pop.weights, pop.atoms)
        )

    # -- MargVar: Var_P[dep] -------------------------------------------------
    if isinstance(expr, MargVar):
        mean = sum(
            w * eval_expr(expr.dep, pop, obs=z)
            for w, z in zip(pop.weights, pop.atoms)
        )
        return sum(
            w * (eval_expr(expr.dep, pop, obs=z) - mean) ** 2
            for w, z in zip(pop.weights, pop.atoms)
        )

    # -- CondExpect: E_P[dep | given=obs_vals, fixed_vals] -------------------
    if isinstance(expr, CondExpect):
        # Build the conditioning set from current obs (free vars) + fixed vals
        cond = {}
        for var in expr.given:
            if obs is None:
                raise ValueError(
                    f"CondExpect conditioning on {var!r} but no current "
                    f"observation is available (scalar context)."
                )
            cond[var] = obs[var]
        for var, val in expr.fixed_vals.items():
            cond[var] = val

        # Filter and average
        matching_vals = []
        matching_wts = []
        for w, z in zip(pop.weights, pop.atoms):
            if all(z[k] == v for k, v in cond.items()):
                matching_vals.append(eval_expr(expr.dep, pop, obs=z))
                matching_wts.append(w)
        if not matching_wts:
            raise ValueError(
                f"No atoms match CondExpect conditioning {cond}. "
                f"Expression: {expr!r}"
            )
        total = sum(matching_wts)
        return sum(v * w / total for v, w in zip(matching_vals, matching_wts))

    # -- SymbolicAtom --------------------------------------------------------
    if isinstance(expr, SymbolicAtom):
        raise ValueError(
            f"Cannot numerically evaluate SymbolicAtom({expr.text!r}). "
            f"Numerical verification is only supported for expressions "
            f"built from ObsVar, CondExpect, MargExpect, MargVar, and Const."
        )

    # -- Add -----------------------------------------------------------------
    if isinstance(expr, Add):
        return eval_expr(expr.a, pop, obs) + eval_expr(expr.b, pop, obs)

    # -- Sub -----------------------------------------------------------------
    if isinstance(expr, Sub):
        return eval_expr(expr.a, pop, obs) - eval_expr(expr.b, pop, obs)

    # -- Mul -----------------------------------------------------------------
    if isinstance(expr, Mul):
        return eval_expr(expr.a, pop, obs) * eval_expr(expr.b, pop, obs)

    # -- Div -----------------------------------------------------------------
    if isinstance(expr, Div):
        return eval_expr(expr.a, pop, obs) / eval_expr(expr.b, pop, obs)

    # -- Neg -----------------------------------------------------------------
    if isinstance(expr, Neg):
        return -eval_expr(expr.a, pop, obs)

    # -- Pow -----------------------------------------------------------------
    if isinstance(expr, Pow):
        return eval_expr(expr.base, pop, obs) ** eval_expr(expr.exp, pop, obs)

    # -- Sum -----------------------------------------------------------------
    if isinstance(expr, Sum):
        return sum(eval_expr(t, pop, obs) for t in expr.terms)

    raise TypeError(f"eval_expr: unsupported expression type {type(expr).__name__}")


# ---------------------------------------------------------------------------
# Graph verification
# ---------------------------------------------------------------------------

def verify_graph(
    graph_json: Union[str, dict],
    target_fn: Callable[[DiscretePopulation], float],
    pop: DiscretePopulation,
    *,
    atol: float = 1e-10,
    check_eif_mean: bool = True,
) -> dict:
    """Verify a computation graph produces the correct ψ(P).

    Args:
        graph_json:     JSON string or parsed dict of the computation graph.
        target_fn:      ``target_fn(pop) -> float`` computes the target
                        parameter ψ(P) directly from the population.
        pop:            Discrete population to evaluate on.
        atol:           Absolute tolerance for numerical comparison.
        check_eif_mean: Also verify E_P[φ(Z; P)] ≈ 0 (necessary condition).

    Returns:
        A dict with keys:
          - ``graph_psi``:    ψ(P) from the computation graph's symbolic expression
          - ``target_psi``:   ψ(P) from the direct target function
          - ``psi_match``:    bool, whether they agree within atol
          - ``psi_error``:    absolute difference
          - ``eif_mean``:     E_P[φ(Z; P)] (should be ≈ 0), or None
          - ``eif_mean_ok``:  bool, whether |E[φ]| < atol, or None
    """
    # Avoid circular import — deserializer and graph depend on same package
    from .deserializer import deserialize
    from .graph import compute_eif

    if isinstance(graph_json, str):
        graph_json = json.loads(graph_json)

    # Build graph and compute EIF (runs forward + backward)
    output_node = deserialize(graph_json)
    eif_expr = compute_eif(output_node)
    psi_expr = output_node._value.expr

    # Evaluate symbolic ψ(P) on the population
    graph_psi = eval_expr(psi_expr, pop)

    # Evaluate direct target
    target_psi = target_fn(pop)

    psi_error = abs(graph_psi - target_psi)
    psi_match = psi_error < atol

    result = {
        "graph_psi": graph_psi,
        "target_psi": target_psi,
        "psi_match": psi_match,
        "psi_error": psi_error,
        "eif_mean": None,
        "eif_mean_ok": None,
    }

    # Check E_P[φ(Z; P)] = 0
    if check_eif_mean:
        eif_mean = sum(
            w * eval_expr(eif_expr, pop, obs=z)
            for w, z in zip(pop.weights, pop.atoms)
        )
        result["eif_mean"] = eif_mean
        result["eif_mean_ok"] = abs(eif_mean) < atol

    return result
