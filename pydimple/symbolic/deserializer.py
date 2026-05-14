"""
deserializer.py — JSON computation graph → GraphNode DAG.

Converts the JSON produced by the LLM prompt into a live computation
graph ready for ``compute_eif()``.

Supported JSON schema
---------------------
{
  "expressible": true,        # optional; raises ValueError if false
  "variables":  ["Y", "X"],  # optional; inferred from RandomVariable nodes
  "nodes": {                  # dict  OR  list-with-"id"-fields
    "n1": {"type": "RandomVariable", "var_name": "Y"},
    ...
  },
  "output_node": "n5"
}

Expression strings (func / partials)
-------------------------------------
Evaluated as Python expressions.  Available names:
  h0, h1, h2, ...   — positional references to inputs (Expr objects)
  Const(n)           — numeric constant
  +  -  *  /  **     — arithmetic (works through Expr operator overloading)

Examples:
  "h0 - h1"                     → subtraction of two L2 expressions
  "Const(2) * (h0 - h1)"        → scaled difference
  "h0 ** 2"   or  "h0 ** Const(2)"   → squaring (both accepted)
"""

from __future__ import annotations

import json
from typing import Callable, Union

from .expr import Const, SymbolicAtom, Expr
from .core import Measure
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
# Internal helpers
# ---------------------------------------------------------------------------

def _compile(expr_str: str, n_inputs: int) -> Callable:
    """Compile a func/partial string into a callable (*Expr) -> Expr.

    The string is evaluated as a Python lambda with positional arguments
    h0, h1, ..., h{n-1}.  ``Const`` is available in the expression
    namespace; all arithmetic operators work via Expr overloading.
    """
    arg_names = [f"h{i}" for i in range(n_inputs)]
    code = f"lambda {', '.join(arg_names)}: {expr_str}"
    try:
        return eval(code, {"Const": Const})  # noqa: S307 — trusted LLM output
    except SyntaxError as exc:
        raise ValueError(
            f"Cannot compile expression {expr_str!r} "
            f"(n_inputs={n_inputs}): {exc}"
        ) from exc


def _compile_scalar_expr(expr_str: str) -> Expr:
    """Compile a no-input symbolic expression string into an Expr."""
    try:
        return eval(expr_str, {"Const": Const, "SymbolicAtom": SymbolicAtom})  # noqa: S307
    except Exception as exc:
        raise ValueError(f"Cannot compile scalar expression {expr_str!r}: {exc}") from exc


def _normalize_nodes(raw: Union[dict, list]) -> dict[str, dict]:
    """Accept both ``{"n1": {...}, ...}`` and ``[{"id": "n1", ...}, ...]``."""
    if isinstance(raw, dict):
        return raw
    result: dict[str, dict] = {}
    for entry in raw:
        node_id = entry.get("id")
        if node_id is None:
            raise ValueError(
                f"Node in list format is missing an 'id' field: {entry}"
            )
        result[node_id] = {k: v for k, v in entry.items() if k != "id"}
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def deserialize(
    data: Union[str, dict],
    measure_name: str = "P",
) -> GraphNode:
    """Deserialize a JSON computation graph into a GraphNode DAG.

    Args:
        data:         JSON string or already-parsed dict.
        measure_name: Name for the probability measure (default ``"P"``).

    Returns:
        The output ``GraphNode`` (scalar-valued).  Pass it directly to
        :func:`~pydimple.symbolic.compute_eif` or
        :meth:`~pydimple.symbolic.ComputationGraph.eif`.

    Raises:
        ValueError: if ``"expressible"`` is ``false``, a node type is
                    unknown, or a func/partial string fails to compile.
        KeyError:   if a node references a parent id that has not yet
                    been defined (i.e. topological order violated).

    Example::

        node = deserialize(json_string)
        eif  = compute_eif(node)
        print(eif)
    """
    if isinstance(data, str):
        data = json.loads(data)

    # ── check expressibility ──────────────────────────────────────────────────
    if not data.get("expressible", True):
        reason = data.get("reason", "(no reason given)")
        raise ValueError(f"Parameter declared not expressible: {reason}")

    # ── normalise node map ────────────────────────────────────────────────────
    raw_nodes = _normalize_nodes(data["nodes"])
    output_id: str = data["output_node"]

    # ── build measure ─────────────────────────────────────────────────────────
    if "variables" in data:
        variables = list(data["variables"])
    else:
        # infer from all RandomVariable entries
        variables = [
            spec["var_name"]
            for spec in raw_nodes.values()
            if spec.get("type") == "RandomVariable"
        ]
        if not variables:
            raise ValueError(
                "JSON has no 'variables' key and no RandomVariable nodes; "
                "cannot infer the measure domain."
            )

    measure = Measure(measure_name, variables)

    # ── build nodes in declaration order (must be topological) ───────────────
    nodes: dict[str, GraphNode] = {}

    def ref(node_id: str) -> GraphNode:
        try:
            return nodes[node_id]
        except KeyError:
            raise KeyError(
                f"Node {node_id!r} is referenced before it is defined. "
                f"Nodes must appear in topological order (parents first)."
            ) from None

    for node_id, spec in raw_nodes.items():
        node_type = spec.get("type")

        if node_type == "RandomVariable":
            node = RandomVariable(measure, spec["var_name"], name=node_id)

        elif node_type == "ConstantMap":
            node = ConstantMap(measure, ref(spec["dep"]), name=node_id)

        elif node_type == "Variance":
            node = Variance(measure, ref(spec["dep"]), name=node_id)

        elif node_type == "ConditionalMean":
            node = ConditionalMean(
                measure,
                ref(spec["dep"]),
                given=list(spec["given"]),
                name=node_id,
            )

        elif node_type == "RFoldConditionalMean":
            node = RFoldConditionalMean(
                measure,
                ref(spec["dep"]),
                given=list(spec["given"]),
                name=node_id,
            )

        elif node_type == "ConditionalVariance":
            node = ConditionalVariance(
                measure,
                ref(spec["dep"]),
                given=list(spec["given"]),
                name=node_id,
            )

        elif node_type == "ConditionalCovariance":
            node = ConditionalCovariance(
                measure,
                ref(spec["left"]),
                ref(spec["right"]),
                given=list(spec["given"]),
                name=node_id,
            )

        elif node_type == "LiftToDomain":
            node = LiftToDomain(measure, ref(spec["l2x_node"]), name=node_id)

        elif node_type == "PointwiseOperation":
            inputs = [ref(i) for i in spec["l2_inputs"]]
            n = len(inputs)
            func     = _compile(spec["func"], n)
            partials = [_compile(p, n) for p in spec["partials"]]
            node = PointwiseOperation(
                measure, func, partials, inputs, name=node_id
            )

        elif node_type == "MarginalMean":
            node = MarginalMean(measure, ref(spec["dep"]), name=node_id)

        elif node_type == "InnerProduct":
            node = InnerProduct(
                measure,
                ref(spec["left"]),
                ref(spec["right"]),
                name=node_id,
            )

        elif node_type == "SquaredNorm":
            node = SquaredNorm(measure, ref(spec["dep"]), name=node_id)

        elif node_type == "KernelEmbedding":
            node = KernelEmbedding(
                measure,
                ref(spec["dep"]),
                given=list(spec.get("given", [])),
                kernel=spec.get("kernel", "K"),
                derivative_label=spec.get("derivative_label"),
                name=node_id,
            )

        elif node_type == "BoundedAffineMap":
            scale = _compile_scalar_expr(spec.get("scale", "Const(1)"))
            shift = _compile_scalar_expr(spec.get("shift", "Const(0)"))
            node = BoundedAffineMap(
                measure,
                ref(spec["dep"]),
                scale=scale,
                shift=shift,
                name=node_id,
            )

        elif node_type == "ChangeOfMeasure":
            density_ratio = _compile_scalar_expr(spec["density_ratio"])
            node = ChangeOfMeasure(
                measure,
                ref(spec["dep"]),
                density_ratio=density_ratio,
                name=node_id,
            )

        elif node_type == "OptimalValue":
            node = OptimalValue(
                measure,
                ref(spec["dep"]),
                objective_label=spec.get("objective_label", "F"),
                arg_label=spec.get("arg_label", "y"),
                derivative_label=spec.get("derivative_label"),
                influence_label=spec.get("influence_label"),
                name=node_id,
            )

        elif node_type == "OptimalSolution":
            node = OptimalSolution(
                measure,
                ref(spec["dep"]),
                objective_label=spec.get("objective_label", "F"),
                arg_label=spec.get("arg_label", "y"),
                derivative_label=spec.get("derivative_label"),
                influence_label=spec.get("influence_label"),
                name=node_id,
            )

        elif node_type == "PathwiseDiffParameter":
            node = PathwiseDiffParameter(
                measure,
                label=spec["label"],
                domain_vars=list(spec.get("domain_vars", [])),
                influence_label=spec.get("influence_label"),
                name=node_id,
            )

        elif node_type == "RootDensity":
            node = RootDensity(
                measure,
                density_label=spec.get("density_label", "p"),
                name=node_id,
            )

        elif node_type == "ConditionalDensity":
            node = ConditionalDensity(
                measure,
                dep_vars=list(spec["dep_vars"]),
                given=list(spec["given"]),
                name=node_id,
            )

        elif node_type == "DoseResponseFunction":
            node = DoseResponseFunction(
                measure,
                outcome_var=spec["outcome_var"],
                treatment_var=spec["treatment_var"],
                given=list(spec["given"]),
                name=node_id,
            )

        elif node_type == "CounterfactualDensity":
            node = CounterfactualDensity(
                measure,
                outcome_var=spec["outcome_var"],
                treatment_var=spec["treatment_var"],
                value=spec.get("value", 1),
                given=list(spec.get("given", [])),
                name=node_id,
            )

        elif node_type == "DifferentiableFunction":
            inputs = [ref(i) for i in spec["scalar_inputs"]]
            n = len(inputs)
            func     = _compile(spec["func"], n)
            partials = [_compile(p, n) for p in spec["partials"]]
            node = DifferentiableFunction(
                measure, func, partials, inputs, name=node_id
            )

        elif node_type == "FixBinaryArgument":
            node = FixBinaryArgument(
                measure,
                ref(spec["dep"]),
                binary_var=spec["binary_var"],
                value=spec["value"],
                name=node_id,
            )

        elif node_type == "HadamardDiffMap":
            inputs = [ref(i) for i in spec["scalar_inputs"]]
            n = len(inputs)
            func = _compile(spec["func"], n)
            partials = [_compile(p, n) for p in spec["partials"]]
            node = DifferentiableFunction(
                measure, func, partials, inputs, name=node_id
            )

        elif node_type in ("CoordinateProjection", "LiftToNewDomain"):
            node = LiftToDomain(measure, ref(spec["l2x_node"]), name=node_id)

        else:
            valid = (
                "RandomVariable, ConstantMap, Variance, ConditionalMean, "
                "RFoldConditionalMean, ConditionalVariance, ConditionalCovariance, "
                "LiftToDomain, PointwiseOperation, MarginalMean, InnerProduct, "
                "SquaredNorm, KernelEmbedding, BoundedAffineMap, ChangeOfMeasure, "
                "OptimalValue, OptimalSolution, PathwiseDiffParameter, RootDensity, "
                "ConditionalDensity, DoseResponseFunction, CounterfactualDensity, "
                "DifferentiableFunction, HadamardDiffMap, CoordinateProjection, "
                "LiftToNewDomain, FixBinaryArgument"
            )
            raise ValueError(
                f"Unknown node type {node_type!r} at {node_id!r}. "
                f"Valid types: {valid}."
            )

        nodes[node_id] = node

    if output_id not in nodes:
        raise KeyError(
            f"output_node {output_id!r} is not defined in 'nodes'."
        )

    return nodes[output_id]
