"""
validator.py — Structural validation for JSON computation graphs.

Two entry points:

    validate_graph(data)  → list[str]   # errors in the JSON graph
    validate_eif(eif)     → list[str]   # sanity checks on the computed EIF

Both return a list of human-readable error strings.  Empty list = all good.
"""

from __future__ import annotations

from typing import Union

from .expr import (
    Expr, Const, ObsVar, CondExpect, MargExpect, MargVar,
    Add, Sub, Mul, Div, Neg, Pow, Sum,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_L2_TYPES = {
    "RandomVariable", "ConstantMap", "ConditionalMean",
    "RFoldConditionalMean", "ConditionalVariance", "ConditionalCovariance",
    "LiftToDomain", "PointwiseOperation", "KernelEmbedding",
    "BoundedAffineMap", "ChangeOfMeasure", "PathwiseDiffParameter",
    "RootDensity", "ConditionalDensity", "DoseResponseFunction",
    "CounterfactualDensity",
}
_SCALAR_TYPES = {
    "Variance", "MarginalMean", "InnerProduct", "SquaredNorm",
    "DifferentiableFunction", "OptimalValue", "OptimalSolution",
}
_ALL_TYPES = _L2_TYPES | _SCALAR_TYPES | {"FixBinaryArgument"}

# Domain sentinel values
_FULL = "L2(P)"          # full-domain
_SCALAR = "Scalar"


# ---------------------------------------------------------------------------
# Graph validation (pre-computation)
# ---------------------------------------------------------------------------

def validate_graph(data: Union[str, dict]) -> list[str]:
    """Validate the structural correctness of a JSON computation graph.

    Checks performed:
      1. All referenced node ids exist and are defined before use.
      2. Node types are known.
      3. RandomVariable only appears as dep of ConstantMap.
      4. Domain consistency for PointwiseOperation inputs.
      5. MarginalMean receives a full-domain L2(P) node.
      6. LiftToDomain receives a marginal-domain node.
      7. DifferentiableFunction inputs are all Scalar.
      8. Variance / SquaredNorm dep is L2.
      9. Partials count matches input count.
     10. output_node exists and is Scalar.
     11. All numeric literals in func/partials use Const(n).

    Returns:
        List of error strings.  Empty list means the graph is valid.
    """
    import json as _json

    if isinstance(data, str):
        try:
            data = _json.loads(data)
        except _json.JSONDecodeError as exc:
            return [f"Invalid JSON: {exc}"]

    errors: list[str] = []

    if not data.get("expressible", True):
        return []  # nothing to validate

    # ── normalize nodes ──────────────────────────────────────────────────────
    raw_nodes = data.get("nodes")
    if raw_nodes is None:
        return ["Missing 'nodes' key in JSON."]

    if isinstance(raw_nodes, list):
        nodes = {}
        for entry in raw_nodes:
            nid = entry.get("id")
            if nid is None:
                errors.append(f"Node missing 'id' field: {entry}")
                continue
            nodes[nid] = {k: v for k, v in entry.items() if k != "id"}
    elif isinstance(raw_nodes, dict):
        nodes = raw_nodes
    else:
        return [f"'nodes' must be a dict or list, got {type(raw_nodes).__name__}"]

    output_id = data.get("output_node")
    if output_id is None:
        errors.append("Missing 'output_node' key.")

    variables = set(data.get("variables", []))

    # ── track domains and output types ───────────────────────────────────────
    # domain: "L2(P)", "L2(P_X)", "L2(P_{A,X})", or "Scalar"
    domain_of: dict[str, str] = {}
    type_of: dict[str, str] = {}    # "l2" or "scalar"
    seen: set[str] = set()
    rv_nodes: set[str] = set()      # nodes that are RandomVariable

    def _ref_check(nid: str, parent_id: str, field: str) -> bool:
        """Check that nid is defined before parent_id."""
        if nid not in seen:
            errors.append(
                f"[{parent_id}] references {field}={nid!r} which is "
                f"not yet defined (topological order violation)."
            )
            return False
        return True

    def _marginal_domain(given: list[str]) -> str:
        return f"L2(P_{{{','.join(sorted(given))}}})"

    def _vars_from_domain(domain: str) -> list[str] | None:
        """Recover marginal-domain variables from the validator's domain tag."""
        if domain == _FULL:
            return sorted(variables)
        if domain.startswith("L2(P_{") and domain.endswith("})"):
            inner = domain[len("L2(P_{"):-2]
            return [] if not inner else inner.split(",")
        return None

    for nid, spec in nodes.items():
        ntype = spec.get("type", "(missing)")
        seen.add(nid)

        if ntype not in _ALL_TYPES:
            errors.append(f"[{nid}] Unknown type {ntype!r}.")
            type_of[nid] = "unknown"
            domain_of[nid] = "unknown"
            continue

        # ── RandomVariable ───────────────────────────────────────────────────
        if ntype == "RandomVariable":
            vn = spec.get("var_name")
            if vn and variables and vn not in variables:
                errors.append(
                    f"[{nid}] var_name={vn!r} not in declared variables {sorted(variables)}."
                )
            rv_nodes.add(nid)
            type_of[nid] = "l2"
            domain_of[nid] = _FULL

        # ── ConstantMap ──────────────────────────────────────────────────────
        elif ntype == "ConstantMap":
            dep = spec.get("dep")
            if dep and _ref_check(dep, nid, "dep"):
                if dep not in rv_nodes:
                    errors.append(
                        f"[{nid}] ConstantMap dep={dep!r} is not a "
                        f"RandomVariable (it is {nodes.get(dep, {}).get('type', '?')})."
                    )
            type_of[nid] = "l2"
            domain_of[nid] = _FULL

        # ── Variance ─────────────────────────────────────────────────────────
        elif ntype == "Variance":
            dep = spec.get("dep")
            if dep and _ref_check(dep, nid, "dep"):
                if type_of.get(dep) != "l2":
                    errors.append(f"[{nid}] Variance dep={dep!r} is not L2-valued.")
            type_of[nid] = "scalar"
            domain_of[nid] = _SCALAR

        # ── ConditionalMean ──────────────────────────────────────────────────
        elif ntype in {"ConditionalMean", "RFoldConditionalMean"}:
            dep = spec.get("dep")
            given = spec.get("given", [])
            if dep and _ref_check(dep, nid, "dep"):
                if type_of.get(dep) != "l2":
                    errors.append(f"[{nid}] {ntype} dep={dep!r} is not L2-valued.")
                if dep in rv_nodes:
                    errors.append(
                        f"[{nid}] {ntype} dep={dep!r} is a raw "
                        f"RandomVariable — wrap it in ConstantMap first."
                    )
            if not given:
                errors.append(f"[{nid}] {ntype} missing 'given' list.")
            type_of[nid] = "l2"
            domain_of[nid] = _marginal_domain(given)

        # ── ConditionalVariance ─────────────────────────────────────────────
        elif ntype == "ConditionalVariance":
            dep = spec.get("dep")
            given = spec.get("given", [])
            if dep and _ref_check(dep, nid, "dep"):
                if type_of.get(dep) != "l2":
                    errors.append(f"[{nid}] ConditionalVariance dep={dep!r} is not L2-valued.")
            if not given:
                errors.append(f"[{nid}] ConditionalVariance missing 'given' list.")
            type_of[nid] = "l2"
            domain_of[nid] = _marginal_domain(given)

        # ── ConditionalCovariance ───────────────────────────────────────────
        elif ntype == "ConditionalCovariance":
            given = spec.get("given", [])
            for field in ("left", "right"):
                dep = spec.get(field)
                if dep and _ref_check(dep, nid, field):
                    if type_of.get(dep) != "l2":
                        errors.append(f"[{nid}] ConditionalCovariance {field}={dep!r} is not L2-valued.")
            if not given:
                errors.append(f"[{nid}] ConditionalCovariance missing 'given' list.")
            type_of[nid] = "l2"
            domain_of[nid] = _marginal_domain(given)

        # ── LiftToDomain ────────────────────────────────────────────────────
        elif ntype == "LiftToDomain":
            src = spec.get("l2x_node")
            if src and _ref_check(src, nid, "l2x_node"):
                if type_of.get(src) != "l2":
                    errors.append(f"[{nid}] LiftToDomain l2x_node={src!r} is not L2-valued.")
                if domain_of.get(src) == _FULL:
                    errors.append(
                        f"[{nid}] LiftToDomain l2x_node={src!r} is already "
                        f"full-domain L2(P) — lift is unnecessary."
                    )
            type_of[nid] = "l2"
            domain_of[nid] = _FULL

        # ── PointwiseOperation ───────────────────────────────────────────────
        elif ntype == "PointwiseOperation":
            inputs = spec.get("l2_inputs", [])
            partials = spec.get("partials", [])
            if len(partials) != len(inputs):
                errors.append(
                    f"[{nid}] len(partials)={len(partials)} != "
                    f"len(l2_inputs)={len(inputs)}."
                )
            domains_seen = set()
            for inp in inputs:
                if _ref_check(inp, nid, "l2_inputs"):
                    if inp in rv_nodes:
                        errors.append(
                            f"[{nid}] l2_inputs contains raw RandomVariable "
                            f"{inp!r} — wrap in ConstantMap first."
                        )
                    if type_of.get(inp) != "l2":
                        errors.append(
                            f"[{nid}] l2_inputs contains {inp!r} which is "
                            f"not L2-valued (type={type_of.get(inp, '?')})."
                        )
                    domains_seen.add(domain_of.get(inp, "?"))
            if len(domains_seen) > 1:
                errors.append(
                    f"[{nid}] PointwiseOperation inputs have mixed domains: "
                    f"{domains_seen}. All must share the same domain."
                )
            # Validate func/partials for bare integers
            _check_bare_ints(spec.get("func", ""), nid, "func", errors)
            for i, p in enumerate(partials):
                _check_bare_ints(p, nid, f"partials[{i}]", errors)

            type_of[nid] = "l2"
            domain_of[nid] = domains_seen.pop() if len(domains_seen) == 1 else _FULL

        # ── MarginalMean ─────────────────────────────────────────────────────
        elif ntype == "MarginalMean":
            dep = spec.get("dep")
            if dep and _ref_check(dep, nid, "dep"):
                if type_of.get(dep) != "l2":
                    errors.append(f"[{nid}] MarginalMean dep={dep!r} is not L2-valued.")
                if domain_of.get(dep) != _FULL:
                    errors.append(
                        f"[{nid}] MarginalMean dep={dep!r} is in "
                        f"{domain_of.get(dep, '?')}, not L2(P). "
                        f"Wrap in LiftToDomain first."
                    )
            type_of[nid] = "scalar"
            domain_of[nid] = _SCALAR

        # ── InnerProduct ─────────────────────────────────────────────────────
        elif ntype == "InnerProduct":
            for field in ("left", "right"):
                dep = spec.get(field)
                if dep and _ref_check(dep, nid, field):
                    if type_of.get(dep) != "l2":
                        errors.append(f"[{nid}] InnerProduct {field}={dep!r} is not L2-valued.")
            type_of[nid] = "scalar"
            domain_of[nid] = _SCALAR

        # ── SquaredNorm ──────────────────────────────────────────────────────
        elif ntype == "SquaredNorm":
            dep = spec.get("dep")
            if dep and _ref_check(dep, nid, "dep"):
                if type_of.get(dep) != "l2":
                    errors.append(f"[{nid}] SquaredNorm dep={dep!r} is not L2-valued.")
            type_of[nid] = "scalar"
            domain_of[nid] = _SCALAR

        # ── KernelEmbedding ─────────────────────────────────────────────────
        elif ntype == "KernelEmbedding":
            dep = spec.get("dep")
            given = spec.get("given", [])
            if dep and _ref_check(dep, nid, "dep"):
                if type_of.get(dep) != "l2":
                    errors.append(f"[{nid}] KernelEmbedding dep={dep!r} is not L2-valued.")
            type_of[nid] = "l2"
            domain_of[nid] = _marginal_domain(given) if given else domain_of.get(dep, _FULL)

        # ── BoundedAffineMap ────────────────────────────────────────────────
        elif ntype == "BoundedAffineMap":
            dep = spec.get("dep")
            if dep and _ref_check(dep, nid, "dep"):
                type_of[nid] = type_of.get(dep, "unknown")
                domain_of[nid] = domain_of.get(dep, "unknown")
            else:
                type_of[nid] = "unknown"
                domain_of[nid] = "unknown"
            _check_bare_ints(str(spec.get("scale", "")), nid, "scale", errors)
            _check_bare_ints(str(spec.get("shift", "")), nid, "shift", errors)

        # ── ChangeOfMeasure ─────────────────────────────────────────────────
        elif ntype == "ChangeOfMeasure":
            dep = spec.get("dep")
            if dep and _ref_check(dep, nid, "dep"):
                if type_of.get(dep) != "l2":
                    errors.append(f"[{nid}] ChangeOfMeasure dep={dep!r} is not L2-valued.")
            if "density_ratio" not in spec:
                errors.append(f"[{nid}] ChangeOfMeasure missing 'density_ratio'.")
            _check_bare_ints(str(spec.get("density_ratio", "")), nid, "density_ratio", errors)
            type_of[nid] = "l2"
            domain_of[nid] = domain_of.get(dep, _FULL)

        # ── OptimalValue / OptimalSolution ──────────────────────────────────
        elif ntype in {"OptimalValue", "OptimalSolution"}:
            dep = spec.get("dep")
            if dep:
                _ref_check(dep, nid, "dep")
            type_of[nid] = "scalar"
            domain_of[nid] = _SCALAR

        # ── Generic pathwise-differentiable parameters ─────────────────────
        elif ntype == "PathwiseDiffParameter":
            type_of[nid] = "l2"
            domain_vars = spec.get("domain_vars", [])
            domain_of[nid] = _marginal_domain(domain_vars) if domain_vars else _FULL

        elif ntype == "RootDensity":
            type_of[nid] = "l2"
            domain_of[nid] = _FULL

        elif ntype == "ConditionalDensity":
            dep_vars = spec.get("dep_vars", [])
            given = spec.get("given", [])
            if not dep_vars:
                errors.append(f"[{nid}] ConditionalDensity missing 'dep_vars'.")
            if not given:
                errors.append(f"[{nid}] ConditionalDensity missing 'given'.")
            type_of[nid] = "l2"
            domain_of[nid] = _marginal_domain(list(dep_vars) + list(given))

        elif ntype == "DoseResponseFunction":
            if "outcome_var" not in spec:
                errors.append(f"[{nid}] DoseResponseFunction missing 'outcome_var'.")
            if "treatment_var" not in spec:
                errors.append(f"[{nid}] DoseResponseFunction missing 'treatment_var'.")
            type_of[nid] = "l2"
            domain_of[nid] = _marginal_domain(["a"] + list(spec.get("given", [])))

        elif ntype == "CounterfactualDensity":
            if "outcome_var" not in spec:
                errors.append(f"[{nid}] CounterfactualDensity missing 'outcome_var'.")
            if "treatment_var" not in spec:
                errors.append(f"[{nid}] CounterfactualDensity missing 'treatment_var'.")
            type_of[nid] = "l2"
            domain_of[nid] = _marginal_domain([spec.get("outcome_var", "Y")])

        # ── DifferentiableFunction ───────────────────────────────────────────
        elif ntype == "DifferentiableFunction":
            inputs = spec.get("scalar_inputs", [])
            partials = spec.get("partials", [])
            if len(partials) != len(inputs):
                errors.append(
                    f"[{nid}] len(partials)={len(partials)} != "
                    f"len(scalar_inputs)={len(inputs)}."
                )
            for inp in inputs:
                if _ref_check(inp, nid, "scalar_inputs"):
                    if type_of.get(inp) != "scalar":
                        errors.append(
                            f"[{nid}] scalar_inputs contains {inp!r} which "
                            f"is not Scalar-valued (type={type_of.get(inp, '?')})."
                        )
            _check_bare_ints(spec.get("func", ""), nid, "func", errors)
            for i, p in enumerate(partials):
                _check_bare_ints(p, nid, f"partials[{i}]", errors)
            type_of[nid] = "scalar"
            domain_of[nid] = _SCALAR

        # ── FixBinaryArgument ────────────────────────────────────────────────
        elif ntype == "FixBinaryArgument":
            dep = spec.get("dep")
            bv = spec.get("binary_var")
            val = spec.get("value")
            remaining = None
            if dep and _ref_check(dep, nid, "dep"):
                if type_of.get(dep) != "l2":
                    errors.append(f"[{nid}] FixBinaryArgument dep={dep!r} is not L2-valued.")
                dep_type = nodes.get(dep, {}).get("type")
                if dep_type not in {"ConditionalMean", "FixBinaryArgument"}:
                    errors.append(
                        f"[{nid}] FixBinaryArgument dep={dep!r} is "
                        f"{dep_type!r}; expected ConditionalMean or "
                        f"FixBinaryArgument."
                    )
                dep_domain_vars = _vars_from_domain(domain_of.get(dep, "unknown"))
                if dep_domain_vars is None:
                    errors.append(
                        f"[{nid}] cannot infer domain variables for dep={dep!r} "
                        f"with domain {domain_of.get(dep, '?')}."
                    )
                elif bv and bv not in dep_domain_vars:
                    errors.append(
                        f"[{nid}] binary_var={bv!r} not in dep's domain "
                        f"{dep_domain_vars}."
                    )
                elif bv:
                    remaining = [v for v in dep_domain_vars if v != bv]
            if val not in (0, 1):
                errors.append(f"[{nid}] value must be 0 or 1, got {val!r}.")
            # Output domain: remove binary_var from dep's current L2 domain.
            if remaining is not None:
                type_of[nid] = "l2"
                domain_of[nid] = _marginal_domain(remaining) if remaining else _FULL
            else:
                type_of[nid] = "l2"
                domain_of[nid] = "unknown"

    # ── output_node check ────────────────────────────────────────────────────
    if output_id:
        if output_id not in seen:
            errors.append(f"output_node={output_id!r} is not defined in nodes.")
        elif type_of.get(output_id) != "scalar":
            errors.append(
                f"output_node={output_id!r} is {type_of.get(output_id, '?')}-valued, "
                f"expected Scalar."
            )

    return errors


# ---------------------------------------------------------------------------
# Bare-integer detector for func/partial strings
# ---------------------------------------------------------------------------

import re as _re

# Matches a bare integer that is NOT inside Const(...) and NOT part of
# an identifier like h0 or h12.  Uses negative lookbehind/lookahead.
_BARE_INT_RE = _re.compile(
    r"(?<![a-zA-Z_(])"   # not preceded by letter, underscore, or open-paren
    r"\b([0-9]+)\b"      # an integer token
    r"(?!\s*\))"          # not immediately followed by closing paren (inside Const())
)


def _check_bare_ints(expr_str: str, nid: str, field: str, errors: list[str]) -> None:
    """Warn if an expression string contains bare integers outside Const(...)."""
    if not expr_str:
        return
    # Strip out all Const(...) occurrences first, then look for remaining ints
    stripped = _re.sub(r"Const\([^)]*\)", "", expr_str)
    # Also strip h0, h1, ... references
    stripped = _re.sub(r"\bh\d+\b", "", stripped)
    # Now look for remaining bare integers
    matches = _re.findall(r"\b\d+\b", stripped)
    for m in matches:
        # Exponents like ** 2 are common — warn but don't block
        errors.append(
            f"[{nid}] {field} contains bare integer {m!r} — "
            f"use Const({m}) instead. (Expression: {expr_str!r})"
        )


# ---------------------------------------------------------------------------
# EIF validation (post-computation)
# ---------------------------------------------------------------------------

def validate_eif(eif: Expr, variables: list[str] | None = None) -> list[str]:
    """Run sanity checks on a computed symbolic EIF expression.

    Checks:
      1. EIF is not trivially zero (would indicate a failed backward pass).
      2. EIF contains at least one ObsVar (it should depend on the observation).
      3. EIF is a Sum or a single term (expected structure from compute_eif).
      4. Each term contains a centering pattern (u - E[u]) suggesting mean-zero.

    Returns:
        List of warning strings.  Empty = looks good.
    """
    warnings: list[str] = []

    # Check 1: not trivially zero
    if isinstance(eif, Const) and eif.value == 0:
        warnings.append(
            "EIF is identically zero. This usually means the backward pass "
            "failed to propagate adjoints (check that all parent edges are "
            "registered in the graph)."
        )
        return warnings

    # Check 2: contains observation variables
    obs_vars = _collect_obs_vars(eif)
    if not obs_vars:
        warnings.append(
            "EIF contains no observation variables (ObsVar). "
            "An EIF should be a function of the data point z."
        )

    if variables:
        for v in variables:
            if v not in obs_vars:
                # Not necessarily an error — some variables may only appear
                # inside conditional expectations
                pass

    # Check 3: structure
    if isinstance(eif, Sum):
        n_terms = len(eif.terms)
        if n_terms == 0:
            warnings.append("EIF is an empty Sum.")
    # Single term is fine for simple parameters

    # Check 4: look for centering patterns (u - E[u])
    terms = eif.terms if isinstance(eif, Sum) else [eif]
    has_centering = False
    for term in terms:
        if _has_centering(term):
            has_centering = True
            break
    if not has_centering:
        warnings.append(
            "No centering pattern (u - E_P[u]) found in any EIF term. "
            "A valid EIF must have E_P[φ(Z;P)] = 0. This may indicate "
            "a missing MarginalMean or ConditionalMean subtraction."
        )

    return warnings


def _collect_obs_vars(expr: Expr) -> set[str]:
    """Recursively collect all ObsVar names from an expression."""
    result: set[str] = set()
    _walk(expr, result)
    return result


def _walk(expr: Expr, acc: set[str]) -> None:
    if isinstance(expr, ObsVar):
        acc.add(expr.name)
    elif isinstance(expr, Const):
        pass
    elif isinstance(expr, (CondExpect, MargExpect, MargVar)):
        _walk(expr.dep, acc)
    elif isinstance(expr, (Add, Sub, Mul, Div)):
        _walk(expr.a, acc)
        _walk(expr.b, acc)
    elif isinstance(expr, Neg):
        _walk(expr.a, acc)
    elif isinstance(expr, Pow):
        _walk(expr.base, acc)
        _walk(expr.exp, acc)
    elif isinstance(expr, Sum):
        for t in expr.terms:
            _walk(t, acc)


def _has_centering(expr: Expr) -> bool:
    """Check if an expression contains a (u - E_P[u]) pattern anywhere."""
    if isinstance(expr, Sub):
        # u - E_P[u]  or  u - E_P[u|X=x]
        if isinstance(expr.b, (MargExpect, CondExpect)):
            return True
        return _has_centering(expr.a) or _has_centering(expr.b)
    elif isinstance(expr, (Add, Mul, Div)):
        return _has_centering(expr.a) or _has_centering(expr.b)
    elif isinstance(expr, Neg):
        return _has_centering(expr.a)
    elif isinstance(expr, Pow):
        return _has_centering(expr.base)
    elif isinstance(expr, Sum):
        return any(_has_centering(t) for t in expr.terms)
    return False
