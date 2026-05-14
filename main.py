#!/usr/bin/env python3
"""
main.py — Compute the symbolic EIF from a JSON computation graph.

Usage
-----
    python main.py graph.json          # from file
    python main.py < graph.json        # from stdin
    cat graph.json | python main.py    # piped
    python main.py -                   # explicit stdin

The JSON must follow the schema produced by the LLM prompt
(see pydimple/symbolic/deserializer.py for the full spec).
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path


# ---------------------------------------------------------------------------
# Bootstrap: load pydimple.symbolic without importing pydimple.funs
# (funs.py requires scipy/lightgbm which may not be installed)
# ---------------------------------------------------------------------------

def _load_symbolic():
    repo_root    = Path(__file__).resolve().parent
    pydimple_dir = repo_root / "pydimple"

    if "pydimple" not in sys.modules:
        stub = types.ModuleType("pydimple")
        stub.__path__ = [str(pydimple_dir)]
        sys.modules["pydimple"] = stub

    init = pydimple_dir / "symbolic" / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        "pydimple.symbolic", init,
        submodule_search_locations=[str(init.parent)],
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pydimple.symbolic"] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_input(args: list[str]) -> str:
    """Return raw JSON text from file path arg or stdin."""
    if not args or args[0] == "-":
        if sys.stdin.isatty():
            print("Reading JSON from stdin (Ctrl-D to finish):", file=sys.stderr)
        return sys.stdin.read()
    path = Path(args[0])
    if not path.exists():
        sys.exit(f"Error: file not found: {path}")
    return path.read_text()


def _print_section(title: str, content: str) -> None:
    width = 60
    print("=" * width)
    print(title)
    print("=" * width)
    print(content)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    sym = _load_symbolic()
    deserialize = sym.deserialize
    compute_eif = sym.compute_eif

    raw = _read_input(sys.argv[1:])

    # ── parse JSON ────────────────────────────────────────────────────────────
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        sys.exit(f"Error: invalid JSON — {exc}")

    validate_graph = sym.validate_graph
    validate_eif   = sym.validate_eif

    # ── Stage 1: validate the graph structure ────────────────────────────────
    graph_errors = validate_graph(data)
    if graph_errors:
        _print_section("Graph validation errors", "\n".join(
            f"  [{i+1}] {e}" for i, e in enumerate(graph_errors)
        ))
        sys.exit(f"Aborting: {len(graph_errors)} graph error(s) found.")

    print("Graph validation passed.\n")

    # ── deserialize ───────────────────────────────────────────────────────────
    try:
        output_node = deserialize(data)
    except ValueError as exc:
        sys.exit(f"Error: {exc}")
    except KeyError as exc:
        sys.exit(f"Error: {exc}")

    # ── compute EIF ───────────────────────────────────────────────────────────
    try:
        eif = compute_eif(output_node)
    except Exception as exc:
        sys.exit(f"Error during EIF computation: {exc}")

    # ── print results ─────────────────────────────────────────────────────────
    psi_expr = output_node._value.expr   # set by compute_eif's forward pass

    _print_section("Functional value  ψ(P)", repr(psi_expr))
    _print_section("Efficient Influence Function  φ(z; P)", repr(eif))

    # ── Stage 2: validate the EIF ────────────────────────────────────────────
    variables = data.get("variables", [])
    eif_warnings = validate_eif(eif, variables)
    if eif_warnings:
        _print_section("EIF validation warnings", "\n".join(
            f"  [{i+1}] {w}" for i, w in enumerate(eif_warnings)
        ))
    else:
        print("EIF validation passed.")


if __name__ == "__main__":
    main()
