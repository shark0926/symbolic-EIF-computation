"""
expr.py — Symbolic expression language for EIF output.

An EIF φ(v; P) is a real-valued function of a single observation v.
We represent it as a tree of Expr nodes that can be pretty-printed as
closed-form mathematical notation, e.g.

    (-(Y - E_P[Y|X=x])^2 / Var_P[Y]) + (Var_P[Y|X] * (Y - E_P[Y])^2 / Var_P[Y]^2)

Atomic expressions:
  ObsVar      – an observed variable value, e.g. Y or X at the current point v
  CondExpect  – E_P[dep | cond_var=cond_var(v)], evaluated at current obs
  MargExpect  – E_P[dep], a distributional scalar constant w.r.t. v
  MargVar     – Var_P[dep], a distributional scalar constant w.r.t. v
  SymbolicAtom – arbitrary symbolic object rendered verbatim

Compound expressions: Add, Sub, Mul, Div, Neg, Pow, Sum.
All arithmetic operators return simplified forms automatically.
"""

from __future__ import annotations
from typing import Union

Number = Union[int, float]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wrap(value: "Union[Expr, Number]") -> "Expr":
    """Coerce a Python number or Expr to Expr."""
    if isinstance(value, Expr):
        return value
    if isinstance(value, (int, float)):
        return Const(value)
    raise TypeError(f"Cannot coerce {type(value).__name__!r} to Expr")


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class Expr:
    """Base class for symbolic expressions.

    All subclasses support standard arithmetic via operator overloading.
    Arithmetic operators call .simplify() on the resulting node, so
    trivial cases (multiply-by-zero, add-zero, etc.) collapse immediately.
    """

    # ---- arithmetic -------------------------------------------------------

    def __add__(self, other: "Union[Expr, Number]") -> "Expr":
        return Add(self, _wrap(other)).simplify()

    def __radd__(self, other: "Union[Expr, Number]") -> "Expr":
        return Add(_wrap(other), self).simplify()

    def __sub__(self, other: "Union[Expr, Number]") -> "Expr":
        return Sub(self, _wrap(other)).simplify()

    def __rsub__(self, other: "Union[Expr, Number]") -> "Expr":
        return Sub(_wrap(other), self).simplify()

    def __mul__(self, other: "Union[Expr, Number]") -> "Expr":
        return Mul(self, _wrap(other)).simplify()

    def __rmul__(self, other: "Union[Expr, Number]") -> "Expr":
        return Mul(_wrap(other), self).simplify()

    def __truediv__(self, other: "Union[Expr, Number]") -> "Expr":
        return Div(self, _wrap(other)).simplify()

    def __rtruediv__(self, other: "Union[Expr, Number]") -> "Expr":
        return Div(_wrap(other), self).simplify()

    def __neg__(self) -> "Expr":
        return Neg(self).simplify()

    def __pow__(self, n: "Union[Expr, Number]") -> "Expr":
        return Pow(self, _wrap(n)).simplify()

    # ---- simplification ---------------------------------------------------

    def simplify(self) -> "Expr":
        """Return a simplified equivalent expression. Default: identity."""
        return self

    # ---- display ----------------------------------------------------------

    def __repr__(self) -> str:
        raise NotImplementedError(f"{type(self).__name__}.__repr__")

    def __str__(self) -> str:
        return repr(self)


# ---------------------------------------------------------------------------
# Atomic expressions
# ---------------------------------------------------------------------------

class Const(Expr):
    """A numeric constant, e.g. Const(2) or Const(-1.0)."""

    def __init__(self, value: Number):
        if not isinstance(value, (int, float)):
            raise TypeError(f"Const expects a number, got {type(value)}")
        self.value = value

    def simplify(self) -> "Expr":
        return self

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Const) and self.value == other.value

    def __hash__(self) -> int:
        return hash(("Const", self.value))

    def __repr__(self) -> str:
        # Print integers without decimal point for cleanliness
        if isinstance(self.value, float) and self.value == int(self.value):
            return str(int(self.value))
        return str(self.value)


class ObsVar(Expr):
    """The value of a named random variable at the current observation v.

    Example: ObsVar("Y") represents the scalar y = Y(v).
    """

    def __init__(self, name: str):
        self.name = name

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ObsVar) and self.name == other.name

    def __hash__(self) -> int:
        return hash(("ObsVar", self.name))

    def __repr__(self) -> str:
        return self.name


class CondExpect(Expr):
    """E_P[dep | given₁=given₁(v), ..., fixed₁=c₁, ...] at the current obs.

    This represents the conditional expectation *as a function* of the
    current data point — i.e., it is random w.r.t. the observation.

    Args:
        measure_name: Name of the measure (e.g. "P").
        dep:          Expression inside the expectation.
        given:        Variable names evaluated at the current obs (free).
        fixed_vals:   Variable names fixed to specific values, e.g. {"A": 1}.
                      Used by FixBinaryArgument to represent E_P[Y|A=1, X=x].
    """

    def __init__(
        self,
        measure_name: str,
        dep: Expr,
        given: list[str],
        fixed_vals: dict[str, int] | None = None,
    ):
        self.measure_name = measure_name
        self.dep = dep
        self.given = list(given)
        self.fixed_vals: dict[str, int] = fixed_vals or {}

    def __repr__(self) -> str:
        # Free variables evaluated at the current obs: X=x
        free_parts  = [f"{v}={v.lower()}" for v in self.given]
        # Fixed variables pinned to a constant: A=1
        fixed_parts = [f"{v}={c}" for v, c in self.fixed_vals.items()]
        cond_str = ", ".join(fixed_parts + free_parts)
        return f"E_{self.measure_name}[{self.dep} | {cond_str}]"


class MargExpect(Expr):
    """E_P[dep] — marginal expectation, a distributional scalar constant.

    This is a fixed number determined by P (not random w.r.t. v).

    Args:
        measure_name: Name of the measure.
        dep: Expression inside the expectation.
    """

    def __init__(self, measure_name: str, dep: Expr):
        self.measure_name = measure_name
        self.dep = dep

    def __repr__(self) -> str:
        return f"E_{self.measure_name}[{self.dep}]"


class MargVar(Expr):
    """Var_P[dep] — marginal variance, a distributional scalar constant."""

    def __init__(self, measure_name: str, dep: Expr):
        self.measure_name = measure_name
        self.dep = dep

    def __repr__(self) -> str:
        return f"Var_{self.measure_name}[{self.dep}]"


class SymbolicAtom(Expr):
    """An arbitrary symbolic atom rendered verbatim.

    This is used for primitives whose natural symbolic output does not fit one
    of the more structured atomic expression classes above, such as kernel
    embeddings, optimization primitives, or density-like quantities.
    """

    def __init__(self, text: str):
        self.text = text

    def __eq__(self, other: object) -> bool:
        return isinstance(other, SymbolicAtom) and self.text == other.text

    def __hash__(self) -> int:
        return hash(("SymbolicAtom", self.text))

    def __repr__(self) -> str:
        return self.text


# ---------------------------------------------------------------------------
# Compound expressions
# ---------------------------------------------------------------------------

class Add(Expr):
    def __init__(self, a: Expr, b: Expr):
        self.a = a
        self.b = b

    def simplify(self) -> Expr:
        a = self.a.simplify()
        b = self.b.simplify()
        if isinstance(a, Const) and a.value == 0:
            return b
        if isinstance(b, Const) and b.value == 0:
            return a
        if isinstance(a, Const) and isinstance(b, Const):
            return Const(a.value + b.value)
        return Add(a, b)

    def __repr__(self) -> str:
        return f"({self.a} + {self.b})"


class Sub(Expr):
    def __init__(self, a: Expr, b: Expr):
        self.a = a
        self.b = b

    def simplify(self) -> Expr:
        a = self.a.simplify()
        b = self.b.simplify()
        if isinstance(b, Const) and b.value == 0:
            return a
        if isinstance(a, Const) and a.value == 0:
            return Neg(b).simplify()
        if isinstance(a, Const) and isinstance(b, Const):
            return Const(a.value - b.value)
        return Sub(a, b)

    def __repr__(self) -> str:
        return f"({self.a} - {self.b})"


class Mul(Expr):
    def __init__(self, a: Expr, b: Expr):
        self.a = a
        self.b = b

    def simplify(self) -> Expr:
        a = self.a.simplify()
        b = self.b.simplify()
        if isinstance(a, Const) and a.value == 0:
            return Const(0)
        if isinstance(b, Const) and b.value == 0:
            return Const(0)
        if isinstance(a, Const) and a.value == 1:
            return b
        if isinstance(b, Const) and b.value == 1:
            return a
        if isinstance(a, Const) and a.value == -1:
            return Neg(b).simplify()
        if isinstance(b, Const) and b.value == -1:
            return Neg(a).simplify()
        if isinstance(a, Const) and isinstance(b, Const):
            return Const(a.value * b.value)
        return Mul(a, b)

    def __repr__(self) -> str:
        return f"({self.a} * {self.b})"


class Div(Expr):
    def __init__(self, a: Expr, b: Expr):
        self.a = a
        self.b = b

    def simplify(self) -> Expr:
        a = self.a.simplify()
        b = self.b.simplify()
        if isinstance(a, Const) and a.value == 0:
            return Const(0)
        if isinstance(b, Const) and b.value == 1:
            return a
        if isinstance(a, Const) and isinstance(b, Const) and b.value != 0:
            return Const(a.value / b.value)
        return Div(a, b)

    def __repr__(self) -> str:
        return f"({self.a} / {self.b})"


class Neg(Expr):
    def __init__(self, a: Expr):
        self.a = a

    def simplify(self) -> Expr:
        a = self.a.simplify()
        if isinstance(a, Const):
            return Const(-a.value)
        if isinstance(a, Neg):
            return a.a  # double negation
        return Neg(a)

    def __repr__(self) -> str:
        inner = repr(self.a)
        # Avoid extra parens for simple terms
        if isinstance(self.a, (Const, ObsVar, CondExpect, MargExpect, MargVar, SymbolicAtom)):
            return f"-{inner}"
        return f"(-{inner})"


class Pow(Expr):
    def __init__(self, base: Expr, exp: Expr):
        self.base = base
        self.exp = exp

    def simplify(self) -> Expr:
        base = self.base.simplify()
        exp = self.exp.simplify()
        if isinstance(exp, Const):
            if exp.value == 0:
                return Const(1)
            if exp.value == 1:
                return base
        if isinstance(base, Const) and isinstance(exp, Const):
            return Const(base.value ** exp.value)
        return Pow(base, exp)

    def __repr__(self) -> str:
        # Use superscript-style for integer exponents
        if isinstance(self.exp, Const) and isinstance(self.exp.value, int):
            return f"{self.base}^{self.exp.value}"
        return f"({self.base})^({self.exp})"


class Sum(Expr):
    """A flat sum of terms — used to collect EIF contributions.

    Simplification flattens nested Sums and drops zero terms.
    """

    def __init__(self, terms: list[Expr]):
        self.terms = list(terms)

    def simplify(self) -> Expr:
        flat: list[Expr] = []
        for t in self.terms:
            s = t.simplify()
            if isinstance(s, Sum):
                flat.extend(s.terms)
            elif not (isinstance(s, Const) and s.value == 0):
                flat.append(s)
        if not flat:
            return Const(0)
        if len(flat) == 1:
            return flat[0]
        return Sum(flat)

    def __repr__(self) -> str:
        if not self.terms:
            return "0"
        lines = []
        for i, t in enumerate(self.terms):
            s = repr(t)
            if i == 0:
                lines.append(s)
            else:
                # Pretty-print additions of negatives as subtractions
                if isinstance(t, (Neg,)) or (isinstance(t, Mul) and isinstance(t.a, Const) and t.a.value < 0):
                    lines.append(f"\n  + {s}")
                else:
                    lines.append(f"\n  + {s}")
        return "".join(lines)
