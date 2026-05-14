"""
pydimple.symbolic — Semi-symbolic EIF computation engine.

This subpackage implements a computation-graph-based system for deriving
Efficient Influence Functions (EIFs) symbolically, following the Dimple
framework of Luedtke (2025, JRSSB).

Public API
----------
Expression language (expr.py):
  Const, ObsVar, CondExpect, MargExpect, MargVar
  Add, Sub, Mul, Div, Neg, Pow, Sum

Core types (core.py):
  Measure
  ScalarValue, L2Value
  ScalarAdjoint, L2Adjoint

Graph (graph.py):
  GraphNode
  ComputationGraph
  compute_eif          ← main entry point

Primitives (primitives.py):
  RandomVariable       ← leaf node (raw observed variable)
  ConstantMap          ← θ₁/θ₃ in R² example
  Variance             ← θ₂
  ConditionalMean      ← θ₄
  RFoldConditionalMean
  ConditionalVariance
  ConditionalCovariance
  LiftToDomain         ← θ₅
  PointwiseOperation   ← θ₆
  MarginalMean         ← θ₇
  InnerProduct
  SquaredNorm
  KernelEmbedding
  BoundedAffineMap
  ChangeOfMeasure
  OptimalValue
  OptimalSolution
  PathwiseDiffParameter
  RootDensity
  ConditionalDensity
  DoseResponseFunction
  CounterfactualDensity
  DifferentiableFunction ← θ₈
  HadamardDiffMap
  CoordinateProjection
  LiftToNewDomain
  FixBinaryArgument

Quickstart (R² example skeleton)::

    from pydimple.symbolic import (
        Measure, Const, compute_eif,
        RandomVariable, ConstantMap, Variance, ConditionalMean,
        LiftToDomain, PointwiseOperation, MarginalMean,
        DifferentiableFunction,
    )

    P = Measure("P", ["Y", "X"])
    Y = RandomVariable(P, "Y")

    theta1 = ConstantMap(P, Y, name="theta1")          # z ↦ y
    theta2 = Variance(P, theta1, name="theta2")        # Var_P[Y]
    theta3 = ConstantMap(P, Y, name="theta3")          # z ↦ y
    theta4 = ConditionalMean(P, theta3, given=["X"],   # E_P[Y|X=·]
                             name="theta4")
    theta5 = LiftToDomain(P, theta4, name="theta5")    # z ↦ μ_P(x)
    theta6 = PointwiseOperation(                       # z ↦ (y−μ(x))²
        P,
        func=lambda h1, h5: (h1 - h5) ** 2,
        partials=[
            lambda h1, h5: Const(2) * (h1 - h5),
            lambda h1, h5: Const(-2) * (h1 - h5),
        ],
        l2_inputs=[theta1, theta5],
        name="theta6",
    )
    theta7 = MarginalMean(P, theta6, name="theta7")    # E_P[(Y−μ)²]
    theta8 = DifferentiableFunction(                   # 1 − h₇/h₂
        P,
        func=lambda h2, h7: Const(1) - h7 / h2,
        partials=[
            lambda h2, h7: h7 / h2 ** 2,
            lambda h2, h7: Const(-1) / h2,
        ],
        scalar_inputs=[theta2, theta7],
        name="theta8",
    )

    # Compute the symbolic EIF
    # eif = compute_eif(theta8)
"""

from .expr import (
    Expr, Const, ObsVar, CondExpect, MargExpect, MargVar, SymbolicAtom,
    Add, Sub, Mul, Div, Neg, Pow, Sum,
)
from .core import (
    Measure,
    NodeValue, ScalarValue, L2Value,
    AdjointValue, ScalarAdjoint, L2Adjoint,
)
from .graph import GraphNode, ComputationGraph, compute_eif
from .deserializer import deserialize
from .validator import validate_graph, validate_eif
from .verify import DiscretePopulation, eval_expr, verify_graph
from . import api  # installs operator overloading and Measure factory methods
from .primitives import (
    RandomVariable,
    Primitive,
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
    HadamardDiffMap,
    CoordinateProjection,
    LiftToNewDomain,
    FixBinaryArgument,
)

__all__ = [
    # expr
    "Expr", "Const", "ObsVar", "CondExpect", "MargExpect", "MargVar", "SymbolicAtom",
    "Add", "Sub", "Mul", "Div", "Neg", "Pow", "Sum",
    # core
    "Measure",
    "NodeValue", "ScalarValue", "L2Value",
    "AdjointValue", "ScalarAdjoint", "L2Adjoint",
    # graph
    "GraphNode", "ComputationGraph", "compute_eif",
    # deserializer
    "deserialize",
    # validator
    "validate_graph", "validate_eif",
    # verify
    "DiscretePopulation", "eval_expr", "verify_graph",
    # primitives
    "RandomVariable",
    "Primitive",
    "ConstantMap",
    "Variance",
    "ConditionalMean",
    "RFoldConditionalMean",
    "ConditionalVariance",
    "ConditionalCovariance",
    "LiftToDomain",
    "PointwiseOperation",
    "MarginalMean",
    "InnerProduct",
    "SquaredNorm",
    "KernelEmbedding",
    "BoundedAffineMap",
    "ChangeOfMeasure",
    "OptimalValue",
    "OptimalSolution",
    "PathwiseDiffParameter",
    "RootDensity",
    "ConditionalDensity",
    "DoseResponseFunction",
    "CounterfactualDensity",
    "DifferentiableFunction",
    "HadamardDiffMap",
    "CoordinateProjection",
    "LiftToNewDomain",
    "FixBinaryArgument",
]
