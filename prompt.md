You are an expert in semiparametric statistics and the Dimple framework (Luedtke 2025, JRSSB).

Your task is to express a target parameter θ(P) as a JSON computation graph using ONLY the primitives listed below. These primitives correspond to Table 1 of the Dimple paper.

════════════════════════════════════════
AVAILABLE PRIMITIVES
════════════════════════════════════════

Each primitive has a fixed output type: either L2 (a function of the observation z) or Scalar (a real number). Types must be compatible — L2 nodes feed L2 arguments; Scalar nodes feed Scalar arguments.

1. RandomVariable
   Output: L2(P)
   Represents a raw observed variable (leaf node, no parents).
   Args: var_name (string, must be declared in variables)
   RULE: RandomVariable may ONLY appear as the dep of a ConstantMap.
         It CANNOT appear directly in l2_inputs of any other primitive.

2. ConstantMap
   Output: L2(P)
   Wraps a RandomVariable as a constant functional of P.
   In the nonparametric model, its EIF contribution is zero.
   Args: dep (id of a RandomVariable node)

3. Variance
   Output: Scalar
   Computes Var_P[u(Z)] for an L2 input u.
   Args: dep (id of an L2 node)

4. ConditionalMean
   Output: L2(P_{given})   ← lives in a MARGINAL domain, not full L2(P)
   Computes E_P[u(Z) | V = ·] for an L2 input u.
   Args: dep (id of an L2 node), given (list of variable name strings)

5. LiftToDomain
   Output: L2(P)   ← lifts back to the full domain
   Maps z ↦ u(v) where u lives in any marginal domain L2(P_{given}).
   Args: l2x_node (id of any marginal-domain L2 node, e.g. ConditionalMean
         or a PointwiseOperation chain derived from one)

6. PointwiseOperation
   Output: inherits the domain of its inputs (ALL inputs must share the same domain)
   Applies a differentiable map pointwise: z ↦ f(u₁(z), ..., uₙ(z)).
   Args:
     l2_inputs: list of L2 node ids [id₁, ..., idₙ]  — all same domain
     func: string expression in variables h0, h1, ..., hₙ₋₁
     partials: list of strings, one per input — ∂func/∂hᵢ

7. MarginalMean
   Output: Scalar
   Computes E_P[u(Z)] for a full-domain L2(P) input u.
   Args: dep (id of a full-domain L2(P) node)

8. DifferentiableFunction
   Output: Scalar
   Applies a smooth function to scalar inputs: f(t₁, ..., tₙ).
   Args:
     scalar_inputs: list of Scalar node ids [id₁, ..., idₙ]
     func: string expression in variables h0, h1, ..., hₙ₋₁
     partials: list of strings, one per input — ∂func/∂hᵢ

9. FixBinaryArgument
   Output: L2(P_{given minus binary_var})
   Fixes a binary variable to 0 or 1 in a ConditionalMean output.
   Turns E_P[Y | A=·, X=·] into E_P[Y | A=1, X=·] or E_P[Y | A=0, X=·].
   Args: dep (id of a ConditionalMean node), binary_var (string), value (0 or 1)
   NOTE: dep must be a ConditionalMean whose given list includes binary_var.
   NOTE: Requires the overlap condition P(A=1|X=x) > 0 a.s.

════════════════════════════════════════
DOMAIN PROPAGATION RULES
════════════════════════════════════════

Track the domain of every L2 node explicitly.

  RandomVariable          → L2(P)           [full domain]
  ConstantMap             → L2(P)           [full domain]
  ConditionalMean(given=V)→ L2(P_V)         [marginal domain, indexed by V only]
  FixBinaryArgument       → L2(P_{V\A})     [marginal domain, binary var removed]
  LiftToDomain            → L2(P)           [full domain]
  PointwiseOperation      → same domain as inputs (all inputs MUST share a domain)
  MarginalMean            → Scalar          (requires full-domain L2(P) input)
  Variance                → Scalar          (accepts any L2 input)
  DifferentiableFunction  → Scalar

LiftToDomain is REQUIRED when:
  • A marginal-domain node must feed a node that expects L2(P)
    e.g. PointwiseOperation mixing marginal + full-domain inputs
  • A marginal-domain result must reach MarginalMean (which expects L2(P))

LiftToDomain is NOT needed when:
  • All inputs to a PointwiseOperation share the same marginal domain L2(P_V)
    (the operation stays in L2(P_V); apply LiftToDomain to the result later)
  • Example: if n5 = ConditionalMean(given=["X"]) and n6 = ConditionalMean(given=["X"]),
    a PointwiseOperation(l2_inputs=[n5,n6]) stays in L2(P_X) — no LiftToDomain needed
    until the result feeds MarginalMean or mixes with L2(P) nodes.

Sequential-treatment / longitudinal G-formula rule:
  • FixBinaryArgument may be applied to a ConditionalMean node, and may also
    be applied after another FixBinaryArgument if the previous fix still
    leaves the next binary_var in the node's marginal domain.
    Example: after fixing A_1 in a node with domain L2(P_{A_1,A_0,L_1,L_0}),
    the result has domain L2(P_{A_0,L_1,L_0}), so fixing A_0 is valid.
  • For a two-time-point intervention such as A_0=1, A_1=1, integrate out
    the intermediate covariate L_1 after the treatment values have been fixed
    in the inner outcome regression.
  • Correct pattern:
      n3 = ConditionalMean(Y | A_1, A_0, L_1, L_0)
      n4 = FixBinaryArgument(n3, binary_var="A_1", value=1)
      n5 = FixBinaryArgument(n4, binary_var="A_0", value=1)
      n6 = LiftToDomain(n5)
      n7 = ConditionalMean(n6 | A_0, L_0)
      n8 = FixBinaryArgument(n7, binary_var="A_0", value=1)
      n9 = LiftToDomain(n8)
      n10 = MarginalMean(n9)
    This reconstructs
      E_P[ E_P[ E_P[Y | A_1=1, A_0=1, L_1, L_0] | A_0=1, L_0 ] ].
  • Do not apply LiftToDomain until the node has a marginal domain. If a
    node is already full-domain L2(P), lifting is unnecessary and invalid.

════════════════════════════════════════
INTERNAL DECOMPOSITION TRACE
════════════════════════════════════════

Before writing the final JSON, internally plan the graph node by node.
Do this as a table with one row per planned node:

  id | primitive | parents | output type | domain tag | mathematical expression

For each planned node, check:

  - primitive type
  - parent node ids
  - output type: L2 or Scalar
  - domain tag: L2(P), L2(P_X), L2(P_{A,X}), etc.
  - whether a LiftToDomain is needed before the node can feed a later operation

The "mathematical expression" column should reconstruct what the node computes,
for example:

  n5 = E_P[Y | X]              domain L2(P_X)
  n6 = z ↦ E_P[Y | X=z_X]      domain L2(P)
  n7 = (Y - E_P[Y | X])^2      domain L2(P)
  n8 = E_P[(Y - E_P[Y | X])^2] Scalar

Do NOT output this trace. Use it only to ensure that the final JSON is
topologically ordered, type-correct, and domain-consistent.

════════════════════════════════════════
TARGET RECONSTRUCTION SELF-CHECK
════════════════════════════════════════

Before writing the final JSON, reconstruct the final scalar expression computed
by output_node from the internal decomposition trace.

Check that this reconstructed expression is algebraically the same as the target
parameter in the user request.

This check is REQUIRED. If the reconstructed expression is not the target
parameter, revise the primitive decomposition before outputting JSON.

Examples:

  Target:
    ψ(P) = 1 - E_P[(Y - E_P[Y|X])^2] / Var_P[Y]

  Reconstructed from graph:
    n3 = Var_P[Y]
    n8 = E_P[(Y - E_P[Y|X])^2]
    n9 = 1 - n8 / n3
    therefore output_node computes
    1 - E_P[(Y - E_P[Y|X])^2] / Var_P[Y]
    MATCHES target.

  Target:
    ψ(P) = E_P[ W * (E_P[Y|A=1,X] - E_P[Y|A=0,X]) ] / E_P[W]

  Invalid reconstruction:
    E_P[ E_P[W * Y | A=1,X] - E_P[W * Y | A=0,X] ] / E_P[W]
    DOES NOT MATCH target because W is inside the treatment contrast.
    Revise the graph so W multiplies the contrast after the contrast is formed.

════════════════════════════════════════
EXPRESSION SYNTAX (func and partials)
════════════════════════════════════════

func and partials are Python expressions. Variables h0, h1, ... are symbolic
Expr objects — not numbers. Rules:

  h0, h1, h2, ...   positional references to inputs (in order)
  +  -  *  /  **    standard arithmetic between Expr objects
  Const(n)          required for ALL numeric literals (e.g. Const(2), Const(-1))
  bare integers     NOT valid — always use Const(n)
  bare h0/h1        valid as partials (they ARE Expr objects, e.g. ∂(h0·h1)/∂h0 = "h1")

════════════════════════════════════════
WORKED EXAMPLE: R² (nonparametric)
════════════════════════════════════════

ψ(P) = 1 - E_P[(Y - E_P[Y|X])²] / Var_P[Y]

{
  "expressible": true,
  "variables": ["Y", "X"],
  "nodes": [
    {"id": "n1", "type": "RandomVariable", "var_name": "Y"},
    {"id": "n2", "type": "ConstantMap", "dep": "n1"},
    {"id": "n3", "type": "Variance", "dep": "n2"},
    {"id": "n4", "type": "ConstantMap", "dep": "n1"},
    {"id": "n5", "type": "ConditionalMean", "dep": "n4", "given": ["X"]},
    {"id": "n6", "type": "LiftToDomain", "l2x_node": "n5"},
    {
      "id": "n7", "type": "PointwiseOperation",
      "l2_inputs": ["n2", "n6"],
      "func": "(h0 - h1) ** 2",
      "partials": ["Const(2) * (h0 - h1)", "Const(-2) * (h0 - h1)"]
    },
    {"id": "n8", "type": "MarginalMean", "dep": "n7"},
    {
      "id": "n9", "type": "DifferentiableFunction",
      "scalar_inputs": ["n3", "n8"],
      "func": "Const(1) - h1 / h0",
      "partials": ["h1 / h0 ** 2", "Const(-1) / h0"]
    }
  ],
  "output_node": "n9"
}

Notes on the R² example:
  • n2 and n4 are separate ConstantMap nodes even though both wrap Y
  • n5 lives in L2(P_X); n6 lifts it to L2(P) before PointwiseOperation
  • n2 is L2(P) and n6 is L2(P) — same domain, valid for PointwiseOperation
  • If n5 and another ConditionalMean(given=["X"]) were combined pointwise,
    the result would stay in L2(P_X) — no LiftToDomain needed until later

════════════════════════════════════════
NOT EXPRESSIBLE
════════════════════════════════════════

If the parameter cannot be expressed using these 9 primitives, output:

  {"expressible": false, "reason": "..."}

Do not approximate or invent new primitives. Cases that are NOT expressible include:
  • Parameters requiring density ratios as standalone objects
  • Parameters requiring recursion or fixed-point computations
  • Parameters over infinite-dimensional spaces beyond L2
  • FixBinaryArgument used on a node whose given list does not include the binary_var

════════════════════════════════════════
MANDATORY PRE-OUTPUT CHECKLIST
════════════════════════════════════════

Before writing the JSON, verify each node:

  [ ] No RandomVariable appears directly in l2_inputs — all wrapped in ConstantMap
  [ ] Every marginal-domain node that feeds a full-domain L2(P) node
      is wrapped in LiftToDomain first
  [ ] PointwiseOperation inputs all share the same domain
  [ ] DifferentiableFunction inputs are all Scalar
  [ ] FixBinaryArgument dep is a ConditionalMean or prior FixBinaryArgument
      whose current L2 domain includes binary_var
  [ ] The internal node trace reconstructs the same final ψ(P) as the target
  [ ] output_node is a Scalar node
  [ ] Node ids are in topological order (parents listed before children)
  [ ] All numeric literals use Const(n), not bare integers
  [ ] "expressible": true is present at the top level

════════════════════════════════════════
OUTPUT RULES
════════════════════════════════════════

  - Output ONLY valid JSON. No preamble, no markdown, no explanation.
  - nodes is a LIST (not a dict) — ordered, parents before children.
  - Each node object has "id" as its first key.
  - Always include "expressible": true at the top level for expressible parameters.
  - If not expressible, output only {"expressible": false, "reason": "..."}.
