# Symbolic EIF Artifact

This repository contains a small standalone artifact for computing symbolic
efficient influence functions (EIFs) from JSON computation graphs.

The intended workflow is:

1. Give `prompt.md` to an AI model together with the target statistical
   parameter.
2. Ask the model to return only the JSON computation graph.
3. Save that JSON as `graph.json`, replacing the example file in this directory.
4. Run:

```bash
python3 main.py graph.json
```

The program validates the graph, constructs the symbolic computation graph, runs
the backward pass, and prints both the functional value `psi(P)` and the symbolic
EIF.

## Files

- `main.py`: command-line entry point.
- `prompt.md`: prompt template for asking an AI model to produce a supported
  JSON computation graph.
- `graph.json`: default graph file. Replace this with a new AI-generated graph
  when running a new example.
- `pydimple/symbolic/`: standalone symbolic backend.
- `examples/`: JSON graphs for the examples discussed in the paper draft.

## Prompting an AI Model

Use `prompt.md` as the model instruction. After the model returns JSON, paste the
result into `graph.json` and run the command-line tool:

```bash
python3 main.py graph.json
```

If validation fails, the error message identifies the graph node and structural
rule that failed. Revise the JSON or re-prompt the model, then rerun the command.

## Example Commands

Run the default graph:

```bash
python3 main.py graph.json
```

Run one of the included examples:

```bash
python3 main.py examples/marginal_mean.json
python3 main.py examples/R2.json
python3 main.py examples/ATE.json
python3 main.py examples/subgroup_CATE.json
python3 main.py examples/TTP_longitudinal.json
```

You can also pipe JSON through standard input:

```bash
python3 main.py - < examples/ATE.json
```

## JSON Format

A graph should have the following top-level fields:

```json
{
  "expressible": true,
  "variables": ["Y", "A", "X"],
  "nodes": [
    {"id": "n1", "type": "RandomVariable", "var_name": "Y"}
  ],
  "output_node": "n1"
}
```

Nodes must appear in topological order: every dependency should be defined
before it is referenced. The supported primitive names and node-specific fields
are implemented in `pydimple/symbolic/deserializer.py` and checked by
`pydimple/symbolic/validator.py`.

## Requirements

The command-line artifact uses only the Python standard library. It has been
tested with Python 3.10+.

## Notes

- The printed EIF is intentionally a raw symbolic chain-rule output. It may
  contain algebraic or conditional-expectation terms that can be simplified by
  hand.
- Validation checks whether a graph is structurally well formed for the backend.
  It does not prove that the graph represents the intended statistical
  functional.
