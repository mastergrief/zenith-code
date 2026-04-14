"""L5 Family A — learned IR synthesis via autoregressive decoding.

Given 3 `(a, b) → out` IO example pairs, the model emits a math
expression (the program) that explains them. The expression is then
parsed into a GateGraph via `calm.llm_computer.parse.parse_expression`
and executed via `calm.llm_computer.interpret.interpret` against the
query to verify functional correctness.

Scope: arithmetic programs over variables {a, b} and small integer
constants. Expanded scopes (Family B: +Delegate backends, Family C:
hardware nodes) will live in the same module once Family A ships.
"""
