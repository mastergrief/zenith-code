---
paths:
  - "calm/llm_computer/recursion.py"
  - "calm/llm_computer/auto_upgrade.py"
  - "calm/llm_computer/persistent_knowledge.py"
  - "calm/llm_computer/facades/*_auto.py"
  - "calm/llm_computer/facades/*_meta.py"
  - "scripts/*metafacade*.py"
  - "scripts/*autonomous*.py"
  - "scripts/gemma_learning_loop_demo.py"
---

# Recursion — card-level self-improvement via CALM oracle

The substrate's killer property is that **cards can build cards**.
Each level of recursion is guarded by CALM's deterministic verifier,
so improvements compound without the bias-amplification failure mode
of self-training on a learned judge (Self-Instruct, RLAIF, etc.).

**Shipped capabilities**:
- **Fact-level recursion** — `auto_upgrade.py` +
  `gemma_learning_loop_demo.py`: wrong → correct pipeline.
- **Level 1** — generic decode-path facade auto-generator
  (`calm/llm_computer/recursion.py`). Shipped auto-facades include
  factorial, fibonacci, combinations, permutations, power, next_prime.
- **Level 2** — `MetaFacade.from_oracle(fn_name, arity)` synthesizes
  the `FacadeSpec` itself. Shipped meta-facades include factorial,
  combinations, gcd, lcm, fibonacci.

Level 3 (substrate designs NEW meta-facades from observed failure
traces) is the remaining frontier.

> Historical receipts (shipped-facade dated inventory with commit
> SHAs, per-level demo scripts + eval files, code-PT self-distill
> roadmap R-numbers): see `MEMORY/atlas/recursion_arc.md`.

## Level 1 — decode-path facade auto-generator (SHIPPED)

**Core module**: `calm/llm_computer/recursion.py`.

```python
from calm.llm_computer.recursion import (
    FacadeSpec, validate_facade, generate_facade, import_facade_class,
)

spec = FacadeSpec(
    name="Factorial",
    module_name="factorial_auto",
    description="Factorial (n!) via CALM safe_eval oracle.",
    parse_patterns=[r"factorial\s+of\s+(-?\d+)", r"(-?\d+)\s*!"],
    eval_expr="factorial({a})",
    max_operand=20,
    operand_count=1,
)

# Three CALM gates:
validate_facade(spec, oracle_test_cases)   # 1. safe_eval validates
path = generate_facade(spec, overwrite=True)  # 2. ast.parse-checked write
Cls = import_facade_class(spec); facade = Cls()
facade.install(gemma, tok)                 # 3. live A/B gate
```

**Three CALM-anchored gates** keep the loop drift-free:

1. `validate_facade(spec, cases)` — `safe_eval(spec.eval_expr.format(a=..))`
   must match expected for every oracle case BEFORE any file touches disk.
2. `generate_facade(spec)` — `compile(code, path, "exec")` syntax-checks
   the rendered Python source before writing. Path resolved via
   `_REPO_ROOT` absolute, not cwd-relative.
3. Live A/B — baseline (`use_bias=False`) vs facade (`use_bias=True`)
   on Gemma. Facade gets added to the registry only if it beats
   baseline with 0 regressions.

**What's in the template** (`_TEMPLATE` at line ~70 of recursion.py):
- `@dataclass {name}Result` — per-facade result type
- `class {name}Facade` — standard install/detach/parse/evaluate/solve API
- `_PARSE_RES` list of compiled regexes (rendered via
  `_render_parse_res_literals`; uses `{pat!r}` NOT `r{pat!r}` to
  avoid double-escaping — see inline comment at `_render_...` definition)
- `_EVAL_TEMPLATE` — safe_eval formatter string with `{a}` / `{b}` slots
- Standard `_SPACE_TOKEN_ID = 236743` strip + `POST_BIAS_BUDGET = 4`
  post-bias truncation (see `compute_facades.md` + `embed_intelligence.md`)
- 12-digit `_parse_int` cap

Extended to other card types: if a future PT or compiled card follows
the "parse → oracle → bias" shape, drop a `FacadeSpec`-analog and
re-use the three CALM gates.

## Level 2 — MetaFacade synthesizes the spec itself (SHIPPED)

Instead of a human writing `parse_patterns` + `eval_expr`, MetaFacade
encodes canonical NL patterns per arity:

```python
from calm.llm_computer.recursion import MetaFacade

spec = MetaFacade.from_oracle(
    fn_name="combinations", arity=2, max_operand=100,
    extra_patterns=[r"(-?\d+)\s+choose\s+(-?\d+)"],
)
# spec is a standard FacadeSpec, goes through the same Level-1 pipeline
```

Canonical patterns (per MetaFacade source):

**1-arg** (3 patterns): `fn(N)` / `fn of N` / `[what is] fn N`.

**2-arg** (4 patterns):
- `fn(A, B)`
- `fn of A and B`
- `A fn B` — for verb-like names ("choose", "permute")
- `fn A by|with|taken B`

User supplies: `fn_name` (must exist in safe_eval), `arity` ∈ {1, 2},
optional `domain_name` / `module_name` / `max_operand` / `extra_patterns`.

`MetaFacade.batch_from_oracles(list_of_dicts)` generates a list of
FacadeSpecs in one call — demo at `scripts/m2a_metafacade_demo.py`.

**Scope limits** (what Level 2 does NOT yet handle):
- Higher arities (≥3 args) — template library doesn't cover
- Non-regex parsing — e.g. ISO dates, structured strings
- Non-integer outputs — float, tuple, boolean (e.g. `is_prime` bool).
  `sqrt` returns float but falls back via the existing
  `val.is_integer()` check in the generated `evaluate()`.
- Domain-specific idioms — e.g. "N choose K" needs to be supplied
  via `extra_patterns` for now.

All of these are the natural Level-3 expansion scope.

## Level 3 — substrate designs new spec templates (FUTURE)

A **MetaMetaFacade** observes MetaFacade failure modes (e.g. misses
non-regex parsers, arity=3, non-integer outputs) and proposes new
template patterns. Not shipped — current Level-2 template library is
the upper bound of the present system.

When Level 3 ships, the pipeline becomes:

```
Gemma fails at domain X (CALM verifier catches)
    ↓
MetaMetaFacade proposes: (fn_name, arity, output_type, parser_family)
    ↓
MetaFacade synthesizes FacadeSpec from the proposal
    ↓
Level-1 pipeline takes it from there
```

Same CALM gates apply at every level — correctness pushed to the
deterministic oracle, not the LLM-as-judge.

## Why this is safe where Self-Instruct / RLAIF fails

| Approach | Oracle | Failure mode |
|---|---|---|
| Self-Instruct (Wang 2022) | the generating model itself | amplifies biases, reinforces hallucinations (student ≈ teacher) |
| RLAIF / constitutional AI | judge LLM | judge bias leaks into student |
| Evol-Instruct | LLM scoring | same bias amplification |
| **Substrate card recursion** | **deterministic CALM tests + compiled verification** | **cannot amplify what's verified wrong** |

Every card in the recursion chain is gated by running its output
against tests (or against compiled verifiers for specific domains).
Whatever survives has **passed objective correctness checks**, not
"looked good to another LLM". Drift-free on compiled domains.

For open-ended creative tasks with no verifiable correctness — no
card is trained, Gemma's probabilistic output is preserved (Tier 1).
The system never claims more than it can prove.

## Capability completeness as a fixed point

As recursion continues at all three levels:

- Card library grows (more domains covered)
- Each card covers more of its domain (self-distill fills gaps)
- MetaCard gets better at spotting which domains need cards
- MetaMetaCard gets better at designing MetaCard variants

Asymptotically: **for every task with a verifiable success criterion,
the substrate has a card that solves it exactly.** Tasks without
verifiable criteria fall to Gemma's native output with CALM claim-
verification on anything factual.

This is a different performance profile from scaling a monolithic
model:

- Monolithic scaling: statistical average improves, tail failures
  persist (including arithmetic errors at 100B+)
- Substrate recursion: compiled domains become **provably correct**;
  domains without oracles stay probabilistic but with per-claim
  verification overlay

The commercial positioning follows: regulated industries want the
"provably correct on compiled domains" half. General-purpose gets
the probabilistic half with claim verification. Same substrate,
different card stack per customer.

## Concrete state

### What's shipped

- `calm/llm_computer/auto_upgrade.py` — fact-level recursion
  (`AutoUpgradeEngine.commit()` compiles corrections into recall card)
- `calm/llm_computer/persistent_knowledge.py` — `KnowledgeStore` with
  `add_correction(key, value)` + `build_recall_model()` + save/load
- `scripts/gemma_learning_loop_demo.py` — fact-level demo
- `calm/llm_computer/recursion.py` — Level-1 generator + Level-2 MetaFacade
- **Level-1 shipped facades** (`*_auto.py` in `calm/llm_computer/facades/`):
  `factorial_auto`, `fibonacci_auto`, `combinations_auto`,
  `permutations_auto`, `power_auto`, `next_prime_auto`
- **Level-2 shipped facades** (`*_meta.py`): `factorial_meta`,
  `combinations_meta`, `gcd_meta`, `lcm_meta`, `fibonacci_meta`

Shipped-facade dated inventory + eval file cross-refs:
`MEMORY/atlas/recursion_arc.md`.

### What's next (Level 3 + code DT)

- **Level 3 MetaMetaFacade**: observe MetaFacade failures (higher
  arity, non-regex parsers, non-integer outputs), propose new template
  families, synthesize MetaFacade variants. Current template library
  is the upper bound of the present system.
- **Code DT self-distill** (roadmap): train code-skeleton DT on DB,
  install via CardSlot, run `CodeVerifierFacade`-gated
  self-distillation loop. See `delta_rule.md` §"Code-skeleton recipe".
- **Commercial vertical decks**: use Level-2 MetaFacade to rapidly
  stand up hospital / legal / financial card decks (each is ~5-10
  domain-specific `FacadeSpec`s produced by hand + MetaFacade).
- **Closed loop**: CALM verifier catches Gemma failure → infers
  oracle signature → MetaFacade proposes spec → Level-1 pipeline
  ships the facade. Last missing link is the CALM → oracle-signature
  inference step.

## Related rules

- `augmentation_thesis.md` §"Auto-upgrade loop" — the factorial
  scaling property that makes recursion economically viable
- `Substrate.md` §"Auto-Upgrade Loop" — technical install path for
  compiled recall cards
- `calm.md` §"Auto-Upgrade Loop" — CALM's role as oracle
- `capability_gain.md` — measurement discipline per recursion step
- `probing_methodology.md` — circuit mapping (Level 2 input)
- `commercial.md` — verifiable-augmentation-as-product positioning
