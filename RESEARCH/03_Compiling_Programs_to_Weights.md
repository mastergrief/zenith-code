# Compiling Programs to Weights

The construction. An abstract machine (ALM), a language over it (CALM),
an intermediate representation (gate graph), and an analytical compiler
that produces transformer weights directly — no training loop involved.

**Concept-owner for:** ALM and its five primitives; the CALM language;
the primitive-to-transformer-mechanism correspondences; exact
realizations of each primitive (cumsum, step functions, products); the
gate-graph IR; the WASM interpreter fragment; MILP scheduling;
interval-coloring slot reuse; the additive-residual-stream subtlety;
partial evaluation / program specialization; PL implications beyond
specialization.

This doc does **not** re-derive the 2D exact-lookup algebra — that lives
in [`02`](02_Fast_Attention_2D_Heads.md) §5. Referenced here, not
duplicated.

See also: [`01`](01_LLM_Computer_Overview.md) for motivation and demos,
[`00_INDEX.md`](00_INDEX.md) for the full doc set.

---

## 1. The ALM abstract machine

Transformers can express complex behavior through attention and FFNs,
but to **program them directly** you need a small set of primitives that
are (a) expressive enough for general computation and (b) native enough
to compile exactly into weights.

The authors work at three levels of abstraction:

| Level | Name | What it is |
|---|---|---|
| Abstract machine | **ALM** (Append-only Lookup Machine) | Five primitives, Turing-complete, transformer-native |
| Language | **CALM** (Code for ALMs) | A small DSL over ALM |
| IR | **Gate graph** | DAG of LookUp + ReGLU gates, wired with linear ops |

### The five primitives

**1. Read / Write (keyed memory).** Store a value under a key in some
channel `c`, retrieve it later by querying that key:

    writeₓ(k, v)
    readₓ(q)

Hard constraint: a channel must be written *before* it can be read —
this is the "append-only" in "Append-only Lookup Machine." No in-place
updates, just a growing log that earlier positions write to and later
positions read from.

**2. Cumulative sum.** Running total of some value across the sequence:

    cumsumₓ(v)

How you maintain counters — instruction pointer, stack depth, call-stack
depth — across time in a world where you cannot mutate state.

**3. Product.** Multiply two values: `a · b`. The one genuinely
nonlinear arithmetic primitive. Everything else arithmetic reduces to
products plus linear combinations.

**4. Conditional.** Choose between two values based on a condition:
`if a then b else c`. The control-flow primitive.

**5. Linear combination.** Any `c₁·x + c₂·y` for constant `c₁, c₂`.
Free in a transformer because the residual stream is additive — it
costs no nonlinear gate.

These five, properly combined, are enough to simulate any Turing
machine. The paper proves ALM Turing-completeness. Intuition: read/write
+ cumsum give tape and head position; products + conditionals give
branching and arithmetic; linear combinations give wiring.

---

## 2. CALM — the language

**CALM** (Code for Append-only Lookup Machines) is a DSL over ALM.
Programs written in CALM compile to a **gate graph** — a DAG where every
node is either a LookUp gate (attention) or a ReGLU gate (feed-forward),
connected by linear combinations.

The gate graph is the *key intermediate representation* of the compiler.
A CALM program at this stage is fully mechanical: no ambiguity, no
learned approximation, just a computation DAG with exact semantics.

---

## 3. Primitive → transformer mechanism

Each ALM primitive maps to exactly one transformer component:

| ALM primitive | Transformer mechanism |
|---|---|
| read / write (keyed memory) | Attention head with 2D keys |
| cumulative sum | Attention head with uniform keys |
| product `a · b` | ReGLU neuron in FFN |
| conditional `if C then u else v` | ReGLU with 0/1 indicator |
| linear combination | Residual-stream wiring (no gate needed) |

Attention handles anything that reaches into the past. Feed-forward
handles anything nonlinear that can be computed locally from current
residual state. The residual stream itself handles wiring.

**Every ALM primitive falls cleanly into exactly one bucket.** This is
why this abstraction is the right one for programming transformer
weights — it matches the architecture's natural seams.

---

## 4. Exact realizations

Each primitive gets an **exact** mathematical realization, not an
approximation that happens to work most of the time.

### 4a. Keyed lookup via parabolic keys

The central operation is `read(q)` — retrieve the value stored under key
`q` from somewhere earlier in the sequence. The attention head has to
*exactly* select the right past position.

The construction uses the parabolic-keys trick:

    kⱼ = (2j, −j²),   q = (i, 1)   ⟹   argmax_j (q · kⱼ) = i

Full derivation in [`02`](02_Fast_Attention_2D_Heads.md) §5
(completing-the-square on `2ij − j²`). This doc just uses the result: a
2D hard-max head with `kⱼ = (2k, −k²)` returns exactly the value stored
under key `k` when queried with `(k, 1)`.

This is the deep reason 2D heads are special in the construction. Two
dimensions are the **minimum** needed for a uniquely-maximized keyed
lookup (one dimension gives you only monotone selection), and two
dimensions are also what the HullKVCache needs to decode in `O(log t)`.
**The efficient-inference story and the exact-lookup story are the
same story.**

**Latest-write semantics.** Hard-max attention averages when keys tie.
If position 10 wrote `k=5, v=A` and position 50 wrote `k=5, v=B`, a
naïve lookup of `k=5` returns `(A + B) / 2` — not what "read the latest
value at key 5" should mean. The fix is a small position-dependent
perturbation added to each key, so among entries with the same logical
key the latest one scores strictly highest. Exact latest-write memory,
at the cost of one tiny extra term.

### 4b. Cumulative sums from uniform-key attention

Same attention mechanism, different key pattern. If every position
writes the *same* attention key, there's no selective retrieval — the
head returns the uniform average of all values in the prefix:

    attention output = (1 / (t+1)) · Σᵢ≤ₜ vᵢ

The current position `t` is available from the positional embedding.
Scaling the average by `t + 1` recovers the exact cumulative sum:

    Σᵢ≤ₜ vᵢ = (t + 1) · (1 / (t+1)) · Σᵢ≤ₜ vᵢ

This is how the interpreter maintains the **instruction pointer**,
**stack depth**, and **call-stack depth** exactly across millions of
steps. Each is a `cumsum(delta)` of a small per-step delta (`-1`, `0`,
or `+1`).

### 4c. Step functions from ReLU

The conditional `if a then b else c` needs exact `0/1` indicators on
integer inputs. For integer-valued `z`:

    1[z ≥ 0] = ReLU(z + 1) − ReLU(z)

Walking through it:

- If `z ≥ 0`: `ReLU(z+1) = z+1`, `ReLU(z) = z`, difference is `1`. ✓
- If `z ≤ −1`: `ReLU(z+1) = 0` (since `z+1 ≤ 0`), `ReLU(z) = 0`,
  difference is `0`. ✓

Because inputs are guaranteed integer, a continuous activation produces
exact discrete logic. Everything else in the indicator zoo follows
mechanically:

    1[x = c]     = 1[x − c ≥ 0] − 1[x − c − 1 ≥ 0]
    1[x ≤ c]     = 1 − 1[x − c − 1 ≥ 0]
    1[a ≤ x ≤ b] = 1[x − a ≥ 0] − 1[x − b − 1 ≥ 0]

And the conditional itself:

    if C then u else v  =  u · 1[C] + v · (1 − 1[C])

Two ReLU evaluations plus linear wiring = exact integer branching.

### 4d. Products from ReGLU

A gated FFN natively computes `a · ReLU(b)` — the ReGLU pattern. For
non-negative integer `b`, `ReLU(b) = b`, so `a · ReLU(b) = a · b`. For
signed integers:

    a · b = a · ReLU(b) − a · ReLU(−b)

Two ReGLU neurons reproduce the full signed product. In the paper's
terminology, these are **ReGLU gates**.

---

## 5. The gate graph IR

Every CALM program expands into a DAG of two gate families connected by
linear wiring:

- **LookUp gates** (attention): exact keyed lookup + cumulative sums.
- **ReGLU gates** (feed-forward): indicators, products, conditionals.
- **Residual-stream wiring**: linear combinations, free.

The gate graph says exactly what computation happens at each token
position during autoregressive generation — no ambiguity. Here's a
simplified fragment from the WASM interpreter:

~~~
stack_depth  = CumSum(delta_stack)
cursor       = CumSum(delta_cursor)

opcode, immediate = LookUp(cursor)           # instruction fetch
first_value       = LookUp(stack_depth)      # top of stack
second_value      = LookUp(stack_depth − 1)  # next on stack

add_value  = first_value + second_value + carry
mem_byte   = LookUp(immediate + first_value) # for i32.load

gated_add  = ReGLU(add_value, is_op(opcode, "i32.add"))
gated_load = ReGLU(mem_byte,  is_op(opcode, "i32.load"))

result     = gated_add + gated_load + …      # dispatch by opcode
~~~

Read top to bottom:

1. Two cumulative sums maintain the interpreter's running state
   (cursor = program counter, stack depth).
2. Three LookUps fetch the current instruction and the top two stack
   values, using those counters as keys.
3. A ReLU-indicator-based `is_op(opcode, "…")` check activates exactly
   one branch of the giant opcode switch.
4. Gated sums collapse into a single `result`, written back to the
   residual stream.

The full WebAssembly interpreter is a much larger version of this — one
gate graph with one branch per opcode, dispatched by indicators. The
size of the switch is what makes the interpreter's gate graph nontrivial
to schedule into layers (next section).

---

## 6. MILP scheduling — from gate graph to layers

A gate graph is exact but infinite-width: conceptually every value lives
in its own variable. A real transformer has **finite depth** (`L`
layers) and **finite width** (`d_model` for the residual stream, `d_ffn`
for each FFN). The compiler has to pack the gate graph into those
dimensions.

Four concrete questions:

1. Which gate goes in which **layer**?
2. Which value lives in which **residual-stream slot**?
3. How many **layers** are needed overall?
4. How **wide** does the residual stream have to be?

### Layer structure

Each transformer layer provides four phases in fixed order:

| Phase | What happens |
|---|---|
| Attention | LookUp gates read from earlier positions |
| Materialization | Attention outputs written to residual stream |
| Feed-forward | ReGLU gates do local nonlinear logic |
| Materialization | FFN outputs written to residual stream |

A gate can only go in a phase after everything it depends on has been
computed. LookUp gates *must* go in attention phases. ReGLU gates
*must* go in FFN phases. Any value computed but not yet read by its
final consumer occupies a residual-stream slot in the meantime.

This is **scheduling + register allocation, simultaneously.** Depth is
bounded by the precedence graph's critical path. Width is bounded by
the peak number of simultaneously-live values.

### The integer program

The compiler expresses both jointly as a **Mixed-Integer Linear
Program**. Decision variables assign each gate to a specific
`(layer, phase)`. Constraints encode:

- **Precedence.** Every consumer of a value is scheduled strictly after
  the producer — respecting attention → FFN ordering within a layer and
  layer ordering across the stack.
- **Type compatibility.** LookUp gates only in attention phases; ReGLU
  gates only in FFN phases.
- **Co-location.** Operations that must share intermediate results
  (e.g., the two sides of `ReLU(z+1) − ReLU(z)`) forced into the same
  layer.

**Objective:** fit all gates within a fixed layer budget while
**minimizing the peak number of simultaneously live values** — because
that peak directly determines `d_model`. Narrower residual stream means
faster inference, smaller weights, everything better.

Using an off-the-shelf MILP solver is unusual in ML compilers (which
typically use greedy heuristics) but not surprising once you look at
the problem: a finite discrete allocation with hard constraints,
exactly what MILPs are for.

---

## 7. Interval coloring + the additive-stream subtlety

Once the schedule is fixed, every value has a **lifetime**: the
interval from when it's computed to when its last consumer reads it.
Slots on the residual stream can be reused across lifetimes that don't
overlap.

This is the classic **interval graph coloring problem**: values are
intervals, slots are colors, no two overlapping intervals get the same
color. It's polynomial-time solvable (greedy with sweepline), and the
compiler runs it to assign every value to a concrete dimension in the
embedding space.

### The additive-stream subtlety

A subtlety unique to transformers: **the residual stream is additive.**
Standard CPU registers are *overwritten* on write. The residual stream
*accumulates* — every attention and FFN output is added to what's
already there. When a slot is reused, the stale value must be
**explicitly subtracted** before the new value is written.

The compiler inserts clearing terms as part of the wiring, making the
slot appear to be "overwritten" from the gate graph's perspective. This
is mechanical but easy to overlook — it's the difference between
"allocation works on paper" and "compiler produces correct weights."

### The resulting model

Once schedule, slot assignment, and head packing are fixed, weight
matrices follow mechanically. The output is a **completely standard
transformer** — multi-head attention (with 2D heads), gated FFN,
residual stream, no custom operators. Everything exotic lives in the
*values* of the weights, not the *structure* of the network.

That portability is a real feature. You can run the resulting model in
any PyTorch / JAX / GGUF / vLLM / llama.cpp pipeline without
modification. The only change a serving stack needs is the HullKVCache
for the fast-path speedup (see
[`02`](02_Fast_Attention_2D_Heads.md) §4).

---

## 8. The WASM interpreter + analytical correctness

With CALM + MILP compiler in hand, the authors embed a **WebAssembly
interpreter** in CALM, compile it to a gate graph, and lower that to
weights. The result is the universal executor.

At inference time:

1. A C program (or any language with a WASM backend) is compiled to
   WASM with a standard toolchain (e.g., `clang --target=wasm32`).
2. The WASM bytecode is tokenized into the transformer's vocabulary.
3. That token stream goes into the prompt, alongside the program's
   input.
4. The transformer generates the execution trace token by token.

The interpreter is written so that **the correct next execution token
is always scored strictly highest** among all vocabulary items. Greedy
decoding therefore produces a guaranteed-correct execution trace — not
"almost always correct" like a trained model, but **analytically
exact**, verifiable by examining the weights.

### Hand-coded specialized transformers

The same pipeline works *without* WebAssembly: hand-write a specific
algorithm in CALM (an adder, a sorter, a Sudoku solver), compile it,
ship the weights. No interpreter overhead — just a transformer that
*is* that one algorithm.

Clean, but requires writing custom logic in a new language for each
target. The authors go one step further with program specialization.

---

## 9. Partial evaluation — baking programs into weights

In the universal executor, the interpreter lives in the weights and
the program lives in the prompt. Every execution step fetches the
current instruction from the prompt via attention. That's necessary
when the program is unknown until inference — but when the program is
known in advance, fetching it from the prompt every step is wasted
work.

Classical **partial evaluation** (specifically the **first Futamura
projection**) says: given an interpreter `interp(program, input)` and
a known `program`, you can specialize it into a dedicated executor
`interp_program(input)`. The static structure of `program` moves into
the generated code; only the dynamic input remains.

Applied here: **the instruction table moves from the prompt into the
feed-forward weights.**

### How specialization works

For a program with `N` instructions, the specializer builds `2N` shared
ReGLU neurons that compute step functions of the program counter:

    sᵢ(cursor) = 1[cursor ≥ i]    for i = 0, 1, …, N − 1

That's `N` indicators; `2N` ReGLU neurons because
`1[z ≥ 0] = ReLU(z+1) − ReLU(z)` takes two ReLUs.

Given these step functions, **every fetched field** of the current
instruction (opcode, immediate, operand, whatever the instruction
layout defines) becomes a **single linear combination** whose
coefficients are the fixed program:

    fetched_field(cursor) = c₀ + Σᵢ₌₁ᴺ⁻¹ (cᵢ − cᵢ₋₁) · 1[cursor ≥ i]

where `cᵢ` is the value of that field at instruction `i`. Intuition:
this is a telescoping sum. When `cursor = k`, indicators for `i ≤ k`
are `1` and the rest are `0`, so the sum collapses to `cₖ`. **The
program is now literally encoded as weights in the FFN.**

### Cost accounting

- `d_ffn` grows by `O(N)` — the `2N` new ReGLU neurons.
- `d_model` grows by only a small constant — the number of fetched
  fields materialized on the residual stream (opcode, immediate, etc.).

For small programs (hundreds of instructions), FFN grows modestly; for
large ones, a lot. But the entire program prefix disappears from the
prompt — for fixed programs the prompt is just *the input*. For long-
running programs that's a significant reduction in context length, and
therefore in every-step attention cost.

### What specialization changes

|  | Universal executor | Specialized executor |
|---|---|---|
| Program location | Prompt prefix | FFN weights |
| Instruction fetch | Attention into prompt | Shared ReGLU step functions |
| Prompt contents | `[program bytes] [input]` | `[input]` only |
| Interpreter logic | Unchanged across programs | Unchanged across inputs |

The token-by-token execution logic is **identical** in both cases. The
only thing that moves is *where the static instruction table lives.*

---

## 10. PL implications beyond specialization

Partial evaluation is a classical programming-languages technique for
specializing general-purpose interpreters into dedicated code. It's
been studied for decades on conventional programs. Here it's applied to
**neural network weights**, and it works for the same reason it works
on normal programs: the interpreter has a static part (instruction
dispatch table) and a dynamic part (input), and the static part can
always be folded into the generated artifact. What's new is that the
"generated artifact" is a weight tensor rather than machine code.

If partial evaluation applies to transformer weights, **other classical
PL techniques probably do too**:

- **Constant folding.** Precompute any subgraph whose inputs are known
  at compile time.
- **Dead-code elimination.** Prune gates whose outputs are never read.
- **Inlining.** Expand small CALM functions into their call sites.
- **Loop unrolling.** For known-bounded loops, unroll into straight-
  line gate graphs.
- **Supercompilation.** Aggressive specialization based on analysis of
  execution traces.

The bigger picture: once you have an IR with well-defined semantics
(gate graph), a compiler (MILP + interval coloring), and a target
(transformer weights), you have all the ingredients of a mature
compiler toolchain. What's missing is the ecosystem — multiple front
ends, optimization passes, standard libraries, profile-guided
specialization, debug tooling — and none of that is research. It's
engineering.

### Formal verification

Once an IR has well-defined semantics, you can do what PL people have
been doing for fifty years: write specifications, prove the gate graph
implements them, and inherit the proof through mechanical compilation
to weights.

"This transformer provably halts on all inputs." "This transformer's
output matches this Coq-verified reference implementation." That class
of guarantee is out of reach for trained models. It becomes routine
for compiled ones.

### Surgical weight editing

Gradient descent modifies weights globally and opaquely. This pipeline
modifies weights **locally and transparently** — you know which weights
implement which gate. You can rewrite a single gate's worth of weights
to change a single piece of logic. Fine-tuning without data;
compilation as a targeted editor.

### Hybrid compiled + trained models

Nothing prevents merging compiled and trained weights in the same
model. A large learned LM could include a compiled 2D-head executor
block, trained end-to-end to *dispatch to* the executor when exact
computation is needed. Gradients flow through the whole thing — unlike
tool use.

This is probably the highest-impact direction in the paper's roadmap:
compiled sub-networks as architectural priors; compilation as a final
fine-tuning stage that bolts on exact computation without sacrificing
learned capabilities.
