# The LLM-Computer — Overview

The *what* and *why*. Motivation, demos, conceptual framing, caveats. No
math — that lives in [`02`](02_Fast_Attention_2D_Heads.md) and
[`03`](03_Compiling_Programs_to_Weights.md).

Based on two Percepta posts (March 2026): *Can LLMs Be Computers?* and
*Constructing an LLM-Computer*. See [`00_INDEX.md`](00_INDEX.md) for the
full doc set.

---

## 1. TL;DR

Language models reach Olympiad-grade reasoning but stumble on purely
computational tasks — multi-digit multiplication, Sudoku, graph algorithms.
The standard workaround is **tool use**: the model emits code, an external
interpreter runs it, the result is injected back into the token stream.
This works, but the computation happens *outside* the model.

Percepta shows a different path: **make the transformer itself a computer.**
Arbitrary C programs are compiled to WebAssembly bytecode, fed to a standard
autoregressive transformer as tokens, and executed step-by-step by the
transformer's forward pass — no external tool, no round-trip. Headline
numbers: a 10×10 Hungarian algorithm at **33,583 tok/s**, Arto Inkala's
"hardest Sudoku" solved in under three minutes via **4.99M tokens** of
compiled DFS.

Two technical pieces make it work:

1. **Exponentially Fast Attention.** Restricting attention heads to 2D
   turns "which past key scores highest?" into a classic computational-
   geometry problem (supporting point on a convex hull), solvable in
   `O(log t)` per step instead of `Θ(t)`. See
   [`02`](02_Fast_Attention_2D_Heads.md).
2. **Analytical compilation of programs into weights.** A small abstract
   machine (ALM) and DSL (CALM) define primitives that transformers
   realize exactly. A MILP-based compiler schedules the resulting gate
   graph into transformer layers. A WebAssembly interpreter written in
   CALM turns "transformer = weights that execute WASM" into a concrete
   build step. See [`03`](03_Compiling_Programs_to_Weights.md).

This is a possibility proof, not a production system. The prototype is
orders of magnitude slower than a native CPU and handles a subset of
WebAssembly. The interesting question is where the pipeline leads — formal
verification of model logic, surgical weight editing, hybrid compiled/
trained models — not whether today's implementation is fast enough to ship.

---

## 2. Motivation: LLMs cannot compute

State-of-the-art language models reach gold-medal standard on the
International Mathematical Olympiad and make real progress on open
problems in math and physics. The same models make arithmetic mistakes on
five-digit multiplications and fail to solve modestly hard Sudokus.
Benchmarks like Sudoku-Bench consistently report low unaided solve rates.

The industry's standard response is to bolt execution on from the outside:

- **Tool use.** The model emits a code block; generation pauses; an
  external interpreter runs the code; the result is injected back. This
  is what every frontier assistant does for arithmetic and anything
  numerically exact.
- **Agentic orchestration.** An outer loop stores intermediate state,
  splits a task into smaller prompts, drives the model through them —
  effectively wrapping the model in a state machine.

Both work, and both highlight the same gap: **the model cannot do the
computation itself.** It can describe an algorithm and invoke a runtime
that implements it, but cannot carry the computation through on its own
forward pass. Every exact-compute step has to leave the model and come
back.

Percepta's analogy: humans cannot fly. Building airplanes doesn't change
that; it just means we have a machine that flies for us. LLMs today are
in that position with respect to computation — they coordinate external
machines that compute on their behalf.

The question this work asks: can a model internalize computation to the
point where it *is* the machine?

### Why "talk about algorithms" isn't enough

There's a theoretical result that transformers are Turing-complete: given
enough width, depth, and steps, they can simulate any computer. So the
capability is, in principle, there. The catch is that these universality
constructions map a single real-machine instruction onto a long sequence
of transformer steps — the simulation works but is wildly impractical.
**Universality is not practicality.**

This work's contribution is not a new universality result. It's an
efficient *practical* construction: a RAM-style computer where each
instruction is at most five tokens, paired with a decoding scheme that
keeps per-step cost logarithmic. The combination is what makes
multi-million-step execution feasible.

---

## 3. In-model execution vs. tool use

To compute `3 + 5`:

### Tool use (today's frontier assistants)

The model outputs a Python snippet:

~~~python
print(3 + 5)
~~~

Generation pauses. An external sandbox runs the code. The result `8` is
injected back into the token stream. The model resumes: "The answer is 8."

### In-model execution (this work)

The model outputs the WASM:

~~~
i32.const 03 00 00 00
i32.const 05 00 00 00
i32.add   00 00 00 00
output    00 00 00 00
~~~

Then — without any pause, without any external call — it produces the
execution trace itself:

~~~
03 00 00 00  commit(+1, sts=1, bt=0)
05 00 00 00  commit(+1, sts=1, bt=0)
08 00 00 00  commit(-1, sts=1, bt=0)
out(08)
halt
~~~

And resumes: "The answer is 8."

### What's different

The two look similar from outside; the difference is whose CPU produced
the `8`. In tool use it's the host machine's Python interpreter. In
in-model execution it's the transformer's own forward pass — every stack
push (`+1`), pop (`-1`), and output byte is a token the model itself
generates.

Two consequences matter:

- **Transparency.** Every intermediate step appears in the trace. Tool
  use is opaque: hand off, receive answer. In-model execution makes
  debugging, verification, and gradient flow through computation all
  tractable.
- **No round-trip.** No external process, no IPC, no sandbox handshake.
  Execution runs at the model's native token-emission rate.

---

## 4. Demos

Two long-horizon problems show the pipeline end-to-end.

### 10×10 min-cost perfect matching (Hungarian algorithm)

Given a 10×10 cost matrix, find the row → column assignment that
minimizes total cost. The Hungarian algorithm is the standard polynomial
solution — non-trivial control flow with dual variable updates and
augmenting paths.

The model receives the matrix, emits the compiled program, and streams
the execution trace: one row assigned per iteration with explicit
"running Dijkstra on reduced costs", "found augmenting path", "move row
X from col A to col B" events. Final: optimal cost 206, full assignment.

**Numbers:** 439,194 tokens at **33,583 tok/s** (~7,301 output lines/s).
No tool calls.

### Arto Inkala's Sudoku

Inkala's 2012 puzzle was constructed to be as hard as possible for human
solvers and is routinely used as a worst-case input for Sudoku solvers.
Learned neural approaches famously fail on it — the usual diagnosis is
that autoregressive models commit token-by-token and cannot revise early
wrong guesses, making them structurally unsuited to constraint
satisfaction.

The counter-argument Percepta offers: the autoregressive *paradigm* isn't
the bottleneck; the *cost* of long autoregressive generation is. A
compiled backtracking solver revises fine — it just needs millions of
steps, which standard attention makes prohibitively expensive. With the
fast decoding path, those steps become cheap.

The transformer executes a compiled DFS Sudoku solver with "Trying X at
row R, col C", "Contradiction", "Undoing row R col C. Going back up"
events in its trace. Inkala's puzzle solves correctly in under three
minutes.

**Numbers:** 4,994,876 tokens at **31,361 tok/s** (~6,908 lines/s).

The correctness guarantee is important: because the compiled solver is
provably correct, the *transformer's* execution is provably correct.
There's no gap between "model proposed" and "external verifier checked"
— verification is implicit in the fact that the model is running the
verified algorithm.

---

## 5. Computation as an append-only trace

The mental model that makes the construction make sense.

Think of a transformer as a machine living inside its own history. A
traditional computer has editable memory — registers and RAM are
overwritten in place. A transformer has no such thing. It has:

- **A fixed prompt** (input and program).
- **A trace that only grows** (tokens generated so far).
- **Attention heads** that look back at a small number of earlier
  positions.
- **An append operation** (emit the next token).

That's it. No in-place edits. And yet most of what a CPU does can be
expressed in this append-only form, as long as each step needs to consult
only a bounded number of earlier entries.

Toy example: count whether a sentence has an odd or even number of verbs.
Each trace token attends to two positions — the corresponding word in the
prompt (is it a verb?) and the previous trace token (running parity?).
Two look-backs per step, regardless of sentence length.

This generalizes. An interpreter's state — instruction pointer, top of
stack, memory cells being read or written, branch-taken flags — can be
encoded in recent trace tokens. A new execution step reads a few
specific earlier positions (instruction, operands, memory address),
combines them, and appends the result.

The model doesn't store the entire memory of the simulated machine as
live state. **The memory is the trace.** When the program wants to read
address `0x400`, it emits a query that attends back to wherever in the
trace `0x400` was most recently written. Attention is the memory-access
primitive.

What makes this work in practice is that lookups are *structured* — not
scanning history for arbitrary patterns but asking precise, geometric
questions. Which leads to the technical core — see
[`02`](02_Fast_Attention_2D_Heads.md) for how those geometric queries
become logarithmic-time.

---

## 6. Relation to tool use, CALM, and other "make LLMs reliable" work

Three families of work address the same underlying problem.

**Tool use / function calling.** Model emits code; external runtime
executes; result injected. Strengths: works today, off-the-shelf
runtimes, arbitrary languages. Weaknesses: opaque to gradient flow,
round-trip overhead, computation lives *outside* the model.

**Compute-augmented language models (e.g., CALM — this project's own
verification engine).** Model generates naturally; a parallel
verification engine extracts claims, re-computes on CPU, corrects the
output. Strengths: works on existing models without retraining,
deterministic verification of factual and numerical claims, cross-checks
against modular backends (1002 verified functions across 116 domains at
last count). Weaknesses: catches errors *after* generation; verifier is
a separate system, not part of the forward pass.

**In-model execution (Percepta's work).** Computation happens inside the
forward pass as part of the model's own token stream. Strengths: fully
differentiable, no round-trip, provably correct execution when the
compiled program is correct. Weaknesses: 2D-head architectural
constraint is new territory; training-at-scale story is unproven;
executor is currently a separate artifact from a general language model.

The three are complementary. Tool use is pragmatic and works today.
Verification frameworks like CALM catch errors in models you can't
retrain. In-model execution is a research direction that, if it pans
out, changes what a "model" fundamentally is.

---

## 7. What this enables

Pushing the pipeline gives several concrete capabilities, some of which
are impossible in the tool-use paradigm.

**Weights as a deployment target for software.** A compiled algorithm
and a trained model become the same kind of artifact: a weight tensor
plus a runtime. Shipping a Sudoku solver and shipping a language model
differ in how the weights were generated (compilation vs. training), not
in what they are.

**Analytical correctness.** Because weights are compiled from a gate
graph that is exact, the transformer's execution is correct by
construction. Given a program known to be correct, the transformer's
output is *provably* correct — not "low-loss on the eval set", but
theorem-grade correct.

**Formal verification of model logic.** Once there's an IR (gate graph)
with well-defined semantics, you can write specifications, prove the
gate graph implements them, and inherit the proof through mechanical
compilation to weights.

**Surgical weight editing.** Gradient descent modifies weights globally
and opaquely. This pipeline modifies weights **locally and
transparently** — you know which weights implement which gate, so you
can rewrite a single gate's weights to change a single piece of logic.
Fine-tuning without data; compilation as a targeted editor.

**Hybrid with trained models.** Nothing prevents merging compiled and
trained weights in the same model. A large learned LM could include a
compiled 2D-head executor block, trained end-to-end to *dispatch to* the
executor when exact computation is needed. Gradients flow through the
whole thing — unlike tool use.

---

## 8. Caveats

Explicit in the Percepta paper.

**WASM subset, not full WASM.** The interpreter handles a subset of
WebAssembly. More complex operations are lowered to supported primitives,
not always efficiently. Full coverage is engineering, not research.

**Memory grows with tokens.** Because the trace *is* the memory, programs
that use a lot of memory produce long traces. A program reading 1 GB of
data has to emit 1 GB worth of tokens somewhere. Hybrid architectures
(standard memory subsystem + 2D-head compute subsystem) probably become
necessary for realistic workloads.

**Orders of magnitude slower than native.** The construction is
simplified and unoptimized. Very much slower than a conventional
computer at the same computation. Possibility proof, not production
runtime.

**Training at scale is unproven.** Weights are compiled, not trained.
Whether 2D-head models can be *trained* to competitive general-purpose
capability remains open — see [`02`](02_Fast_Attention_2D_Heads.md) §9.
This is the decisive open question.

---

## 9. Directions & closing thoughts

The paper names four active investigations:

1. **Robust toolkits for programming transformer weights.** CALM is a
   proof of concept; a mature toolkit would look like LLVM for weights —
   multiple front-end languages, optimization passes, standard libraries,
   profile-guided specialization.
2. **Formal verification of transformer logic.** Gate-graph semantics +
   mechanical weight compilation = provable guarantees about model
   behavior.
3. **Faster constructions and specialized attention.** HullKVCache is
   one acceleration; presumably others exist specific to particular gate
   patterns.
4. **Programmatic logic in the training loop.** Compiled sub-networks as
   architectural priors; compilation as a final fine-tuning stage.

The underlying bet: the hardest real problems — sequential decision-
making under uncertainty in healthcare, supply chains, finance — need
systems that both reason flexibly and compute reliably in the same
substrate. In-model execution is one plausible path there.

If the training story works out, this opens a future where AI systems
don't just *use* software — they *contain* it, as compiled logic woven
into weights alongside learned representations. In that world, the
distinction between "running an algorithm" and "running a model" starts
to blur.
