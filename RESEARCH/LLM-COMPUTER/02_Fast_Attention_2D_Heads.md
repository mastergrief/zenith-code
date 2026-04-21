# Fast Attention and 2D Heads

The inference-speedup mechanism. Why standard autoregressive decoding is
`Θ(t)` per step, why that breaks down for million-step execution traces,
and how restricting attention heads to 2D turns each lookup into a
logarithmic-time geometric query.

**Concept-owner for:** the quadratic-scaling problem, 2D heads, the
parabolic-key argmax-indexing trick, convex-hull supporting-point queries,
the HullKVCache data structure, hard-max vs. softmax, k-sparse softmax,
the "vanilla PyTorch" model observation, and the training-at-scale open
question.

See also: [`01`](01_LLM_Computer_Overview.md) for motivation and demos,
[`03`](03_Compiling_Programs_to_Weights.md) for how exact-keyed lookup is
used to build memory in the compilation pipeline.

---

## 1. The architectural mismatch

A real CPU runs a program by reading a small, bounded amount of state —
registers, a stack pointer, a memory word — and writing back an equally
small update. Per-instruction work is essentially constant, regardless of
how long the program has been running.

A transformer doing autoregressive decoding is the opposite. Each new
token interacts with the *entire history* via attention. KV caching saves
recomputing past projections, but doesn't remove the fundamental scaling:
at step `t` the query still has to score against a cache of size `t`.
**Per-step work is `Θ(t)`; total work over `t` steps is `Θ(t²)`.**

For a 5-million-token Sudoku trace, that quadratic blow-up is the
difference between "finishes in three minutes" and "finishes next year."
Any realistic in-model execution requires breaking the linear-per-step
scaling.

Flash attention, paged KV, speculative decoding — all the standard tricks
for fast inference — still respect this underlying scaling. They make the
per-step scan faster per byte; they don't change the fact that it's a
scan.

---

## 2. The geometric reformulation

The authors restrict attention heads to **2-dimensional** keys and
queries (`d_head = 2`). This sounds like a crippling limitation but turns
out to be surprisingly powerful (see §5 for why).

With 2D heads, attention becomes:

- Each past token contributes a **key** `kⱼ ∈ ℝ²` and a **value** `vⱼ`.
- The current step produces a **query** `q ∈ ℝ²`, which can be thought
  of as a **direction in the plane**.
- With **hard-max** attention, the head returns `v_{j*}` where
  `j* = argmax_j (q · kⱼ)`.

In plain geometric terms: **given a direction, find the past point that
sticks out furthest in that direction.**

This is one of the most classic problems in computational geometry: the
**supporting-point query on a convex hull.** The point that maximizes
`q · k` is always a vertex of the convex hull of the key set. Anything
*not* on the hull is dominated in every direction by some hull vertex,
so it can never be the argmax — interior points are irrelevant.

---

## 3. Convex-hull supporting-point queries

Decades of CS research have produced data structures that answer this
query in `O(log n)` time over `n` 2D points, supporting online insertion
of new points in `O(log n)` amortized time per insertion. Standard
constructions: balanced BSTs over hull vertices ordered by polar angle,
or kinetic/offline hull trees.

The geometric picture:

- The full set of past keys forms a point cloud in ℝ².
- Only hull vertices can ever be the argmax for any query direction.
- Interior points can be ignored entirely.
- A supporting-point query is essentially a binary search over hull
  vertices: starting from any vertex, walk in the direction that improves
  `q · k`. Convexity guarantees convergence in `O(log n)` steps.

So instead of scoring against all `t` past keys, the decoder scores
against `O(log t)` hull vertices and returns the best.

---

## 4. HullKVCache

The paper replaces the standard KV cache with a **HullKVCache** — a
dynamic convex hull built on top of the 2D keys as they stream in:

- When a new token is generated, its key is inserted into the hull.
- At query time, a supporting-point query in direction `q` returns the
  best-scoring past token in `O(log t)` rather than `Θ(t)`.

Reported head-to-head on a 9,580-line trace, same weights, same input:

| Decoder | Total tokens | Throughput | Wall clock |
|---|---:|---:|---:|
| HullKVCache | 41,709 | **31,037 tok/s** | 1.3s |
| Standard KV | 23,691 | 259 tok/s | 258.9s |

**~120× faster.** And the gap widens with trace length, because one
decoder is logarithmic per step and the other is linear.

### Where the speedup actually lives

The speedup compounds hardest on **deterministic, mechanical spans**:
copying bytes, stepping a state machine, running straight-line arithmetic,
dispatching opcodes. These are exactly the steps where you don't want to
burn full attention budget, and they're the bulk of any long execution
trace.

Abstract reasoning steps — where the model might legitimately need to
mix many past positions softly — can still use the conventional path.
The natural architecture is a **hybrid**: fast path for bulk execution,
regular path for reasoning.

---

## 5. Why 2D suffices — the parabolic-keys trick

The obvious objection: 2D heads feel like an enormous expressivity loss.
A single 2D head can only distinguish points up to a 2D projection. How
can that possibly be enough to, say, index a specific past position?

**The trick:** embed past position `j` as the 2D point

    kⱼ = (2j, −j²)

These points lie on a downward-opening parabola. To look up position
`i`, use query direction `q = (i, 1)`. The attention score becomes:

    score(q, kⱼ) = (i, 1) · (2j, −j²)
                 = 2ij − j²
                 = −(j − i)² + i²

The `i²` term is constant in `j`, so it doesn't affect the argmax. What
matters is `−(j − i)²`:

- When `j = i`, this is `0` — the maximum.
- For any other `j`, the quadratic penalty `−(j − i)²` drives the score
  strictly lower.

**A 2D hard-max head, queried with direction `(i, 1)`, returns exactly
position `i`.** No softmax smear, no approximate match, no interference
from nearby positions. Exact indexing from a 2D head.

### Turing-completeness falls out

With the ability to read *any specific past position* plus a bounded
amount of local computation per step plus an append-only output, you
have enough to simulate any Turing machine. **2D is sufficient for
universality.** Higher-dimensional keys are conveniences; 2D is the
floor.

### Parameter count is preserved

"Head dimension = 2" does **not** mean "tiny model." The paper's example
transformer is `d_model = 36, n_heads = 18, n_layers = 7` — each head is
2D, there are 18 per layer. Total parameter count scales the usual way;
you just have more, smaller heads: `n_heads = d_model / 2`.

In principle you can match a standard transformer's parameter budget by
scaling heads and layers. The open empirical question — flagged
explicitly in the paper — is how well such models **train at scale** on
general tasks (see §9). These weights were compiled rather than learned,
so the training story is still to be written.

---

## 6. Hard-max vs. softmax; k-sparse softmax

The convex-hull trick accelerates **argmax** attention, not the weighted
softmax used in standard transformers. Three practical points.

**Softmax approximates hard-max.** Multiplying scores by a large constant
before softmax sharpens the distribution arbitrarily close to argmax.
Standard softmax transformers can realize the same exact-lookup
construction with exponentially small approximation error. Hard-max is
a cleaner analytical model but not a required architectural choice.

**k-sparse softmax via nested hulls.** Approximate full softmax by
retrieving the top-`k` highest-scoring keys and doing an exact softmax
over just those. Implementation: peel off the outermost convex hull,
remove those vertices from the point set, query the next hull, and so
on. Each peel costs `O(log n)`; `k` peels plus a final local softmax
gives per-step cost `O(k + log n)`. Dense softmax at scale over the full
prefix remains open; k-sparse already covers most practical needs.

**For execution traces, argmax is usually enough.** Deterministic,
mechanical spans — the bulk of any compiled program — are precisely the
steps where argmax is the correct attention operation. The softmax
question matters more when porting the decoding speedup to general
language models.

---

## 7. 3D and beyond

The same machinery extends to 3D heads via 3D convex hulls, though
per-query cost grows and the data structures get more delicate. Beyond
3D, the geometric advantage erodes quickly: convex-hull complexity grows
rapidly with dimension, and the nice "log-time binary search" becomes a
more expensive recursive structure.

The natural hypothesis: 2D already captures most of the benefit; 3D
might be worth the engineering cost for specific workloads where
exact-keyed lookup alone isn't enough but softer similarity is useful.

---

## 8. The deliberately boring PyTorch model

The full model class from the paper:

~~~python
class VanillaTransformer(nn.Module):
    def __init__(self, vocab, d_model=36, n_heads=18, n_layers=7, d_ffn=36):
        super().__init__()
        self.tok = nn.Embedding(vocab, d_model)
        self.attn = nn.ModuleList([
            nn.MultiheadAttention(d_model, n_heads, batch_first=True, bias=False)
            for _ in range(n_layers)
        ])
        self.ff_in  = nn.ModuleList([nn.Linear(d_model, 2*d_ffn, bias=False) for _ in range(n_layers)])
        self.ff_out = nn.ModuleList([nn.Linear(d_ffn, d_model, bias=False) for _ in range(n_layers)])
        self.head = nn.Linear(d_model, vocab, bias=False)

    def forward(self, idx):
        T = idx.shape[1]
        x = self.tok(idx) + pos_emb(T)
        causal = torch.triu(torch.ones(T, T, device=idx.device, dtype=torch.bool), diagonal=1)
        for attn, ff_in, ff_out in zip(self.attn, self.ff_in, self.ff_out):
            y, _ = attn(x, x, x, attn_mask=causal, need_weights=False)
            x = x + y
            gate, val = ff_in(x).chunk(2, dim=-1)
            x = x + ff_out(F.relu(gate) * val)
        return self.head(x)
~~~

Observations:

- `d_model=36` with `n_heads=18` gives `d_head = 2`. That's the one
  structural constraint.
- Everything else is vanilla: `nn.MultiheadAttention`, a gated FFN with
  ReLU, causal masking, learned positional embeddings.
- No custom CUDA kernels, no sparse masks, no MoE, no state-space layers.

What makes it compute? **The weights.** The authors have a compiler that
produces weights implementing a WASM interpreter. The transformer is a
general-purpose computer *because of what its weights encode*, not
because of what its architecture *is*.

This is a significant shift in what a model is. In a trained transformer,
weights are a statistical summary of training data. In this construction,
weights are **compiled source code**. See
[`03`](03_Compiling_Programs_to_Weights.md) for how the compilation works.

The architectural portability matters: you can run the resulting model in
any PyTorch / JAX / GGUF / vLLM / llama.cpp pipeline without modification.
The only change a serving stack needs is the HullKVCache for the fast-
path speedup; the raw forward pass already works in any standard runtime.

---

## 9. Open: training 2D-head models at scale

The authors *compiled* their executor's weights; they did not demonstrate
that 2D-head transformers reach competitive capability when **trained
from scratch** on general text. That is the decisive open question.

Three plausible outcomes:

1. **Useful-but-specialized.** 2D-head models become strong *executors*
   but can't match standard transformers on open-ended language tasks.
   They live as a fast path paired with a conventional model — a
   coprocessor, or a speculative decoder (2D model proposes quickly,
   standard model verifies and accepts).
2. **Competitive via scale.** More heads and more layers recover most
   of the capability lost to the 2D restriction. 2D-head models become
   a Pareto-better point on the speed/capability curve.
3. **Uncompetitive.** The 2D restriction imposes a real capability
   ceiling that no amount of scale closes. Likely for pure language
   modelling, though even then the hybrid story still works.

Any of these makes the work useful; (2) would be transformative.

Independent of the answer, the speedups from HullKVCache alone justify
the 2D-head direction as an **inference-acceleration research agenda** —
even if 2D-head models never become a competitive training target,
retrofitting existing softmax models with 2D-head layers (trained or
distilled) to handle specific long-horizon workloads is a viable
application path.
