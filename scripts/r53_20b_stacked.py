"""R53.20b — Stacked substrate + prompt-RAG + structured repair.

The thesis test: does substrate install add measurable value ON TOP of
the R53.19 v3 stack (channel-code-hybrid hints + structured failure
categorizer + 3-attempt repair), which landed 26/26 post-SWA-fix?

Stack (top to bottom):

  Prompt          channel-code-hybrid retrieval (R53.7)
  Substrate       KnowledgeStore@L41 + per-marker FirstTokenHook (R53.14)
  Repair          structured failure categorizer + targeted retry (R53.19)
  Gemma           prod Gemma 4 E4B tq4 with SWA fix

Substrate is hash-gated — only fires on the 6 enrolled problem hashes.
First-token hook re-armed before every generation (initial + retries).

Hypothesis: if substrate-RAG has structural value (automatic Tier-1
preservation), the combined stack produces MORE wins on the ceiling
problems (csv_column_stats, token_bucket) than R53.19 v3's 26/26.

Counter-hypothesis: substrate CardSlot residual write on hits may
disrupt hinted generation, producing fewer wins than R53.19 v3.

Daemon-only:
  bin/gemma-run scripts/r53_20b_stacked.py
"""

from __future__ import annotations

import hashlib
import random
import re
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import torch


CACHE_DIR = "/mnt/c/Users/gabes/projects/claw-code/.cache/r53_code_db"

MAX_ATTEMPTS = 3
MAX_TOKENS = 400

RECALL_CH_OFF = 2480
MAX_KEY = 4096
MAX_VALUE = 16
RECALL_D_CARD = MAX_VALUE + 1
INSTALL_LAYER = 41
HOOK_BOOST = 50.0
HOOK_MIN_MARGIN = 0.5

# Per-marker → first-token target (from R53.14)
PER_MARKER_TARGETS = {
    1: "class",   # linked_list_bugs
    2: "def",     # date_validation_chain
    3: "def",     # log_level_counts
    4: "def",     # csv_column_stats
    5: "class",   # token_bucket_rate_limiter
    6: "class",   # lru_cache_class
}

COMMON_IMPORTS = {
    "StringIO": "from io import StringIO",
    "BytesIO": "from io import BytesIO",
    "Dict": "from typing import Dict",
    "List": "from typing import List",
    "Optional": "from typing import Optional",
    "Tuple": "from typing import Tuple",
    "Any": "from typing import Any",
    "Callable": "from typing import Callable",
    "Counter": "from collections import Counter",
    "defaultdict": "from collections import defaultdict",
    "OrderedDict": "from collections import OrderedDict",
    "deque": "from collections import deque",
    "datetime": "from datetime import datetime",
    "timedelta": "from datetime import timedelta",
    "date": "from datetime import date",
    "time": "import time",
    "math": "import math",
    "json": "import json",
    "re": "import re",
    "os": "import os",
    "sys": "import sys",
    "csv": "import csv",
    "statistics": "import statistics",
    "mean": "from statistics import mean",
    "stdev": "from statistics import stdev",
    "Path": "from pathlib import Path",
}


def hash_prompt(prompt: str, max_key: int = MAX_KEY) -> int:
    h = hashlib.blake2b(prompt.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "big") % max_key


def find_token_id(tok, target_text: str) -> int:
    candidates = [f"\u2581{target_text}", target_text, f" {target_text}"]
    for cand in candidates:
        if cand in tok.token_to_id:
            return tok.token_to_id[cand]
    raise ValueError(f"target '{target_text}' not in vocab")


class FailureCategory:
    __slots__ = ("kind", "detail", "repair_hint")

    def __init__(self, kind: str, detail: str, repair_hint: str):
        self.kind = kind
        self.detail = detail
        self.repair_hint = repair_hint


def categorize_failure(test_output: str, prev_code: str) -> FailureCategory:
    out = test_output or ""
    if "no extractable code" in out.lower() or not prev_code:
        return FailureCategory(
            kind="NoCode",
            detail="Output contained no extractable Python code",
            repair_hint=(
                "Re-output JUST the Python code in a single ```python``` "
                "fenced block. NO prose, NO explanation, NO markdown "
                "headers — only the function or class definition."
            ),
        )
    mm = re.search(r"NameError: name '(\w+)' is not defined", out)
    if mm:
        sym = mm.group(1)
        suggested = COMMON_IMPORTS.get(sym, f"# add appropriate import for {sym}")
        return FailureCategory(
            kind="NameError",
            detail=f"NameError: '{sym}' is not defined",
            repair_hint=(
                f"Your code uses `{sym}` but doesn't import it. "
                f"Add this line at the top: `{suggested}`. "
                f"Output the complete corrected code."
            ),
        )
    if "'int' object is not callable" in out:
        return FailureCategory(
            kind="TypeError",
            detail="TypeError: 'int' object is not callable",
            repair_hint=(
                "You're calling an integer as if it were a function. "
                "Likely cause: a method/function name was overwritten "
                "with an integer value. Rename the integer attribute "
                "and use the new name where you assigned the value. "
                "Output complete code."
            ),
        )
    mm = re.search(
        r"AttributeError: 'NoneType' object has no attribute '(\w+)'", out)
    if mm:
        attr = mm.group(1)
        return FailureCategory(
            kind="AttributeError",
            detail=f"AttributeError: NoneType has no attribute '{attr}'",
            repair_hint=(
                f"An object is None when `.{attr}` is accessed. "
                f"Add a None-check before the access. Output complete code."
            ),
        )
    mm = re.search(r"SyntaxError: ([^\n]+)", out)
    if mm:
        return FailureCategory(
            kind="SyntaxError",
            detail=f"SyntaxError: {mm.group(1)}",
            repair_hint=(
                f"Python syntax error: {mm.group(1)}. "
                f"Output complete corrected code."
            ),
        )
    mm = re.search(r"ValueError: ([^\n]+)", out)
    if mm:
        return FailureCategory(
            kind="ValueError",
            detail=f"ValueError: {mm.group(1)}",
            repair_hint=(
                f"ValueError: {mm.group(1)}. "
                f"Add validation/try-except around the conversion. "
                f"Output complete code."
            ),
        )
    fail_lines = [l for l in out.splitlines() if l.startswith("FAIL")]
    if fail_lines:
        f = fail_lines[0][:120]
        return FailureCategory(
            kind="FAIL",
            detail=f,
            repair_hint=(
                f"Test assertion failed: '{f}'. "
                f"Trace your logic. Output complete corrected code."
            ),
        )
    if "Runtime error" in out or "Traceback" in out:
        return FailureCategory(
            kind="Other",
            detail=out[:200],
            repair_hint=(
                f"Runtime error: {out[:200]}. Output complete code."
            ),
        )
    return FailureCategory(
        kind="Unknown",
        detail=out[:200],
        repair_hint=f"Tests failed: {out[:200]}. Output complete code.",
    )


REPAIR_PROMPT_TEMPLATE = """\
Fix this Python code based on the SPECIFIC issue identified.

Problem: {prompt}

Your code:
```python
{prev_code}
```

Issue: {repair_hint}

Output the complete corrected ```python``` block:
"""


def run_eval(m, tok) -> None:
    import sys as _sys
    for mod_name in list(_sys.modules.keys()):
        if (mod_name.startswith("calm.llm_computer.")
                and mod_name != "calm.llm_computer"):
            del _sys.modules[mod_name]
    for mod_name in list(_sys.modules.keys()):
        if mod_name == "scripts.r53_eval_complex":
            del _sys.modules[mod_name]

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from r53_eval_complex import (
        CORPUS, gen_stock, gen_hinted, score, extract_code,
        BASE_SYSTEM, _trim_markers,
    )
    import r53_eval_complex as orig
    from calm.llm_computer.facades.code_example_db import (
        CodeExampleDB, RetrievalHit,
    )
    from calm.llm_computer.facades.code_verifier import (
        CodeVerifierFacade,
    )
    from calm.llm_computer.gemma_substrate import (
        CardSlot, VerificationHook,
    )
    from calm.llm_computer.persistent_knowledge import KnowledgeStore
    from calm.sandbox import run_python
    import random as _rng_mod

    # Detach any prior install state
    for layer in m.layers:
        if hasattr(layer, "card_slots"):
            layer.card_slots = []
    m.reserved_channels = []
    m.verification_hooks = []
    print("[r53.20b] cleared prior install state", flush=True)
    print(f"[r53.20b] MAX_ATTEMPTS={MAX_ATTEMPTS}, MAX_TOKENS={MAX_TOKENS}",
          flush=True)

    db = CodeExampleDB.load_default()
    db.load_indices(CACHE_DIR)
    print(f"[r53.20b] DB loaded ({len(db)} examples)", flush=True)

    rng = _rng_mod.Random(0)

    def _build_hints_channel(db, rng_, p, sanity_random):
        facade = CodeVerifierFacade(db=db, top_k=2)
        hints = facade.compute_hints(p.prompt)
        channel_hits = db.retrieve_channel(
            p.prompt, channel="code", k=2, mode="hybrid",
            dense_m=m, dense_tok=tok)
        hints.retrieved_examples = channel_hits
        block = hints.to_system_prefix(max_example_chars=160)
        if len(block) > 1200:
            block = block[:1200] + "\n..."
        return block

    orig._build_hints = _build_hints_channel

    # ------------- Install substrate: KnowledgeStore + FirstTokenHook -------------
    store = KnowledgeStore(max_key=MAX_KEY, max_value=MAX_VALUE)
    eval_keys: List[Tuple[str, int, int]] = []
    for marker, p in enumerate(CORPUS, start=1):
        key = hash_prompt(p.prompt)
        store.add_correction(key, marker)
        eval_keys.append((p.name, key, marker))
    recall = store.build_recall_model().cuda().eval()

    target_ids = {marker: find_token_id(tok, txt)
                  for marker, txt in PER_MARKER_TARGETS.items()}

    current_query = {"key": 0}

    def recall_input(h):
        return torch.tensor([[current_query["key"]]], device="cuda")

    def recall_output(h, logits, ch_lo, ch_hi):
        n = min(RECALL_D_CARD, ch_hi - ch_lo, logits.shape[-1])
        ans = logits[:, -1:, :n]
        h[..., -1:, ch_lo:ch_lo + n] = (
            h[..., -1:, ch_lo:ch_lo + n] + ans)
        return h

    slot = CardSlot(
        layer_idx=INSTALL_LAYER, ch_off=RECALL_CH_OFF, card=recall,
        d_card=RECALL_D_CARD,
        card_input_fn=recall_input,
        use_full_residual=True,
        output_fn=recall_output,
    )
    slot.attach(m, preserve=True)

    class FirstTokenHook:
        def __init__(self, inner: VerificationHook):
            self.inner = inner
            self.fired = False

        def __call__(self, logits):
            if self.fired:
                return logits
            self.fired = True
            return self.inner(logits)

        def reset(self):
            self.fired = False

    inner = VerificationHook(
        slot, vocab_mapping=dict(target_ids),
        boost=HOOK_BOOST, min_margin=HOOK_MIN_MARGIN,
    )
    first_token_hook = FirstTokenHook(inner)
    m.verification_hooks.append(first_token_hook)
    print(f"[r53.20b] installed KnowledgeStore@L{INSTALL_LAYER}, "
          f"hook boost={HOOK_BOOST}, first-token-only, "
          f"{len(eval_keys)} keys", flush=True)

    # ------------- Helpers (substrate-aware) -------------
    def set_key_for_problem(p) -> int:
        k = hash_prompt(p.prompt)
        current_query["key"] = k
        first_token_hook.reset()
        return k

    def get_test_output(code: str, test_code: str) -> str:
        if not code:
            return "no extractable code"
        combined = code + "\n\n" + test_code + "\npass\n"
        result = run_python(combined, timeout=5.0)
        if result.error:
            return f"Runtime error: {result.error}\n{result.stdout or ''}"
        return result.stdout or "(no test output)"

    def gen_repair_with_substrate(p, prev_code: str, hint: str) -> str:
        # Repair prompt uses a different text → different hash → substrate
        # MISS → card output all-zero → Gemma unmodified on repair.
        # We still reset the hook so it doesn't trigger on any other call.
        first_token_hook.reset()
        problem_trim = p.prompt[:200]
        code_trim = prev_code[:280]
        repair_prompt = REPAIR_PROMPT_TEMPLATE.format(
            prompt=problem_trim,
            prev_code=code_trim,
            repair_hint=hint,
        )
        # For repair prompts we want substrate hash-match OFF so native
        # Gemma runs unmodified. set_key to something not in store.
        current_query["key"] = 0  # 0 is not in store (enrolled keys hash to nonzero)
        out = m.generate(repair_prompt, tok, max_tokens=MAX_TOKENS,
                         device="cuda", stop_on_eos=True)
        return _trim_markers(out["text"])

    # ------------- Eval loop -------------
    print(f"\n[r53.20b] running {len(CORPUS)} problems with full stack...",
          flush=True)

    results: List[Tuple[str, int, int, int, int, int, str]] = []

    for i, p in enumerate(CORPUS):
        print(f"\n[{i+1}/{len(CORPUS)}] {p.name}", flush=True)
        t0 = time.time()

        # ATTEMPT 1: substrate active + channel-code-hybrid hints
        set_key_for_problem(p)
        raw = gen_hinted(m, tok, p, db, rng, sanity_random=False,
                         max_tokens=MAX_TOKENS)
        sp1, st1, _ = score(raw, p)
        print(f"  attempt 1 (substrate+hinted): {sp1}/{st1} "
              f"({time.time()-t0:.0f}s)", flush=True)

        best_pass, best_total = sp1, st1
        last_kind = "ok" if (st1 > 0 and sp1 == st1) else "n/a"
        n_attempts = 1
        prev_raw = raw

        for attempt_idx in range(2, MAX_ATTEMPTS + 1):
            if best_total > 0 and best_pass == best_total:
                break
            prev_code = extract_code(prev_raw, p.required)
            test_output = get_test_output(prev_code, p.test_code)
            cat = categorize_failure(test_output, prev_code)
            last_kind = cat.kind
            hint = cat.repair_hint
            code_for_prompt = prev_code if prev_code else "(no code emitted)"
            t1 = time.time()
            new_raw = gen_repair_with_substrate(p, code_for_prompt, hint)
            new_pass, new_total, _ = score(new_raw, p)
            n_attempts += 1
            print(f"  attempt {attempt_idx} ({cat.kind}): "
                  f"{new_pass}/{new_total} ({time.time()-t1:.0f}s) "
                  f"[hint: {cat.detail[:60]}]", flush=True)
            if new_total > best_total or (new_total == best_total
                                          and new_pass > best_pass):
                best_pass, best_total = new_pass, new_total
                prev_raw = new_raw
            else:
                break

        results.append((p.name, sp1, st1, best_pass, best_total,
                        n_attempts, last_kind))

    # ------------- AGGREGATE -------------
    print("\n" + "=" * 110, flush=True)
    print(f"  {'name':<28} {'attempt 1':>11} {'final':>11} "
          f"{'attempts':>9} {'last err':>16}", flush=True)
    print("-" * 110, flush=True)
    s_total = (0, 0)
    f_total = (0, 0)
    for name, sp, st, fp, ft, na, kind in results:
        improved = "✓" if (fp/max(ft, 1) > sp/max(st, 1)) else (
            "=" if fp/max(ft, 1) == sp/max(st, 1) else "↓")
        print(f"  {name:<28} {sp:>4}/{st:<4}    {fp:>4}/{ft:<4}    "
              f"{na:>4}      {kind:>15}  {improved}", flush=True)
        s_total = (s_total[0] + sp, s_total[1] + st)
        f_total = (f_total[0] + fp, f_total[1] + ft)
    print("-" * 110, flush=True)
    print(f"  {'TOTAL':<28} {s_total[0]:>4}/{s_total[1]:<4}    "
          f"{f_total[0]:>4}/{f_total[1]:<4}", flush=True)
    if s_total[1] and f_total[1]:
        delta = (f_total[0]/f_total[1] - s_total[0]/s_total[1]) * 100
        print(f"  Δ final-vs-attempt-1: {delta:+.1f}pp", flush=True)
    print(f"  Baseline (R53.19 v3, no substrate): 26/26", flush=True)


if __name__ == "__main__":
    print("Daemon-only. Use:",
          "  bin/gemma-run scripts/r53_20b_stacked.py",
          flush=True)
    sys.exit(1)
elif "m" in globals() and "tok" in globals():
    run_eval(m, tok)                                  # noqa: F821
