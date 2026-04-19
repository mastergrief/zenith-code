"""R53 dual-path audit — size the reasoning-trace + code-fragment corpora.

Hypothesis: a meaningful subset (>30%) of the 8970 DB examples has
non-empty <think> blocks. Result determines whether dual-path
ingest is straightforward (most examples have both channels) or
requires generator augmentation (reasoning channel sparse).

Measures:
  - count + percent with <think>...</think> block
  - count + percent with extractable code fragment
  - per-source breakdown of both
  - length stats (mean / median / p10 / p90) for each channel
  - cross-tab: both / think-only / code-only / neither

No code changes — pure measurement before designing the schema.
"""

from __future__ import annotations

import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from calm.llm_computer.facades.code_example_db import CodeExampleDB


THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
FENCE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)
DEFCLASS_RE = re.compile(r"(^|\n)(def |class |import |from \w+ import )")


def extract_think(solution: str) -> str:
    """Concatenated content of all <think>...</think> blocks."""
    parts = THINK_RE.findall(solution)
    return "\n\n".join(p.strip() for p in parts if p.strip())


def extract_code(solution: str) -> str:
    """Code-only view: prefer ```python fence, else slice from def/class."""
    s = THINK_RE.sub("", solution)
    for trailer_pat in (r"\*\*Verified test cases",
                         r"\*\*Sample I/O",
                         r"\*\*Unit tests",
                         r"\*\*Test harness"):
        s = re.split(trailer_pat, s, maxsplit=1)[0]
    m = FENCE_RE.search(s)
    if m:
        return m.group(1).strip()
    m = DEFCLASS_RE.search(s)
    if m:
        return s[m.start():].strip()
    return ""


def percentiles(xs: list[int], qs=(0.1, 0.5, 0.9)) -> dict:
    if not xs:
        return {f"p{int(q*100)}": 0 for q in qs}
    s = sorted(xs)
    n = len(s)
    return {f"p{int(q*100)}": s[min(int(q * n), n - 1)] for q in qs}


def short(p: str) -> str:
    return Path(p).name


def main() -> None:
    print("Loading DB...")
    db = CodeExampleDB.load_default()
    n = len(db)
    print(f"Loaded {n} unique examples\n")

    has_think = 0
    has_code = 0
    think_lens: list[int] = []
    code_lens: list[int] = []
    by_source_think: defaultdict[str, int] = defaultdict(int)
    by_source_code: defaultdict[str, int] = defaultdict(int)
    by_source_total: Counter = Counter()
    cross_tab = Counter()  # (has_think, has_code) -> count

    for ex in db.examples:
        src = short(ex.source)
        by_source_total[src] += 1
        t = extract_think(ex.solution)
        c = extract_code(ex.solution)
        h_think = bool(t)
        h_code = bool(c)
        if h_think:
            has_think += 1
            think_lens.append(len(t))
            by_source_think[src] += 1
        if h_code:
            has_code += 1
            code_lens.append(len(c))
            by_source_code[src] += 1
        cross_tab[(h_think, h_code)] += 1

    print("=" * 72)
    print("CHANNEL COVERAGE")
    print("=" * 72)
    print(f"Has reasoning trace (<think>): {has_think:>5d} / {n} = {has_think/n*100:5.1f}%")
    print(f"Has code fragment:             {has_code:>5d} / {n} = {has_code/n*100:5.1f}%")

    print("\nCross-tab (think × code):")
    print(f"  both:         {cross_tab[(True, True)]:>5d}")
    print(f"  think only:   {cross_tab[(True, False)]:>5d}")
    print(f"  code only:    {cross_tab[(False, True)]:>5d}")
    print(f"  neither:      {cross_tab[(False, False)]:>5d}")

    print("\nReasoning-trace length (chars):")
    if think_lens:
        ps = percentiles(think_lens)
        print(f"  count={len(think_lens)} mean={statistics.mean(think_lens):.0f}"
              f" median={statistics.median(think_lens):.0f}"
              f" p10={ps['p10']} p90={ps['p90']}")

    print("\nCode-fragment length (chars):")
    if code_lens:
        ps = percentiles(code_lens)
        print(f"  count={len(code_lens)} mean={statistics.mean(code_lens):.0f}"
              f" median={statistics.median(code_lens):.0f}"
              f" p10={ps['p10']} p90={ps['p90']}")

    print("\n" + "=" * 72)
    print("PER-SOURCE BREAKDOWN")
    print("=" * 72)
    print(f"{'source':<45} {'total':>6} {'think':>6} {'code':>6} {'%th':>5} {'%cd':>5}")
    for src in sorted(by_source_total, key=lambda s: -by_source_total[s]):
        tot = by_source_total[src]
        th = by_source_think[src]
        cd = by_source_code[src]
        print(f"{src:<45} {tot:>6d} {th:>6d} {cd:>6d}"
              f" {th/tot*100:>4.0f}% {cd/tot*100:>4.0f}%")

    print("\n" + "=" * 72)
    print("SAMPLES (first 3 in each category)")
    print("=" * 72)

    samples: dict[tuple, list] = {(True, True): [], (True, False): [],
                                  (False, True): [], (False, False): []}
    for ex in db.examples:
        t = extract_think(ex.solution)
        c = extract_code(ex.solution)
        key = (bool(t), bool(c))
        if len(samples[key]) < 3:
            samples[key].append((ex, t, c))

    for label, key in [("BOTH", (True, True)),
                       ("THINK-ONLY", (True, False)),
                       ("CODE-ONLY", (False, True)),
                       ("NEITHER", (False, False))]:
        print(f"\n--- {label} ---")
        for i, (ex, t, c) in enumerate(samples[key]):
            print(f"\n[{i+1}] source={short(ex.source)}")
            print(f"    problem: {ex.problem[:100]!r}")
            if t:
                print(f"    think  : ({len(t)}c) {t[:120]!r}")
            if c:
                print(f"    code   : ({len(c)}c) {c[:120]!r}")
            if not t and not c:
                print(f"    raw    : {ex.solution[:200]!r}")


if __name__ == "__main__":
    main()
