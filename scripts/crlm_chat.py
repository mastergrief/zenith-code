"""Talk to the CRLM stack directly.

Four input types, routed automatically:

  math-like text       "what is 347 times 289"
                          → RouterHRM classifies → math specialist HRM
                          → parse → interpret → verified answer

  NL word problem      "Sally has 12 toys. Bob has 29. How many total?"
                          → RouterHRM → word specialist → parse → answer

  IO-example task      "a=3 b=5: 8 | a=2 b=7: 9 | ? a=4 b=6"
                          → Discoverer (library lookup or synth+validate)
                          → answer + library growth

  !library             list discovered programs
  !isa <OP> <V>        run the ISA machine directly (INC/DEC/DBL/HLT)
  !quit                exit

The stack answers using ONLY its own weights. No Gemma, no external
LLM — every response comes from a compiled or learned component of
the CRLM substrate.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import torch

from calm.hrm.dispatcher import DEFAULT_ROUTER_CKPT, Dispatcher
from calm.llm_computer.programs.isa import (
    DBL, DEC, HALT, HLT, INC, run_isa,
)
from calm.llm_computer.synth.data import SynthSample
from calm.llm_computer.synth.discoverer import Discoverer
from calm.llm_computer.synth.infer import SynthFamilyAReasoner


IO_RE = re.compile(
    r"a=(-?\d+)\s+b=(-?\d+)\s*:\s*(-?\d+)",
)
QUERY_RE = re.compile(
    r"\?\s*a=(-?\d+)\s+b=(-?\d+)",
)


def _parse_io_task(text: str):
    """Parse 'a=3 b=5: 8 | a=2 b=7: 9 | ... | ? a=4 b=6' into a SynthSample."""
    pairs = []
    query = None
    for part in text.split("|"):
        part = part.strip()
        m_q = QUERY_RE.search(part)
        if m_q:
            query = (int(m_q.group(1)), int(m_q.group(2)))
            continue
        m = IO_RE.search(part)
        if m:
            pairs.append((int(m.group(1)), int(m.group(2)), int(m.group(3))))
    if len(pairs) >= 3 and query is not None:
        qa, qb = query
        # query_out is unknown at chat time — the discoverer will compute it
        # from the first valid program. We pass a sentinel 0 and ignore the
        # validator's use of it (caller accepts the answer the model gives).
        return SynthSample(
            template="<user>",
            examples=pairs[:3],
            query_a=qa, query_b=qb, query_out=0,  # sentinel
        )
    return None


OP_NAMES = {"INC": INC, "DEC": DEC, "DBL": DBL, "HLT": HLT}


def main():
    print("[crlm-chat] loading dispatcher + discoverer...")
    if not Path(DEFAULT_ROUTER_CKPT).exists():
        print("ERROR: router checkpoint missing. Train it first with "
              "scripts/train_hrm_router.py", file=sys.stderr)
        sys.exit(1)

    dispatcher = Dispatcher()
    discoverer_ckpt = Path("calm/hrm/checkpoints/synth_familyA_best.pt")
    discoverer = None
    if discoverer_ckpt.exists():
        discoverer = Discoverer(SynthFamilyAReasoner(str(discoverer_ckpt)))

    print("[crlm-chat] ready. Type a question, IO task, or !help.\n")

    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[bye]")
            break
        if not text:
            continue
        if text in ("!quit", "!exit", "/quit", "/exit"):
            print("[bye]")
            break
        if text == "!help":
            print("commands: !library | !isa <OP> <V> | !quit")
            continue
        if text == "!library":
            if discoverer is None:
                print("  (discoverer unavailable)")
            else:
                print(f"  library has {len(discoverer.library)} programs:")
                for entry in discoverer.library:
                    print(f"    {entry.key:<18} → {entry.expression:<14}  "
                          f"(invoked {entry.times_invoked}×)")
            continue
        if text.startswith("!isa "):
            parts = text.split()
            if len(parts) != 3:
                print("  usage: !isa <INC|DEC|DBL|HLT> <value>")
                continue
            op_name = parts[1].upper()
            try:
                v = int(parts[2])
            except ValueError:
                print("  value must be an integer")
                continue
            if op_name not in OP_NAMES:
                print(f"  opcode must be one of {list(OP_NAMES)}")
                continue
            seq = run_isa(OP_NAMES[op_name], v)
            trace = " → ".join(
                "HALT" if t == HALT else str(t) for t in seq[1:]
            )
            print(f"  [isa {op_name}] {trace}")
            continue

        # IO-example task?
        if discoverer is not None and "|" in text and "?" in text:
            sample = _parse_io_task(text)
            if sample is not None:
                # User doesn't know the query answer — validate on examples only.
                r = discoverer.solve(sample, require_query=False)
                tag = "LIBRARY" if r.hit else "DISCOVERED"
                if r.answer is None:
                    print(f"  [{tag}] failed after {r.attempts} mutation attempts "
                          f"({r.candidates_sampled} samples)")
                else:
                    print(f"  [{tag}] program={r.expression!r} → "
                          f"{sample.query_a},{sample.query_b} = {r.answer}")
                continue

        # Fall through: natural language via dispatcher.
        try:
            result = dispatcher.run(text)
            label = result.label
            ans = result.answer or "(parse failed)"
            expr = result.expression or "(none)"
            print(f"  [{label}] {expr} = {ans}")
        except Exception as e:
            print(f"  [error] {e}")


if __name__ == "__main__":
    main()
