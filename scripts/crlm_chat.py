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

import sys
from pathlib import Path

import torch

from calm.hrm.dispatcher import DEFAULT_ROUTER_CKPT, Dispatcher
from calm.llm_computer.programs.isa import (
    DBL, DEC, HALT, HLT, INC, run_isa,
)
from calm.llm_computer.synth.discoverer import Discoverer
from calm.llm_computer.synth.infer import SynthFamilyAReasoner
from calm.llm_computer.synth.nl_io import parse_nl_io


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

    last_failed_sample = None

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
            print("commands: !library | !isa <OP> <V> | !correct <expr> | !quit")
            print("tips:")
            print("  NL arithmetic task: '3 and 5 give 8, 2 and 7 give 9, ... what about 4 and 6?'")
            print("  IO-style task:      'a=3 b=5: 8 | a=2 b=7: 9 | ? a=4 b=6'")
            print("  single-argument:    '8 becomes 4, 6 becomes 3, ... what about 14?'")
            print("  natural math:       'what is 347 times 289'")
            continue
        if text.startswith("!correct "):
            if last_failed_sample is None:
                print("  nothing to correct — no recent failure")
                continue
            expr = text[len("!correct "):].strip()
            if discoverer is None:
                print("  discoverer unavailable")
                continue
            # Validate user-taught expression against the examples.
            ans = discoverer._validate(expr, last_failed_sample,
                                        require_query=False)
            if ans is None:
                print(f"  {expr!r} doesn't match all examples — rejecting")
                continue
            canon = discoverer._canonical(expr)
            discoverer.library.register(canon, expr)
            q_shown = (f"{last_failed_sample.query_a},{last_failed_sample.query_b}"
                        if last_failed_sample.query_b != 0
                        else f"{last_failed_sample.query_a}")
            print(f"  [TAUGHT] registered {expr!r} in library; query {q_shown} = {ans}")
            last_failed_sample = None
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

        # IO-example or NL IO task? Parse via NL parser (handles both).
        if discoverer is not None:
            sample = parse_nl_io(text)
            if sample is not None:
                # User doesn't know the query answer — validate on examples only.
                # Use behavior matching so equivalent programs dedupe.
                r = discoverer.solve(sample, require_query=False,
                                      use_behavior_match=True)
                tag = "LIBRARY" if r.hit else "DISCOVERED"
                if r.answer is None:
                    last_failed_sample = sample
                    print(f"  [{tag}] failed after {r.attempts} mutation attempts "
                          f"({r.candidates_sampled} samples). "
                          f"Teach me with: !correct <expression>")
                else:
                    last_failed_sample = None
                    q_shown = (f"{sample.query_a},{sample.query_b}"
                                if sample.query_b != 0
                                else f"{sample.query_a}")
                    print(f"  [{tag}] program={r.expression!r} → "
                          f"{q_shown} = {r.answer}")
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
