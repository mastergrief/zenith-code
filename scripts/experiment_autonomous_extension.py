"""Autonomous self-extension + candidate selection ("evolution").

Tests two properties of the CRLM stack at once:

1. Self-extension (#1 in the 'living computer' framing): the system
   encounters a task it has no program for, uses synth-A (L5) to
   propose a candidate expression, validates it, and registers it in
   a persistent library. Future queries matching the signature are
   answered from the library without re-invoking synth.

2. Evolutionary selection (#3): for each task we sample N candidate
   expressions from synth-A (temperature > 0, so decode is stochastic).
   Each candidate is validated by parsing to a GateGraph and
   interpreting against held-out IO pairs. Only candidates that pass
   are kept; failures are discarded. The library grows monotonically
   with only correct programs.

This is growth-by-selection: variation comes from sampling, selection
comes from the interpreter's correctness oracle, and the substrate
persists every survivor.

Measurement: after N tasks, the library contains M programs where M
is the count of successfully-synthesized programs. Library hit rate
on subsequent queries of the same templates: should be 100% (the
programs are exact IR).
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from calm.hrm.data import _CHAR_TO_ID, _ID_TO_CHAR
from calm.llm_computer.interpret import InterpreterError, interpret
from calm.llm_computer.parse import ParseError, parse_expression
from calm.llm_computer.synth.data import SynthFamilyAGenerator, encode_examples
from calm.llm_computer.synth.infer import SynthFamilyAReasoner


@dataclass
class LibraryEntry:
    expression: str
    discovered_on_seed: int
    times_invoked: int = 0


@dataclass
class ExperimentResult:
    library: Dict[str, LibraryEntry] = field(default_factory=dict)
    candidate_counts: List[int] = field(default_factory=list)
    success_counts: List[int] = field(default_factory=list)
    failure_tasks: List[str] = field(default_factory=list)


def _sample_candidates(reasoner, sample, n: int,
                        temperature: float = 0.8) -> List[str]:
    """Sample N candidate expressions from synth-A via temperature decode."""
    pad = _CHAR_TO_ID["<pad>"]
    bos = _CHAR_TO_ID["<bos>"]
    eos = _CHAR_TO_ID["<eos>"]

    enc = reasoner._encode_string(encode_examples(sample))
    candidates = []
    for _ in range(n):
        with torch.no_grad():
            mem = reasoner.model.encode(enc)
            dec = [bos]
            for _ in range(reasoner.config.max_dec_len - 1):
                padded = dec + [pad] * (reasoner.config.max_dec_len - len(dec))
                dt = torch.tensor([padded], dtype=torch.long,
                                   device=reasoner.device)
                logits = reasoner.model.decode_step(dt, mem)
                probs = torch.softmax(logits[0, len(dec) - 1, :] / temperature,
                                       dim=-1)
                nid = int(torch.multinomial(probs, num_samples=1).item())
                if nid == eos:
                    break
                dec.append(nid)
        out = ""
        for tid in dec[1:]:
            if tid in (pad, bos, eos):
                continue
            out += _ID_TO_CHAR.get(tid, "?")
        candidates.append(out.strip())
    return candidates


def _validate_candidate(candidate: str, sample) -> bool:
    """Parse + interpret candidate against ALL IO pairs (examples + query).

    Stricter than single-query validation — candidates must agree on every
    example and the query. Catches coincidence passes where a WRONG
    template happens to match a single point.
    """
    all_pairs = list(sample.examples) + [(sample.query_a, sample.query_b, sample.query_out)]
    for a_val, b_val, expected in all_pairs:
        try:
            expr_concrete = (candidate
                             .replace("a", str(a_val))
                             .replace("b", str(b_val)))
            graph = parse_expression(expr_concrete)
            val = interpret(graph)
            if isinstance(val, float) and val == int(val):
                val = int(val)
            if val != expected:
                return False
        except (ParseError, InterpreterError, ValueError):
            return False
    return True


def _library_key(sample) -> str:
    """Signature used for library indexing — the template shape.

    For Family A, the signature is just 'two-operand arithmetic', so all
    tasks share a bucket. A richer signature (operator type, arity) is
    future work. For this experiment we use the template itself as the
    key, which means each TRUE template gets its own library entry.
    """
    return sample.template


def run(n_tasks: int = 15, n_candidates: int = 5, ckpt: str = "calm/hrm/checkpoints/synth_familyA_best.pt"):
    print(f"[auto-extend] loading synth-A checkpoint from {ckpt}")
    if not Path(ckpt).exists():
        print(f"[auto-extend] ERROR: checkpoint missing")
        return

    reasoner = SynthFamilyAReasoner(ckpt)
    gen = SynthFamilyAGenerator(seed=12345)
    result = ExperimentResult()

    print(f"[auto-extend] running {n_tasks} tasks with {n_candidates} candidate samples each\n")
    print(f"  {'#':<3} {'template':<14} {'candidates':<48} {'library size'}")
    print(f"  {'-'*3} {'-'*14} {'-'*48} {'-'*12}")

    samples = gen.generate(n_tasks)
    for i, sample in enumerate(samples):
        key = _library_key(sample)
        # If already in library, verify it still works (regression check).
        if key in result.library:
            entry = result.library[key]
            ok = _validate_candidate(entry.expression, sample)
            entry.times_invoked += 1
            tag = "CACHE ✓" if ok else "CACHE ✗"
            print(f"  {i:<3} {sample.template:<14} {tag:<48} {len(result.library)}")
            continue

        cands = _sample_candidates(reasoner, sample, n_candidates)
        # Selection: first passing candidate survives.
        survivor = None
        for c in cands:
            if _validate_candidate(c, sample):
                survivor = c
                break

        result.candidate_counts.append(len(cands))
        if survivor is not None:
            result.success_counts.append(1)
            result.library[key] = LibraryEntry(
                expression=survivor,
                discovered_on_seed=i,
            )
            cands_display = ", ".join(cands[:3])
            if len(cands) > 3:
                cands_display += f", +{len(cands)-3}"
            cands_display = cands_display[:46]
            print(f"  {i:<3} {sample.template:<14} {cands_display:<48} "
                  f"{len(result.library)}")
        else:
            result.success_counts.append(0)
            result.failure_tasks.append(sample.template)
            cands_display = ", ".join(cands[:3])[:46]
            print(f"  {i:<3} {sample.template:<14} [FAIL] {cands_display[:42]:<42} "
                  f"{len(result.library)}")

    print("\n[auto-extend] FINAL LIBRARY")
    for key, entry in sorted(result.library.items()):
        print(f"  {key:<14} → {entry.expression!r:14}  "
              f"(discovered task {entry.discovered_on_seed}, "
              f"reused {entry.times_invoked} times)")

    n_tried = len(result.candidate_counts)
    n_solved = sum(result.success_counts)
    print(f"\n[auto-extend] growth: {n_solved}/{n_tried} novel tasks solved, "
          f"{len(result.library)} unique programs in library")
    print(f"[auto-extend] {len(result.failure_tasks)} failures: "
          f"{result.failure_tasks[:5]}")

    return result


if __name__ == "__main__":
    random.seed(0)
    torch.manual_seed(0)
    run(n_tasks=15, n_candidates=5)
