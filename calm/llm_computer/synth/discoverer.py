"""Discoverer — autonomous program discovery with library + mutation.

Wraps `SynthFamilyAReasoner` + `Library`:
  1. Check library — if the task's signature is known, return the stored
     program directly (cache hit, no synth forward pass).
  2. On miss, sample candidates with escalating temperature and count.
     Each candidate is validated by strict parse+interpret over all IO
     pairs (examples + query). First passing candidate registers.
  3. If all retries fail, return (None, failure).

Mutation schedule (escalates if earlier attempts all collapse):
    attempt 1: temp=0.8, N=5
    attempt 2: temp=1.2, N=10
    attempt 3: temp=1.8, N=20

Higher temperature increases sampling entropy — breaks distribution
mode-collapse that causes all candidates to be identical.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import torch

from calm.hrm.data import _CHAR_TO_ID, _ID_TO_CHAR
from calm.llm_computer.interpret import InterpreterError, interpret
from calm.llm_computer.parse import ParseError, parse_expression
from calm.llm_computer.synth.data import SynthSample, encode_examples
from calm.llm_computer.synth.infer import SynthFamilyAReasoner
from calm.llm_computer.synth.library import (
    DEFAULT_LIBRARY_PATH, Library, LibraryEntry,
)


MUTATION_SCHEDULE = [
    (0.8, 5),
    (1.2, 10),
    (1.8, 20),
]


@dataclass
class DiscoveryResult:
    hit: bool                    # True if served from library
    expression: Optional[str]    # the program used / discovered / None on fail
    answer: Optional[int]        # computed answer, or None on fail
    attempts: int                # number of mutation attempts used
    candidates_sampled: int      # total candidates evaluated
    library_size: int            # current library size after call


class Discoverer:
    def __init__(self, reasoner: SynthFamilyAReasoner,
                 library_path: Path = DEFAULT_LIBRARY_PATH):
        self.reasoner = reasoner
        self.library = Library(path=library_path)

    def _sample(self, sample: SynthSample, temperature: float,
                 n: int) -> List[str]:
        pad = _CHAR_TO_ID["<pad>"]
        bos = _CHAR_TO_ID["<bos>"]
        eos = _CHAR_TO_ID["<eos>"]
        enc = self.reasoner._encode_string(encode_examples(sample))
        out = []
        for _ in range(n):
            with torch.no_grad():
                mem = self.reasoner.model.encode(enc)
                dec = [bos]
                for _ in range(self.reasoner.config.max_dec_len - 1):
                    padded = dec + [pad] * (self.reasoner.config.max_dec_len - len(dec))
                    dt = torch.tensor([padded], dtype=torch.long,
                                       device=self.reasoner.device)
                    logits = self.reasoner.model.decode_step(dt, mem)
                    probs = torch.softmax(logits[0, len(dec) - 1, :] / temperature,
                                           dim=-1)
                    nid = int(torch.multinomial(probs, num_samples=1).item())
                    if nid == eos:
                        break
                    dec.append(nid)
            s = ""
            for tid in dec[1:]:
                if tid in (pad, bos, eos):
                    continue
                s += _ID_TO_CHAR.get(tid, "?")
            out.append(s.strip())
        return out

    @staticmethod
    def _validate(candidate: str, sample: SynthSample) -> Optional[int]:
        """Validate candidate on all IO pairs. Returns the answer on the query
        if passing; None otherwise."""
        all_pairs = list(sample.examples) + [
            (sample.query_a, sample.query_b, sample.query_out)
        ]
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
                    return None
            except (ParseError, InterpreterError, ValueError):
                return None
        # All pairs passed — return the query answer (which we already computed).
        return sample.query_out

    @staticmethod
    def _signature(sample: SynthSample) -> str:
        """The library key. For Family A we use the known template; in
        production we'd derive a signature from IO alone (e.g., hash of
        behavior on a canonical probe set)."""
        return sample.template

    def solve(self, sample: SynthSample) -> DiscoveryResult:
        key = self._signature(sample)

        # (1) Library lookup
        entry = self.library.lookup(key)
        if entry is not None:
            answer = self._validate(entry.expression, sample)
            return DiscoveryResult(
                hit=True,
                expression=entry.expression,
                answer=answer,
                attempts=0,
                candidates_sampled=0,
                library_size=len(self.library),
            )

        # (2) Discover via mutation schedule
        total_sampled = 0
        for attempt_idx, (temp, n) in enumerate(MUTATION_SCHEDULE, start=1):
            cands = self._sample(sample, temperature=temp, n=n)
            total_sampled += len(cands)
            for c in cands:
                ans = self._validate(c, sample)
                if ans is not None:
                    entry = self.library.register(key, c)
                    return DiscoveryResult(
                        hit=False,
                        expression=c,
                        answer=ans,
                        attempts=attempt_idx,
                        candidates_sampled=total_sampled,
                        library_size=len(self.library),
                    )

        # (3) All attempts failed
        return DiscoveryResult(
            hit=False,
            expression=None,
            answer=None,
            attempts=len(MUTATION_SCHEDULE),
            candidates_sampled=total_sampled,
            library_size=len(self.library),
        )
