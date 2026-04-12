"""
CALM v0.1 reasoning chain tracker.

Wraps the CalmEngine to produce a structured reasoning trace:
each step records hypothesis (what the model said), computation
(what the engine ran), result (what the engine returned), and
the model's conclusion based on the result.

This is the "deterministic brain on top of a probabilistic nervous
system" — every computational claim is verified, every branch
point is logged, and the full chain is exportable for analysis.

Usage:
    from calm.reasoning import reason
    chain = reason("Is 2^127 - 1 prime? If so, what's the next prime after it?")
    chain.print()
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import List, Optional

from calm.engine import CalmEngine, EngineResult


@dataclass
class ReasoningStep:
    step: int
    hypothesis: str = ""      # model's text before the CALM block
    computation: str = ""     # the CALM block content
    result: str = ""          # engine injection
    conclusion: str = ""      # model's text after injection


@dataclass
class ReasoningChain:
    prompt: str
    plan: str = ""            # thinking phase output
    steps: List[ReasoningStep] = field(default_factory=list)
    final_answer: str = ""
    engine_result: Optional[EngineResult] = None

    def print(self):
        """Pretty-print the reasoning chain."""
        print(f"Question: {self.prompt}\n")
        if self.plan:
            plan_preview = self.plan[:200] + "..." if len(self.plan) > 200 else self.plan
            print(f"Plan: {plan_preview}\n")
        for s in self.steps:
            print(f"Step {s.step}:")
            if s.hypothesis:
                print(f"  Hypothesis: {s.hypothesis[:100]}")
            if s.computation:
                comp_lines = s.computation.strip().splitlines()
                comp_preview = comp_lines[0] if len(comp_lines) == 1 else f"{comp_lines[0]} ... ({len(comp_lines)} lines)"
                print(f"  Compute:    {comp_preview}")
            if s.result:
                print(f"  Result:     {s.result}")
            if s.conclusion:
                print(f"  Conclusion: {s.conclusion[:100]}")
            print()
        if self.final_answer:
            print(f"Answer: {self.final_answer}")

        if self.engine_result:
            er = self.engine_result
            print(f"\nStats: {er.calm_blocks} blocks, {er.iterations} iters, "
                  f"{len(er.training_log)} training entries, {er.tok_per_sec:.0f} tok/s")

    def to_dict(self) -> dict:
        """Export as a JSON-serializable dict."""
        return {
            "prompt": self.prompt,
            "plan": self.plan,
            "steps": [
                {
                    "step": s.step,
                    "hypothesis": s.hypothesis,
                    "computation": s.computation,
                    "result": s.result,
                    "conclusion": s.conclusion,
                }
                for s in self.steps
            ],
            "final_answer": self.final_answer,
        }


def _parse_chain(response: str) -> List[ReasoningStep]:
    """Parse an engine response into reasoning steps."""
    steps = []
    # Split on CALM blocks: text before <calm>, the block, [engine: ...], text after
    parts = re.split(r'(<calm>.*?</calm>)', response, flags=re.DOTALL)

    step_num = 0
    current_hypothesis = ""

    for part in parts:
        part = part.strip()
        if not part:
            continue

        if part.startswith("<calm>") and part.endswith("</calm>"):
            step_num += 1
            # Extract the computation (between tags)
            computation = part[6:-7].strip()

            # Look for [engine: ...] right after
            steps.append(ReasoningStep(
                step=step_num,
                hypothesis=current_hypothesis.strip(),
                computation=computation,
            ))
            current_hypothesis = ""
        elif part.startswith("[engine:"):
            # Engine injection — attach to last step
            if steps:
                steps[-1].result = part
        else:
            # Text — could be hypothesis for next step or conclusion of previous
            if steps and not steps[-1].conclusion:
                steps[-1].conclusion = part.strip()
            else:
                current_hypothesis = part

    # Any trailing text is the final conclusion
    if steps and current_hypothesis:
        steps[-1].conclusion = current_hypothesis.strip()

    return steps


def reason(prompt: str, verbose: bool = True, **engine_kwargs) -> ReasoningChain:
    """
    Run a reasoning chain. Returns a structured ReasoningChain
    with hypothesis→compute→result→conclusion for each step.
    """
    engine = CalmEngine(**engine_kwargs)
    result = engine.run(prompt, verbose=verbose)

    chain = ReasoningChain(
        prompt=prompt,
        engine_result=result,
    )

    # Parse the response into steps.
    chain.steps = _parse_chain(result.response)

    # Extract final answer (last non-empty conclusion or trailing text).
    if chain.steps:
        for s in reversed(chain.steps):
            if s.conclusion:
                chain.final_answer = s.conclusion
                break
    elif result.response:
        chain.final_answer = result.response.strip()

    return chain


if __name__ == "__main__":
    import sys
    prompt = sys.argv[1] if len(sys.argv) > 1 else (
        "Find the smallest prime greater than 1000. "
        "Is it a twin prime? What is its digit sum?"
    )
    chain = reason(prompt)
    print("\n" + "="*60)
    chain.print()
