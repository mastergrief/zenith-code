"""CHRLM-General — reference build on the substrate.

Composes the current substrate card inventory into a general-knowledge
brain + toolset:

  Tier 1 — compiled programs (exact, priority 100-200)
  Tier 2 — HRM specialist cards (structural, priority 300-400)
  Tier 3 — trained SubstrateLM / SubstrateHRLM brain (priority 500-600)
  Tier 4 — external LLM fallback (Gemma via llama-server, priority 900)

At query time the orchestrator tries cards in ascending priority. Exact-
answer cards win when their pattern matches; the brain + external LLM
are fallbacks for open-ended queries.

Usage:
    from calm.llm_computer.builds.chrlm_general import build
    chrlm = build()
    result = chrlm.route("what is 17 + 23")
    print(result.answer, "via", result.card)
"""

from __future__ import annotations

import re
from pathlib import Path

from calm.expression import safe_eval
from calm.llm_computer.cards import (
    CardOrchestrator,
    CompiledProgramCard,
    ExternalLLMCard,
    SubstrateLMCard,
)


REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CKPT_DIR = REPO_ROOT / "calm" / "llm_computer" / "checkpoints"


# ----- Compiled-program card builders -----

def _eval_binary_arith(match: re.Match) -> str:
    a, op, b = match.group(1), match.group(2), match.group(3)
    result = safe_eval(f"{a} {op} {b}")
    if isinstance(result, float) and result == int(result):
        result = int(result)
    return str(result)


def _eval_func_one_arg(match: re.Match) -> str:
    fn, a = match.group(1), match.group(2)
    result = safe_eval(f"{fn}({a})")
    if isinstance(result, bool):
        return "True" if result else "False"
    if isinstance(result, float) and result == int(result):
        result = int(result)
    return str(result)


def _eval_func_two_args(match: re.Match) -> str:
    fn, a, b = match.group(1), match.group(2), match.group(3)
    result = safe_eval(f"{fn}({a}, {b})")
    if isinstance(result, float) and result == int(result):
        result = int(result)
    return str(result)


def compiled_cards() -> list[CompiledProgramCard]:
    """The exact-computation tier. Every card here routes to a verified
    function via `safe_eval` (which underneath calls either a compiled
    gate-graph program or one of the 1002 CALM backend functions).
    """
    return [
        CompiledProgramCard(
            name="binary_arith",
            pattern=re.compile(r"(\b\d+\b)\s*([\+\-\*/%])\s*(\b\d+\b)"),
            evaluator=_eval_binary_arith,
            priority=100,
        ),
        CompiledProgramCard(
            name="gcd",
            pattern=re.compile(r"\b(gcd)\(\s*(\d+)\s*,\s*(\d+)\s*\)"),
            evaluator=_eval_func_two_args,
            priority=110,
        ),
        CompiledProgramCard(
            name="lcm",
            pattern=re.compile(r"\b(lcm)\(\s*(\d+)\s*,\s*(\d+)\s*\)"),
            evaluator=_eval_func_two_args,
            priority=110,
        ),
        CompiledProgramCard(
            name="factorial",
            pattern=re.compile(r"\b(factorial)\(\s*(\d+)\s*\)"),
            evaluator=_eval_func_one_arg,
            priority=120,
        ),
        CompiledProgramCard(
            name="is_prime",
            pattern=re.compile(r"\b(is_prime)\(\s*(\d+)\s*\)"),
            evaluator=_eval_func_one_arg,
            priority=120,
        ),
        CompiledProgramCard(
            name="fibonacci",
            pattern=re.compile(r"\b(fibonacci)\(\s*(\d+)\s*\)"),
            evaluator=_eval_func_one_arg,
            priority=120,
        ),
    ]


# ----- Trained substrate-native card (SubstrateLM MVP fallback) -----

def substrate_lm_mvp_card() -> SubstrateLMCard:
    """SubstrateLM MVP card — format-only, weak content. Acts as a
    substrate-native fallback before routing to the external LLM.
    """
    return SubstrateLMCard(
        name="substrate_lm_mvp",
        checkpoint_path=str(CKPT_DIR / "substrate_lm_mvp.pt"),
        tokenizer_path=str(CKPT_DIR / "substrate_lm_mvp_tokenizer.json"),
        priority=600,
        max_new_tokens=180,
    )


def substrate_hrlm_v2_card() -> SubstrateLMCard:
    """SubstrateHRLM v2 card — trained brain with both LM and HRM modes.
    Preferred over MVP when the checkpoint exists.
    """
    return SubstrateLMCard(
        name="substrate_hrlm_v2",
        checkpoint_path=str(CKPT_DIR / "substrate_hrlm_v2.pt"),
        tokenizer_path=str(CKPT_DIR / "substrate_hrlm_v2_tokenizer.json"),
        priority=500,
        max_new_tokens=200,
        mode_prefix="<|lm|>",
    )


# ----- External LLM fallback (Gemma) -----

def gemma_fallback_card() -> ExternalLLMCard:
    return ExternalLLMCard(
        name="gemma_fallback",
        priority=900,
        system_prompt=(
            "You are Zenith, the external LLM backend of a CHRLM build. "
            "The user's question is only reaching you because no substrate-"
            "native card handled it. Answer concisely."
        ),
    )


# ----- Build factory -----

def build(include_trained_brain: bool = True,
          include_gemma_fallback: bool = True) -> CardOrchestrator:
    """Instantiate the CHRLM-General build.

    Card inventory depends on what's on disk:
      - Compiled cards: always included (pure Python via safe_eval)
      - SubstrateHRLM v2: included if checkpoint exists, else falls back
        to SubstrateLM MVP if that exists
      - Gemma: included if `include_gemma_fallback=True` (card itself
        checks endpoint health at query time)
    """
    cards: list = compiled_cards()

    if include_trained_brain:
        v2_path = CKPT_DIR / "substrate_hrlm_v2.pt"
        mvp_path = CKPT_DIR / "substrate_lm_mvp.pt"
        if v2_path.exists():
            cards.append(substrate_hrlm_v2_card())
        elif mvp_path.exists():
            cards.append(substrate_lm_mvp_card())

    if include_gemma_fallback:
        cards.append(gemma_fallback_card())

    return CardOrchestrator(cards=cards, name="CHRLM-General")
