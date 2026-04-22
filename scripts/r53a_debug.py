"""Debug NumberTheoryFacade — print raw Gemma output + token sequences."""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
assert "m" in globals() and "tok" in globals(), (  # type: ignore[name-defined]
    "run via bin/gemma-run scripts/r53a_debug.py"
)
sys.path.insert(0, str(ROOT))
from calm.llm_computer.facades.number_theory import NumberTheoryFacade
from calm.llm_computer.facades.retrieval import _monkey_patch_fast_encode
_monkey_patch_fast_encode(tok)  # type: ignore[name-defined]


def main():
    facade = NumberTheoryFacade(device="cuda")
    facade.install(m, tok)  # type: ignore[name-defined]

    for prompt, expected in [
        ("What is 25 mod 7?", 4),
        ("What is 127 mod 13?", 10),
        ("What is the LCM of 12 and 18?", 36),
    ]:
        print(f"\n=== prompt={prompt!r} expected={expected} ===")
        op, operands = facade.parse(prompt)
        val = facade.evaluate(op, operands) if op else None
        print(f"parse: op={op!r} operands={operands} eval={val}")

        if val is not None:
            bias_ids = facade._gemma_digit_tokens(val)
            print(f"bias token_ids ({len(bias_ids)}): {bias_ids}")
            # What does tok decode these as
            for i, tid in enumerate(bias_ids):
                t_str = tok.id_to_token.get(tid, "?")  # type: ignore[name-defined]
                print(f"  [{i}] id={tid} str={t_str!r}")

        r0 = facade.solve(prompt, use_bias=False)
        print(f"baseline out: {r0.generated[:200]!r}")
        print(f"baseline parsed: {r0.parsed_answer}")

        r1 = facade.solve(prompt, use_bias=True)
        print(f"facade out: {r1.generated[:200]!r}")
        print(f"facade parsed: {r1.parsed_answer}")


main()
print("R53A_DEBUG_DONE")
