"""Diagnostic: what does Gemma actually emit for an MBPP prompt under
the CodeDtSkeletonFacade template? Syntax errors across the board in
the N=5 eval mean Gemma's output is malformed/truncated. This probe
dumps the raw output for 3 prompts so we can see the failure mode.
"""
from __future__ import annotations

from calm.llm_computer.facades.code_dt_skeleton import CodeDtSkeletonFacade

facade = CodeDtSkeletonFacade(
    checkpoint_path="calm/hrm/checkpoints/dt_code_skel_best.pt",
    max_tokens=600, device="cuda",
)
facade.install(m, tok)

probes = [
    ("reverse_words", "Write a function to reverse words in a given string."),
    ("prime_num", "Write a function to check if the given integer is a prime number."),
    ("first_repeated_char",
     "Write a python function to find the first repeated character in a given string."),
]

for fn_name, prompt in probes:
    print("="*80)
    print(f"fn={fn_name} prompt={prompt}")
    # Stock
    r = facade.solve(prompt, fn_name, use_bias=False)
    print(f"[STOCK] used_bias={r.used_bias}")
    print(f"  raw generated (first 500 chars):\n  {r.generated[:500]!r}")
    print()
    # DT-biased
    r2 = facade.solve(prompt, fn_name, use_bias=True)
    print(f"[DT]    used_bias={r2.used_bias}  skel={r2.skeleton}")
    print(f"  raw generated (first 500 chars):\n  {r2.generated[:500]!r}")
    print()

print("DONE")
