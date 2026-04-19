"""Debug: dump what Gemma actually outputs for a coding problem under
our eval prompt. Runs one short generation on stock + hinted prompts,
dumps full text so we can see why extract_function returns nothing.
"""

from __future__ import annotations


def _reload_facades():
    import sys
    for m in list(sys.modules.keys()):
        if (m.startswith("calm.llm_computer.facades.")
                or m == "calm.llm_computer.facades"):
            del sys.modules[m]


def debug(m, tok):
    _reload_facades()
    from calm.llm_computer.facades.code_example_db import CodeExampleDB
    from calm.llm_computer.facades.code_verifier import CodeVerifierFacade

    db = CodeExampleDB.load_default()
    db.load_indices("/mnt/c/Users/gabes/projects/claw-code/.cache/r53_code_db")
    facade = CodeVerifierFacade(db=db)

    prompt_stock = (
        "You are a careful, correct Python coding assistant. "
        "Output ONLY the requested function — no prose, no markdown fences.\n\n"
        "Problem: Write a Python function `is_prime(n)` that returns True "
        "if n is a prime number, False otherwise.\n\n"
        "def is_prime(n):\n"
    )

    print("=" * 60)
    print("STOCK PROMPT:")
    print(prompt_stock)
    print("=" * 60)
    out = m.generate(prompt_stock, tok, max_tokens=150, device="cuda",
                     stop_on_eos=True)
    print("STOCK OUTPUT:")
    print(repr(out["text"][:800]))
    print("---")
    print(out["text"][:800])
    print("=" * 60)

    # Also try a chat-template prompt
    chat_prompt = (
        "<start_of_turn>user\n"
        "Write a Python function `is_prime(n)` that returns True if n is a prime number, False otherwise. Output only the function.\n"
        "<end_of_turn>\n"
        "<start_of_turn>model\n"
    )
    print("CHAT-TEMPLATE PROMPT:")
    print(chat_prompt)
    print("=" * 60)
    out2 = m.generate(chat_prompt, tok, max_tokens=150, device="cuda",
                       stop_on_eos=True)
    print("CHAT OUTPUT:")
    print(repr(out2["text"][:800]))
    print("---")
    print(out2["text"][:800])


if "m" in globals() and "tok" in globals():
    debug(m, tok)                                        # noqa: F821
