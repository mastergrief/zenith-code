"""Grammar-constrained decoding for math expression generation.

Inference-time mask that ensures the model can only produce syntactically
valid math expressions. Zero parameters, zero training, purely additive —
works with any model that outputs logits over the HRM char vocabulary.

Grammar (covers all current HRM output targets):
  expr     → term (('+' | '-') term)*
  term     → factor (('*' | '/') factor)*
  factor   → NUMBER | funcall | '(' expr ')'
  funcall  → IDENT '(' expr (',' ' ' expr)* ')'
  NUMBER   → DIGIT+ ('.' DIGIT+)?
  IDENT    → [a-z_]+

The mask is conservative: it allows all valid expressions and blocks
only guaranteed-invalid continuations (double operators, trailing
operators, operator after open paren, etc.).
"""

from __future__ import annotations

from calm.hrm.data import _CHAR_TO_ID

# Token ID sets
_DIGITS = frozenset(_CHAR_TO_ID[c] for c in "0123456789")
_OPS = frozenset(_CHAR_TO_ID[c] for c in "+-*/")
_OPEN = frozenset([_CHAR_TO_ID["("]])
_CLOSE = frozenset([_CHAR_TO_ID[")"]])
_COMMA = frozenset([_CHAR_TO_ID[","]])
_SPACE = frozenset([_CHAR_TO_ID[" "]])
_DOT = frozenset([_CHAR_TO_ID["."]])
_EQ = frozenset([_CHAR_TO_ID["="]])
_EOS = frozenset([_CHAR_TO_ID["<eos>"]])
_LETTERS = frozenset(_CHAR_TO_ID[c] for c in "abcdefghijklmnopqrstuvwxyz_")
_TERMINATORS = _EOS | _EQ  # expression can end with = or <eos>

# All expression tokens (digits, ops, parens, letters, dot, comma, space)
_ALL_EXPR = _DIGITS | _OPS | _OPEN | _CLOSE | _LETTERS | _DOT | _COMMA | _SPACE | _TERMINATORS


def allowed_next(last_id: int, paren_depth: int) -> frozenset[int]:
    """Return the set of allowed next token IDs given the last emitted token.

    Args:
        last_id: the token ID just emitted (or -1 for start of expression)
        paren_depth: current nesting depth of parentheses

    Returns:
        frozenset of allowed token IDs
    """
    closeable = _CLOSE if paren_depth > 0 else frozenset()

    # Start of expression: digit, open paren, letter (function name), minus (negative)
    if last_id == -1:
        return _DIGITS | _OPEN | _LETTERS | frozenset([_CHAR_TO_ID["-"]])

    # After digit: digit, operator, close paren, dot, terminator, space
    if last_id in _DIGITS:
        return _DIGITS | _OPS | closeable | _DOT | _TERMINATORS | _SPACE

    # After operator: digit, open paren, letter, space, minus (negative)
    if last_id in _OPS:
        return _DIGITS | _OPEN | _LETTERS | _SPACE | frozenset([_CHAR_TO_ID["-"]])

    # After open paren: digit, open paren, letter, minus (negative)
    if last_id in _OPEN:
        return _DIGITS | _OPEN | _LETTERS | frozenset([_CHAR_TO_ID["-"]])

    # After close paren: operator, close paren, terminator, space
    if last_id in _CLOSE:
        return _OPS | closeable | _TERMINATORS | _SPACE

    # After letter: letter (continue identifier), open paren (start funcall),
    #               underscore is in _LETTERS already
    if last_id in _LETTERS:
        return _LETTERS | _OPEN

    # After dot (decimal): digit
    if last_id in _DOT:
        return _DIGITS

    # After comma (function args): space, digit, letter
    if last_id in _COMMA:
        return _SPACE | _DIGITS | _LETTERS

    # After space: digit, letter, open paren, minus
    if last_id in _SPACE:
        return _DIGITS | _LETTERS | _OPEN | _OPS | frozenset([_CHAR_TO_ID["-"]])

    # After = : eos
    if last_id in _EQ:
        return _EOS

    # Fallback: allow everything (don't block if we can't determine state)
    return _ALL_EXPR


def _is_complete_expr(gen_ids: list[int]) -> bool:
    """Check if generated IDs form a complete valid expression.

    A complete expression ends with a digit or close paren (not an operator
    or open paren), and has balanced parentheses.
    """
    if not gen_ids:
        return False
    last = gen_ids[-1]
    # Must end with digit or close paren
    if last not in _DIGITS and last not in _CLOSE:
        return False
    # Balanced parens
    depth = 0
    for tid in gen_ids:
        if tid in _OPEN:
            depth += 1
        elif tid in _CLOSE:
            depth -= 1
        if depth < 0:
            return False
    return depth == 0


def constrained_decode(model, prefix_ids: list[int], device: str = "cpu",
                       max_gen: int = 30, eos_boost: float = 5.0) -> list[int]:
    """Autoregressive decode with grammar constraints + EOS boosting.

    Two mechanisms:
    1. Grammar mask: blocks syntactically invalid tokens (double ops, etc.)
    2. EOS boost: when the output is already a valid complete expression,
       adds `eos_boost` to the EOS logit. This prevents the model from
       hallucinating extra terms after a correct expression.

    Args:
        model: any model with forward(idx) → logits/log_probs (B, S, V)
        prefix_ids: token IDs for <bos> NL_tokens <sep> (the prompt)
        device: torch device
        max_gen: maximum tokens to generate
        eos_boost: bonus added to EOS logit when expression is complete

    Returns:
        list of generated token IDs (not including prefix, not including <eos>)
    """
    import torch

    eos_id = _CHAR_TO_ID["<eos>"]
    eq_id = _CHAR_TO_ID["="]
    vocab_size = len(_CHAR_TO_ID)

    ids = list(prefix_ids)
    gen = []
    last_id = -1  # start of expression
    paren_depth = 0

    for _ in range(max_gen):
        x = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            out = model(x)
        logits = out[0, -1].clone()  # (vocab,)

        # Apply grammar mask
        allowed = allowed_next(last_id, paren_depth)
        mask = torch.full((vocab_size,), float("-inf"), device=device)
        for tid in allowed:
            if tid < vocab_size:
                mask[tid] = 0.0
        logits = logits + mask

        # EOS boosting: if we already have a complete expression AND the
        # last token is a space (expression boundary), boost EOS. Don't
        # boost after a digit — we might be mid-number (78+4 looks
        # complete but 4 is an incomplete 41).
        if gen and gen[-1] in _SPACE and _is_complete_expr(
                [t for t in gen if t not in _SPACE]):
            logits[eos_id] = logits[eos_id] + eos_boost
            logits[eq_id] = logits[eq_id] + eos_boost

        nxt = int(logits.argmax().item())

        if nxt == eos_id or nxt == eq_id:
            break

        # Track paren depth
        if nxt in _OPEN:
            paren_depth += 1
        elif nxt in _CLOSE:
            paren_depth = max(0, paren_depth - 1)

        gen.append(nxt)
        ids.append(nxt)
        last_id = nxt

    return gen
