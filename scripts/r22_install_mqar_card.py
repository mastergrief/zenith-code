"""R22 — install PT+Delta MQAR card on prod Gemma via CardSlot.

First prototype of the R21 deployable card (copy_augmented_delta_mqar_best.pt,
100% held-out on N=5/10/15) installed inside Gemma's residual stream.

Design:
1. Parse prompt for <mem>k=v k=v ...</mem> block + "what is the value of X?"
   query pattern. Format as MQAR input "a 3 b 7 c 1 ; b".
2. Char-level tokenize via _CHAR_TO_ID (82-char vocab).
3. Stash tokens on `state` dict (closure); `card_input_fn(h)` returns the
   stashed tensor. Pattern lifted from gemma_learning_loop_demo.py.
4. CardSlot.attach at L30 with preserve=True — card's output channels
   survive through output_norm.
5. VerificationHook maps card's argmax digit (char 0-9) to Gemma's BPE
   digit token, biases Gemma's final logit by +boost when card is active
   AND the card's confidence (peak vs median) exceeds min_margin.
6. Inactive prompts (no <mem> block) feed a null input that produces
   low-confidence logits → min_margin gates the hook silent → Gemma
   output unchanged.

Test cases:
- "<mem>a=3 b=7 c=1</mem> What is the value of b?" → expect 7
- "<mem>x=5 y=2</mem> What is the value of x?" → expect 5
- "Hello how are you?" → no <mem>, Gemma unchanged

Usage: bin/gemma-run scripts/r22_install_mqar_card.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import torch

# Daemon pre-binds m, tok. Fall back to manual load for CLI use.
if "m" not in globals():  # type: ignore[name-defined]
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from calm.llm_computer.gemma_substrate import (
        GemmaSubstrate, enable_triton_tq4
    )
    from calm.llm_computer.tokenizer import GemmaTokenizer  # type: ignore
    gguf = Path.home() / "models" / "gemma-4-E4B-it-tq4-aligned.gguf"
    enable_triton_tq4(True)
    m = GemmaSubstrate.from_gguf(str(gguf), device="cuda")  # type: ignore
    tok = GemmaTokenizer.from_gguf(str(gguf))  # type: ignore

from calm.hrm.data import _CHAR_TO_ID, _ID_TO_CHAR, VOCAB_SIZE
from calm.llm_computer.copy_augmented_delta import (
    CopyAugmentedDeltaConfig, build_copy_augmented_delta,
)
from calm.llm_computer.gemma_substrate import (
    CardSlot, VerificationHook, KVCache,
)


# ============================================================================
# Adapter — NL <mem>k=v ...</mem> prompt → MQAR card input
# ============================================================================

_MEM_RE = re.compile(r"<mem>(.+?)</mem>", re.IGNORECASE | re.DOTALL)
_KV_RE = re.compile(r"\b([a-z])\s*=\s*(\d)\b")
# Matches: "value of X", "the value of X", "what is X"
_QUERY_RES = [
    re.compile(r"value of\s+([a-z])\b", re.IGNORECASE),
    re.compile(r"what is\s+([a-z])\b[?\.]?\s*$", re.IGNORECASE),
    re.compile(r"\bis\s+([a-z])\b\s*[?\.]?\s*$", re.IGNORECASE),
]


def parse_mqar_prompt(prompt: str) -> str | None:
    """Extract <mem>...</mem> + query key. Returns 'a 3 b 7 c 1 ; b' or None."""
    mem = _MEM_RE.search(prompt)
    if not mem:
        return None
    pairs = _KV_RE.findall(mem.group(1))
    if not pairs:
        return None
    post_mem = prompt[mem.end():]
    q_key = None
    for q_re in _QUERY_RES:
        m = q_re.search(post_mem)
        if m:
            q_key = m.group(1).lower()
            break
    if q_key is None:
        return None
    keys = [p[0].lower() for p in pairs]
    if q_key not in keys:
        return None
    body = " ".join(f"{k.lower()} {v}" for k, v in pairs)
    return f"{body} ; {q_key}"


def mqar_to_ids(mqar_str: str) -> list[int]:
    """Encode MQAR string as [<bos>, ...chars, <sep>] per train_pt_delta_mqar."""
    ids = [_CHAR_TO_ID["<bos>"]]
    for c in mqar_str:
        if c in _CHAR_TO_ID:
            ids.append(_CHAR_TO_ID[c])
    ids.append(_CHAR_TO_ID["<sep>"])
    return ids


# ============================================================================
# Card load
# ============================================================================

def load_mqar_card(ckpt_path: str | Path, device="cuda"):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg_dict = ckpt["config"]
    card = build_copy_augmented_delta(
        vocab_size=cfg_dict["vocab_size"],
        d_model=cfg_dict["d_model"],
        n_heads=cfg_dict["n_heads"],
        n_layers=cfg_dict["n_layers"],
        d_ffn=cfg_dict["d_ffn"],
        max_len=cfg_dict["max_len"],
        n_copy_heads=cfg_dict.get("n_copy_heads", 4),
    )
    card.config.use_chunkwise = cfg_dict.get("use_chunkwise", True)
    card.config.chunk_size = cfg_dict.get("chunk_size", 32)
    card.load_state_dict(ckpt["model_state_dict"])
    card.to(device).eval()
    for p in card.parameters():
        p.requires_grad = False
    return card


# ============================================================================
# Install
# ============================================================================

def install(m, card, layer_idx=30, ch_off=2480,
            write_margin: float = 0.0, preserve: bool = True):
    """Attach card + VerificationHook. Returns (slot, state) closure handle.

    `write_margin`: if > 0, card_output_fn skips the residual-stream write
    when card's (peak - median) margin is below this threshold. Prevents
    low-confidence card outputs from shifting Gemma's head projection via
    the reserved channels. Round 5 showed that without this gate, the
    residual write affects Gemma even when VerificationHook is silent.

    `preserve`: passed through to CardSlot.attach. preserve=True masks
    subsequent layers' contributions to reserved channels — may subtly
    affect Gemma even when card didn't write (round 6 `q=v margin=0.00`
    regression hypothesis). preserve=False lets Gemma's L31+ freely
    overwrite, but then card's write (if any) only affects Gemma via
    the same layer's residual propagation plus VerificationHook.
    """
    # Remove any stale slots + hooks from prior daemon runs — the daemon's
    # globals persist, so `m.layers[layer_idx].card_slots` and
    # `m.verification_hooks` accumulate across script reloads unless cleaned.
    layer = m.layers[layer_idx]
    if hasattr(layer, "card_slots"):
        layer.card_slots = []
    m.verification_hooks = []
    # Also reset reserved_channels so preserve=True masking doesn't hold
    # stale ranges.
    m.reserved_channels = [
        entry for entry in getattr(m, "reserved_channels", [])
        if entry[2] != layer_idx
    ]

    # State stash — updated by caller before each forward pass.
    state = {"mqar_ids": None, "active": False}

    # Null input — one <pad> token produces ~uniform logits, min_margin kills it.
    _NULL = torch.tensor([[_CHAR_TO_ID["<pad>"]]], device="cuda")

    def card_input_fn(h):
        if state["active"] and state["mqar_ids"] is not None:
            return torch.tensor([state["mqar_ids"]], dtype=torch.long,
                                device="cuda")
        return _NULL

    def card_output_fn(h, logits, ch_lo, ch_hi):
        if not state["active"]:
            # Zero the card output in-place so VerificationHook sees flat
            # logits (peak - median = 0 < min_margin) and stays silent.
            # `slot.last_output = card_out` is assigned AFTER output_fn by
            # _forward_layer, so in-place zero propagates.
            logits.zero_()
            return h  # leave residual untouched
        # Margin gate the residual write (R22b round 6 fix): without this,
        # card's log-probs are written to [ch_off:ch_hi] regardless of
        # confidence, shifting Gemma's head projection. Gate mirrors
        # VerificationHook's margin check.
        if write_margin > 0.0:
            last = logits[0, -1].float()
            margin = (last.max() - last.median()).item()
            if margin < write_margin:
                logits.zero_()
                return h
        ans = logits[:, -1:, :]  # (B, 1, vocab=82)
        # Clamp to actual residual width (ch_hi may exceed d_model if the
        # card vocab is larger than available channels).
        d_model = h.shape[-1]
        hi = min(ch_hi, d_model)
        n = hi - ch_lo
        # Write card's last-step logits into reserved channels at the LAST
        # Gemma position (where VerificationHook reads).
        h[..., -1:, ch_lo:hi] = h[..., -1:, ch_lo:hi] + ans[..., :n]
        return h

    slot = CardSlot(
        layer_idx=layer_idx, ch_off=ch_off, card=card,
        d_card=VOCAB_SIZE,
        card_input_fn=card_input_fn,
        use_full_residual=True,
        output_fn=card_output_fn,
    )
    slot.attach(m, preserve=preserve)

    # Card digit char IDs → Gemma BPE digit token IDs.
    # Gemma tokenizes digits as "▁0", "▁1", ... (space-prefixed). We want the
    # token that would naturally follow "... equals " (digit after whitespace).
    gemma_digit_ids = {}
    for d in range(10):
        # Encode " d" and take the last token (skips BOS + leading ' ').
        enc = tok.encode(f" {d}")
        gemma_digit_ids[d] = enc[-1]

    vocab_mapping = {
        _CHAR_TO_ID[str(d)]: gemma_digit_ids[d] for d in range(10)
    }

    hook = VerificationHook(
        slot, vocab_mapping=vocab_mapping,
        boost=50.0, min_margin=0.5,
    )
    m.verification_hooks.append(hook)
    return slot, state, hook


# ============================================================================
# Main
# ============================================================================

def main():
    ckpt_path = Path(__file__).resolve().parent.parent / \
        "calm/hrm/checkpoints/copy_augmented_delta_mqar_best.pt"
    print(f"[r22] loading MQAR card: {ckpt_path.name}")
    card = load_mqar_card(ckpt_path)
    n_params = sum(p.numel() for p in card.parameters())
    print(f"[r22] card loaded: {n_params:,} params, "
          f"vocab={card.config.vocab_size}, d_model={card.config.d_model}, "
          f"chunkwise={card.config.use_chunkwise}")

    # Sanity: card standalone on a known MQAR problem
    print("\n=== SANITY: card standalone ===")
    test_mqar = "a 3 b 7 c 1 ; b"
    ids = torch.tensor([mqar_to_ids(test_mqar)], device="cuda")
    with torch.no_grad():
        out = card(ids)  # (1, S, vocab)
    top_id = int(out[0, -1].argmax())
    top_char = _ID_TO_CHAR.get(top_id, "?")
    print(f"  input: '{test_mqar}' → card predicts '{top_char}' (expected '7')")
    assert top_char == "7", f"Card sanity failed: got {top_char!r}"
    print("  ✓ card is functional")

    # Baseline: measure stock Gemma on all prompts BEFORE installing
    baseline_prompts = [
        "<mem>a=3 b=7 c=1</mem>\nWhat is the value of b? Answer: ",
        "<mem>x=5 y=2 z=8</mem>\nWhat is the value of x? Answer: ",
        "<mem>p=9 q=4</mem>\nWhat is the value of q? Answer: ",
        "2 plus 3 equals ",
    ]
    print("\n=== BASELINE: stock Gemma (no card) ===")
    baseline_tops = {}
    # Reset any prior slots first
    for lyr in m.layers:
        if hasattr(lyr, "card_slots"):
            lyr.card_slots = []
    m.verification_hooks = []
    m.reserved_channels = []
    for prompt in baseline_prompts:
        ids = tok.encode(prompt)
        cache = KVCache(m.config.n_layers, device="cuda")
        with torch.no_grad():
            logits = m.forward(
                torch.tensor([ids]), device="cuda",
                kv_cache=cache, start_pos=0,
            )
        top = int(logits[0, -1].argmax())
        baseline_tops[prompt] = tok.id_to_token.get(top, "?")
        print(f"  {prompt!r} → {baseline_tops[prompt]!r}")

    # Install
    print("\n=== INSTALL: CardSlot @ L30 ch[2480:2562] + VerificationHook ===")
    slot, state, hook = install(m, card, layer_idx=30, ch_off=2480)
    print(f"  slot: layer={slot.layer_idx} ch_off={slot.ch_off} d_card={slot.d_card}")
    print(f"  vocab_mapping: card digit chars → Gemma BPE tokens")

    # Test prompts
    print("\n=== TEST: Gemma + MQAR card ===")
    test_cases = [
        ("<mem>a=3 b=7 c=1</mem>\nWhat is the value of b? Answer: ", "7"),
        ("<mem>x=5 y=2 z=8</mem>\nWhat is the value of x? Answer: ", "5"),
        ("<mem>p=9 q=4</mem>\nWhat is the value of q? Answer: ", "4"),
        # Regression guard: no <mem>, card should not fire
        ("2 plus 3 equals ", None),
    ]

    # Also capture Gemma baseline (card OFF) for each prompt
    results = []
    for prompt, expected in test_cases:
        # Parse and stash
        mqar_str = parse_mqar_prompt(prompt)
        if mqar_str:
            state["mqar_ids"] = mqar_to_ids(mqar_str)
            state["active"] = True
            parsed = mqar_str
        else:
            state["mqar_ids"] = None
            state["active"] = False
            parsed = "(no <mem>)"

        ids = tok.encode(prompt)
        cache = KVCache(m.config.n_layers, device="cuda")
        with torch.no_grad():
            logits = m.forward(
                torch.tensor([ids]), device="cuda",
                kv_cache=cache, start_pos=0,
            )
        top = int(logits[0, -1].argmax())
        top_tok = tok.id_to_token.get(top, "?")
        # Strip leading space for digit comparison
        got = top_tok.lstrip(" ▁")
        ok_marker = ""
        if expected is not None:
            ok_marker = "✓" if got == expected else "✗"
        print(f"\n  prompt: {prompt!r}")
        print(f"    parsed MQAR: {parsed}")
        print(f"    active: {state['active']}  →  Gemma top: {top_tok!r}  "
              f"expected: {expected}  {ok_marker}")
        results.append((prompt, expected, got, top_tok, state["active"]))

    print("\n=== SUMMARY ===")
    n_pass = sum(
        1 for _, exp, got, _, _ in results
        if exp is not None and got == exp
    )
    n_testable = sum(1 for _, exp, *_ in results if exp is not None)
    print(f"  {n_pass}/{n_testable} MQAR retrievals correct")
    # Baseline-diff check: for each prompt, is the with-card answer == baseline?
    print("\n  baseline vs with-card:")
    for prompt, exp, got, top_tok, active in results:
        base = baseline_tops.get(prompt, "?")
        same = (top_tok == base)
        marker = ""
        if exp is not None:
            marker = f"  (expected {exp!r})"
        elif same:
            marker = "  (match — no regression)"
        else:
            marker = "  (REGRESSION on inactive prompt)"
        print(f"    baseline={base!r:>5}  with-card={top_tok!r:>5}"
              f"  active={active}{marker}")


main()
print("R22_DONE")
