"""Full reasoning stack in one SWA layer (Round 7).

Composes end-to-end inside layer 33 (SWA, shared-KV range, no FP32
conversion required — CardSlot doesn't edit Gemma's attention tensors):

  1. PT (`copy_augmented_hrm_best.pt`, 181K params, 100% autoreg on NL
     math) — CardSlot that autoregressively decodes the expression
     from the NL prompt. Writes one-hot log-probs for each decoded
     token to residual ch [2400:2480] at the trailing positions.
  2. `adder_tiny` compiled card (1,020 params, 16/16 exhaustive,
     0+0=0 … 3+4=7) — CardSlot that reads PT's output via the Round-4
     CardRouter. Router parses "2 + 3" from residual, hands
     adder_tiny `[[2, 3]]` tokens. adder_tiny's argmax logit goes to
     channels [2480:2488].
  3. VerificationHook with min_margin=0.5 reads adder_tiny's argmax
     and biases Gemma's corresponding BPE digit logit by +50.

Gemma's native forward is untouched — layer 33's attn/ffn/tq4 path
still runs. Two CardSlots append after the layer, their output flows
through `preserve=True` channel masking to output_norm + head.
Baseline vs post-install must: (a) fix Gemma on small addition
prompts, (b) not regress on unrelated completions.
"""

from __future__ import annotations

import os
import sys
import time

import torch

from calm.hrm.data import _CHAR_TO_ID, _ID_TO_CHAR
# PT checkpoint trained with vocab_size=80 (pre-dating the '<','>' chars
# added later to _CHAR_TO_ID for 82 entries). PT.forward returns
# (B, S, 80) — d_card and channel range must match the PT's actual
# output dim, not the tokenizer's full alphabet.
PT_VOCAB = 80


GGUF_PATH = os.path.expanduser("~/models/gemma-4-E4B-it-tq4-aligned.gguf")

# Gemma 4 E4B BPE token IDs for single digits 0..9.
DIGIT_TO_GEMMA = {
    0: 236771, 1: 236770, 2: 236778, 3: 236800, 4: 236812,
    5: 236810, 6: 236825, 7: 236832, 8: 236828, 9: 236819,
}

ADDER_VOCAB = 8  # adder_tiny.config.vocab_size
ADDER_CH_LO = 2480
ADDER_CH_HI = ADDER_CH_LO + ADDER_VOCAB
PT_CH_LO = 2400
PT_CH_HI = PT_CH_LO + PT_VOCAB
LAYER = 33  # SWA, shared-KV (≥24), no FP32 conversion needed


# Prompts using "what is X plus Y" — the phrasing the PT was trained on
# (see calm/hrm/nl_data.py _TEMPLATES).
DOMAIN_PROMPTS = [
    "what is 2 plus 3",
    "what is 4 plus 1",
    "what is 3 plus 2",
    "what is 5 plus 1",
    "what is 2 plus 4",
    "what is 1 plus 6",
    "what is 3 plus 4",
]

REGRESSION_PROMPTS = [
    "The capital of France is",
    "The capital of Germany is",
    "The capital of Italy is",
]


# --------- PT wrapper that does autoreg decode inside forward ---------

class AutoregPT(torch.nn.Module):
    """Wraps CopyAugmentedTransformer so its forward produces the full
    autoregressively-decoded expression as a sequence of one-hot log-
    probs.

    Input shape: (B=1, P) — <bos> + prompt tokens + <sep>.
    Output shape: (1, G, PT_VOCAB) where G = # decoded tokens (≤ max_gen).
    """

    def __init__(self, pt_model, max_gen: int = 12):
        super().__init__()
        self.pt = pt_model
        self.max_gen = max_gen
        self.config = pt_model.config
        self.bos = _CHAR_TO_ID["<bos>"]
        self.sep = _CHAR_TO_ID["<sep>"]
        self.eos = _CHAR_TO_ID["<eos>"]

    def forward(self, prefix_ids: torch.Tensor) -> torch.Tensor:
        v = self.pt.config.vocab_size
        with torch.no_grad():
            ids = prefix_ids.clone()
            all_log_probs = []
            for _ in range(self.max_gen):
                log_probs = self.pt(ids)  # (1, S, V)
                last = log_probs[0, -1]
                next_tok = int(last.argmax())
                all_log_probs.append(last)
                if next_tok == self.eos:
                    break
                ids = torch.cat(
                    [ids, torch.tensor([[next_tok]], device=ids.device)],
                    dim=1,
                )
            if not all_log_probs:
                return torch.zeros(1, 1, v, device=prefix_ids.device)
            # (1, G, V) — each row is the log-prob vector that produced
            # one decoded token.
            out = torch.stack(all_log_probs, dim=0).unsqueeze(0)
        return out


# --------- Helpers ---------

def gemma_last_argmax(m, tok, prompt: str) -> int:
    from calm.llm_computer.gemma_substrate import KVCache
    ids = tok.encode(prompt)
    cache = KVCache(m.config.n_layers, device="cuda")
    with torch.no_grad():
        logits = m.forward(torch.tensor([ids]), device="cuda",
                            kv_cache=cache, start_pos=0)
    return int(logits[0, -1].argmax().item())


def encode_for_pt(prompt: str) -> torch.Tensor:
    """Map a prompt string into <bos> + PT-vocab chars + <sep>."""
    bos = _CHAR_TO_ID["<bos>"]
    sep = _CHAR_TO_ID["<sep>"]
    ids = ([bos]
           + [_CHAR_TO_ID[c] for c in prompt.lower() if c in _CHAR_TO_ID]
           + [sep])
    return torch.tensor([ids], device="cuda")


def decode_label(tok, tok_id: int) -> str:
    return tok.id_to_token.get(tok_id, f"?{tok_id}")


def main():
    from calm.llm_computer.gemma_substrate import (
        GemmaSubstrate, enable_triton_tq4, CardSlot, VerificationHook,
    )
    from calm.llm_computer.synth.gemma_tokenizer import GemmaTokenizer
    from calm.llm_computer.copy_augmented import (
        CopyAugmentedTransformer, CopyAugmentedConfig,
    )
    from calm.llm_computer.programs.adder_tiny import build_adder_tiny
    from calm.llm_computer.card_router import CardRouter, Route

    # 1. Gemma
    enable_triton_tq4(True)
    print("[full-stack] loading substrate...")
    m = GemmaSubstrate.from_gguf(GGUF_PATH, max_len=256)
    m.preload_gpu("cuda")
    m.warmup(seq_lens=(1, 6, 8))
    tok = GemmaTokenizer.from_gguf(GGUF_PATH)

    # 2. PT (CopyAugmentedTransformer) — wrapped for autoreg decode
    print("[full-stack] loading PT...")
    ckpt = torch.load(
        "calm/hrm/checkpoints/copy_augmented_hrm_best.pt",
        weights_only=False, map_location="cuda",
    )
    cfg = CopyAugmentedConfig(**ckpt["config"])
    raw_pt = CopyAugmentedTransformer(cfg).cuda().eval()
    raw_pt.load_state_dict(ckpt["model_state_dict"])
    pt = AutoregPT(raw_pt, max_gen=10).cuda().eval()
    print(f"  PT params: {sum(p.numel() for p in raw_pt.parameters()):,}, "
          f"vocab={PT_VOCAB}")

    # 3. adder_tiny (compiled card)
    adder = build_adder_tiny().cuda().eval()
    print(f"  adder_tiny params: {sum(p.numel() for p in adder.parameters()):,}, "
          f"vocab={ADDER_VOCAB}")

    # 4. Router: read PT output channels [2400:2480], find '+', hand
    # adder_tiny [[a, b]] as int tokens. Track parse-success via outer-
    # scope flag so the writer can zero output on failure (keeps the
    # VerificationHook silent via its min_margin gate).
    router = CardRouter(id_to_char=_ID_TO_CHAR)
    parse_state = {"ok": False}

    def adder_translator(operands):
        a, b = operands
        # adder_tiny's vocab is [0..7]; clamp to stay in-domain.
        a = max(0, min(ADDER_VOCAB - 1, int(a)))
        b = max(0, min(ADDER_VOCAB - 1, int(b)))
        return torch.tensor([[a, b]])

    router.register(Route(
        source_ch=(PT_CH_LO, PT_CH_HI),
        operator="+",
        target_card_slot=None,  # filled in after slot construction
        translator=adder_translator,
    ))

    def adder_card_input(h):
        """Read PT output, parse '+', return adder input tokens. Sets
        parse_state['ok'] so the writer can suppress output on failure."""
        text = router.decode_pt_output(h, PT_CH_LO, PT_CH_HI)
        operands = CardRouter._parse_operands(text, "+")
        if operands is None or len(operands) < 2:
            parse_state["ok"] = False
            return torch.tensor([[0, 0]], device=h.device)
        parse_state["ok"] = True
        return adder_translator(operands).to(h.device)

    # 5. Wire PT CardSlot. Mutable outer-scope holds the per-prompt PT
    # input so the same facade handles multiple prompts.
    pt_input_state = {"ids": None}

    def pt_card_input(h):
        return pt_input_state["ids"]

    def pt_writer(h, card_out, ch_lo, ch_hi):
        # card_out: (1, G, PT_VOCAB). h: (B, S, d_model).
        # Raw PT log-probs have unbounded negative values (~-100 on
        # non-argmax tokens) — writing those into residual warps
        # output_norm and shifts Gemma's head. Encode as bounded
        # one-hot at each decoded position instead.
        h[..., ch_lo:ch_hi] = 0.0
        h[..., ch_lo] = 1.0  # channel 2400 = <pad> → Router filters
        B, G, V = card_out.shape
        S = h.shape[1]
        G_eff = min(G, S)
        tokens = card_out[0, -G_eff:, :].argmax(dim=-1)  # (G_eff,)
        # Clear the <pad> default at positions we're about to overwrite,
        # then set the argmax channel to 1.0.
        pos = S - G_eff
        h[..., pos:pos + G_eff, ch_lo] = 0.0
        for i, tok in enumerate(tokens.tolist()):
            ch = ch_lo + int(tok)
            if ch < ch_hi:
                h[..., pos + i, ch] = 1.0
        return h

    pt_slot = CardSlot(
        layer_idx=LAYER, ch_off=PT_CH_LO, card=pt, d_card=PT_VOCAB,
        card_input_fn=pt_card_input, use_full_residual=False,
        output_fn=pt_writer,
    )
    pt_slot.attach(m, preserve=True)

    # 6. Wire adder_tiny CardSlot. Writer zeros output on parse failure
    # so VerificationHook.min_margin (0.5 vs peak-median=0) keeps hook
    # silent. card_out is zeroed IN-PLACE because CardSlot assigns
    # `slot.last_output = card_out` after output_fn runs — the hook
    # reads that reference, not the residual.
    def adder_writer(h, card_out, ch_lo, ch_hi):
        h[..., ch_lo:ch_hi] = 0.0
        if parse_state["ok"]:
            ans = card_out[:, -1:, :]  # (1, 1, 8)
            h[..., -1:, ch_lo:ch_hi] = ans
        else:
            card_out.zero_()
        return h

    adder_slot = CardSlot(
        layer_idx=LAYER, ch_off=ADDER_CH_LO, card=adder, d_card=ADDER_VOCAB,
        card_input_fn=adder_card_input,
        use_full_residual=True, output_fn=adder_writer,
    )
    adder_slot.attach(m, preserve=True)
    router.routes[0].target_card_slot = adder_slot

    # 7. VerificationHook — only fire on confident adder output.
    hook = VerificationHook(
        adder_slot, vocab_mapping=DIGIT_TO_GEMMA,
        boost=50.0, min_margin=0.5,
    )
    m.verification_hooks.append(hook)
    print(f"[full-stack] both cards installed at layer {LAYER}: "
          f"PT ch[{PT_CH_LO}:{PT_CH_HI}] + adder ch[{ADDER_CH_LO}:{ADDER_CH_HI}]")

    # ---------- Baseline Gemma (before cards fire) ----------
    # Quick way to measure baseline: pull out the hooks + slots, rerun,
    # re-attach. Simpler: measure baseline BEFORE install. But we already
    # installed. So for a clean baseline, compute expected digits ourselves
    # (addition is trivial) and measure post-install performance vs that.

    # The fairer comparison: run with hooks disabled (empty verification_hooks
    # and empty card_slots) to get baseline, then restore.
    saved_hooks = m.verification_hooks
    saved_slots = m.layers[LAYER].card_slots

    print("\n=== baseline Gemma (cards detached) ===")
    m.verification_hooks = []
    m.layers[LAYER].card_slots = []
    base_domain = []
    for prompt in DOMAIN_PROMPTS:
        got = gemma_last_argmax(m, tok, prompt + " equals")
        # Expected digit via trivial eval (we control the prompts).
        a, b = [int(s) for s in prompt.split() if s.isdigit()]
        expected = DIGIT_TO_GEMMA.get(a + b)
        match = got == expected if expected is not None else None
        base_domain.append((prompt, got, expected, match))
        print(f"  {'✓' if match else '✗'} {prompt!r:<24} "
              f"got={decode_label(tok, got)!r} "
              f"expected={decode_label(tok, expected)!r}")
    base_reg = [(p, gemma_last_argmax(m, tok, p)) for p in REGRESSION_PROMPTS]
    print(f"\n  regression baseline: "
          f"{[(p, decode_label(tok, t)) for p, t in base_reg]}")

    # Re-attach for the install phase.
    m.verification_hooks = saved_hooks
    m.layers[LAYER].card_slots = saved_slots

    # ---------- Post-install ----------
    print("\n=== full stack active (PT + router + adder + VH) ===")
    post_domain = []
    for prompt in DOMAIN_PROMPTS:
        pt_input_state["ids"] = encode_for_pt(prompt)
        got = gemma_last_argmax(m, tok, prompt + " equals")
        a, b = [int(s) for s in prompt.split() if s.isdigit()]
        expected = DIGIT_TO_GEMMA.get(a + b)
        match = got == expected if expected is not None else None
        post_domain.append((prompt, got, expected, match))
        print(f"  {'✓' if match else '✗'} {prompt!r:<24} "
              f"got={decode_label(tok, got)!r} "
              f"expected={decode_label(tok, expected)!r}")

    post_reg = []
    for prompt in REGRESSION_PROMPTS:
        # For non-math prompts, the PT will emit garbage but the router's
        # '+' match will fail, fallback_operands=[0,0] is used, adder
        # outputs 0+0=0 with confident logits → VerificationHook fires.
        # To suppress, set PT input to something that won't parse.
        pt_input_state["ids"] = encode_for_pt(prompt)
        got = gemma_last_argmax(m, tok, prompt)
        post_reg.append((prompt, got))

    print("\n  regression after install:")
    regressed = 0
    for (p, base), (_, post) in zip(base_reg, post_reg):
        changed = base != post
        regressed += changed
        mark = "✗ REGRESSED" if changed else "✓"
        print(f"  {mark} {p!r:<32} base={decode_label(tok, base)!r} "
              f"post={decode_label(tok, post)!r}")

    # ---------- Verdict ----------
    base_ok = sum(1 for _, _, _, m_ in base_domain if m_)
    post_ok = sum(1 for _, _, _, m_ in post_domain if m_)
    print("\n========== SUMMARY ==========")
    print(f"  domain baseline:  {base_ok}/{len(DOMAIN_PROMPTS)}")
    print(f"  domain post:      {post_ok}/{len(DOMAIN_PROMPTS)}")
    print(f"  fixes:            +{post_ok - base_ok}")
    print(f"  regressions:      {regressed}")
    ok = (post_ok > base_ok and regressed == 0)
    print(f"  verdict: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
