"""MathAdditionFacade — single-digit addition as a prod-Gemma domain.

Packages Round 7's inline wiring into a reusable class. One line to
install, one line to switch prompts. All CardSlots / Router / hook
state lives on the instance so facades compose without leaking
closures.

Layout inside Gemma's layer 33 (SWA, shared-KV, CardSlot-only — no
FP32 conversion needed):

  ch[2400:2480] : PT output (one-hot, 80 chars)
  ch[2480:2488] : adder_tiny output (8 digit slots)
  sub-heads     : none consumed (CardSlot pattern, not in-attention)

Usage:

    facade = MathAdditionFacade(pt_ckpt_path="calm/hrm/checkpoints/"
                                 "copy_augmented_hrm_best.pt")
    facade.install(gemma_substrate)

    facade.set_prompt("what is 2 plus 3")
    logits = gemma.forward(tok("what is 2 plus 3 equals"), ...)
    # logits argmax is Gemma's '5' token, boosted by adder's verified 5.

Adding a second facade is a disjoint-range allocation + another
install() call. See `.claude/MEMORY/substrate_registry.md` for the
allocation table.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from calm.hrm.data import _CHAR_TO_ID, _ID_TO_CHAR
from calm.llm_computer.card_router import CardRouter, Route
from calm.llm_computer.copy_augmented import (
    CopyAugmentedConfig, CopyAugmentedTransformer,
)
from calm.llm_computer.programs.adder_tiny import build_adder_tiny


# Gemma 4 E4B BPE token IDs for single digits 0..9.
_DIGIT_TO_GEMMA = {
    0: 236771, 1: 236770, 2: 236778, 3: 236800, 4: 236812,
    5: 236810, 6: 236825, 7: 236832, 8: 236828, 9: 236819,
}


class _AutoregPT(nn.Module):
    """Wraps CopyAugmentedTransformer so its forward returns the full
    autoregressively-decoded sequence as a (1, G, V) log-prob tensor.
    Private because the facade owns it; callers interact via the
    facade API."""

    def __init__(self, pt_model: CopyAugmentedTransformer, max_gen: int = 12):
        super().__init__()
        self.pt = pt_model
        self.max_gen = max_gen
        self.config = pt_model.config
        self._eos_id = _CHAR_TO_ID["<eos>"]

    def forward(self, prefix_ids: torch.Tensor) -> torch.Tensor:
        v = self.pt.config.vocab_size
        with torch.no_grad():
            ids = prefix_ids.clone()
            rows = []
            for _ in range(self.max_gen):
                log_probs = self.pt(ids)
                last = log_probs[0, -1]
                nxt = int(last.argmax())
                rows.append(last)
                if nxt == self._eos_id:
                    break
                ids = torch.cat(
                    [ids, torch.tensor([[nxt]], device=ids.device)],
                    dim=1,
                )
            if not rows:
                return torch.zeros(1, 1, v, device=prefix_ids.device)
            return torch.stack(rows, dim=0).unsqueeze(0)


@dataclass
class _Allocation:
    """Channel / layer reservation for a facade install. Mirrors the
    row format in substrate_registry.md."""
    layer: int
    pt_ch: tuple[int, int]
    adder_ch: tuple[int, int]


class MathAdditionFacade:
    """Single-digit addition domain for prod Gemma 4 E4B.

    One instance = one install site. The instance holds all state
    (PT, compute card, router, CardSlots, hook, per-prompt PT input,
    parse-success flag). `install()` attaches to a GemmaSubstrate;
    `detach()` reverses it (useful for A/B and tests).
    """

    # Reasonable defaults; override in __init__ for custom allocation.
    DEFAULT_LAYER = 33
    DEFAULT_CH_BASE = 2400
    PT_VOCAB = 80       # copy_augmented_hrm_best.pt was trained at 80
    ADDER_VOCAB = 8     # adder_tiny.config.vocab_size
    PT_OPERATOR = "+"

    def __init__(
        self,
        pt_ckpt_path: str | Path =
            "calm/hrm/checkpoints/copy_augmented_hrm_best.pt",
        layer: int = DEFAULT_LAYER,
        ch_base: int = DEFAULT_CH_BASE,
        verify_boost: float = 50.0,
        min_margin: float = 0.5,
        device: str = "cuda",
        max_gen: int = 10,
    ):
        self.layer = layer
        self.alloc = _Allocation(
            layer=layer,
            pt_ch=(ch_base, ch_base + self.PT_VOCAB),
            adder_ch=(ch_base + self.PT_VOCAB,
                      ch_base + self.PT_VOCAB + self.ADDER_VOCAB),
        )
        self.verify_boost = verify_boost
        self.min_margin = min_margin
        self.device = device

        # Load PT from checkpoint
        ckpt = torch.load(str(pt_ckpt_path), weights_only=False,
                           map_location=device)
        cfg = CopyAugmentedConfig(**ckpt["config"])
        raw_pt = CopyAugmentedTransformer(cfg).to(device).eval()
        raw_pt.load_state_dict(ckpt["model_state_dict"])
        self.pt = _AutoregPT(raw_pt, max_gen=max_gen).to(device).eval()

        # Compute card
        self.adder = build_adder_tiny().to(device).eval()

        # Router: PT output → adder input tokens
        self.router = CardRouter(id_to_char=_ID_TO_CHAR)

        # Mutable state passed via instance (not outer-scope closures,
        # so multiple facades don't clobber each other).
        self._pt_input_ids: Optional[torch.Tensor] = None
        self._parse_ok: bool = False

        self.router.register(Route(
            source_ch=self.alloc.pt_ch,
            operator=self.PT_OPERATOR,
            target_card_slot=None,  # set in install()
            translator=self._translate_adder_operands,
        ))

        # Filled in by install()
        self._pt_slot = None
        self._adder_slot = None
        self._hook = None
        self._installed_on = None

    # --- Public API ---

    def install(self, gemma):
        """Attach CardSlots + VerificationHook to a GemmaSubstrate."""
        from calm.llm_computer.gemma_substrate import (
            CardSlot, VerificationHook,
        )
        if self._installed_on is not None:
            raise RuntimeError(
                f"facade already installed on {self._installed_on!r}; "
                f"call detach() first")

        pt_ch_lo, pt_ch_hi = self.alloc.pt_ch
        ad_ch_lo, ad_ch_hi = self.alloc.adder_ch

        self._pt_slot = CardSlot(
            layer_idx=self.layer, ch_off=pt_ch_lo, card=self.pt,
            d_card=self.PT_VOCAB,
            card_input_fn=self._pt_card_input,
            use_full_residual=False,
            output_fn=self._pt_writer,
        )
        self._pt_slot.attach(gemma, preserve=True)

        self._adder_slot = CardSlot(
            layer_idx=self.layer, ch_off=ad_ch_lo, card=self.adder,
            d_card=self.ADDER_VOCAB,
            card_input_fn=self._adder_card_input,
            use_full_residual=True,
            output_fn=self._adder_writer,
        )
        self._adder_slot.attach(gemma, preserve=True)
        # Now complete the router route's back-pointer.
        self.router.routes[0].target_card_slot = self._adder_slot

        self._hook = VerificationHook(
            self._adder_slot, vocab_mapping=_DIGIT_TO_GEMMA,
            boost=self.verify_boost, min_margin=self.min_margin,
        )
        gemma.verification_hooks.append(self._hook)

        self._installed_on = gemma

    def detach(self, gemma=None):
        """Remove CardSlots + hook. Reverse of install()."""
        gemma = gemma or self._installed_on
        if gemma is None:
            return
        layer = gemma.layers[self.layer]
        for slot in (self._pt_slot, self._adder_slot):
            if slot is not None and hasattr(layer, "card_slots"):
                if slot in layer.card_slots:
                    layer.card_slots.remove(slot)
        if self._hook is not None and self._hook in gemma.verification_hooks:
            gemma.verification_hooks.remove(self._hook)
        # Clear reserved_channels entries we added via preserve=True.
        gemma.reserved_channels = [
            r for r in gemma.reserved_channels
            if r[2] != self.layer or r[0] not in
               (self.alloc.pt_ch[0], self.alloc.adder_ch[0])
        ]
        self._pt_slot = None
        self._adder_slot = None
        self._hook = None
        self._installed_on = None

    def set_prompt(self, nl_prompt: str):
        """Preprocess a natural-language prompt into the PT's input
        tokens for the next forward. Must be called before each
        distinct prompt."""
        bos = _CHAR_TO_ID["<bos>"]
        sep = _CHAR_TO_ID["<sep>"]
        ids = ([bos]
               + [_CHAR_TO_ID[c] for c in nl_prompt.lower()
                  if c in _CHAR_TO_ID]
               + [sep])
        self._pt_input_ids = torch.tensor([ids], device=self.device)

    # --- Internal: CardSlot callbacks ---

    def _translate_adder_operands(self, operands: list[int]
                                   ) -> torch.Tensor:
        a, b = operands
        a = max(0, min(self.ADDER_VOCAB - 1, int(a)))
        b = max(0, min(self.ADDER_VOCAB - 1, int(b)))
        return torch.tensor([[a, b]])

    def _pt_card_input(self, h: torch.Tensor) -> torch.Tensor:
        if self._pt_input_ids is None:
            raise RuntimeError(
                "set_prompt() must be called before Gemma forward")
        return self._pt_input_ids

    def _pt_writer(self, h, card_out, ch_lo, ch_hi):
        # Bounded one-hot: raw log-probs (range [-100, 0]) would warp
        # output_norm. One-hot keeps residual magnitude ~1 and still
        # lets Router.decode_pt_output argmax recover the tokens.
        h[..., ch_lo:ch_hi] = 0.0
        h[..., ch_lo] = 1.0  # channel 0 = <pad> → Router filters
        B, G, V = card_out.shape
        S = h.shape[1]
        G_eff = min(G, S)
        tokens = card_out[0, -G_eff:, :].argmax(dim=-1)
        pos = S - G_eff
        h[..., pos:pos + G_eff, ch_lo] = 0.0
        for i, tok in enumerate(tokens.tolist()):
            ch = ch_lo + int(tok)
            if ch < ch_hi:
                h[..., pos + i, ch] = 1.0
        return h

    def _adder_card_input(self, h: torch.Tensor) -> torch.Tensor:
        text = self.router.decode_pt_output(h, *self.alloc.pt_ch)
        operands = CardRouter._parse_operands(text, self.PT_OPERATOR)
        if operands is None or len(operands) < 2:
            self._parse_ok = False
            return torch.tensor([[0, 0]], device=h.device)
        self._parse_ok = True
        return self._translate_adder_operands(operands).to(h.device)

    def _adder_writer(self, h, card_out, ch_lo, ch_hi):
        h[..., ch_lo:ch_hi] = 0.0
        if self._parse_ok:
            ans = card_out[:, -1:, :]
            h[..., -1:, ch_lo:ch_hi] = ans
        else:
            # Zero card_out in-place so slot.last_output has margin=0
            # and VerificationHook stays silent.
            card_out.zero_()
        return h
