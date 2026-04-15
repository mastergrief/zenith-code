"""UnifiedSubstrateComputer — text → answer via ONE substrate hosting
Gemma-stand-in + SubstrateHRM + dispatched_v4.

The full-stack inference path: a single `HybridGroupedSmall2DTransformer`
contains three specialists at disjoint channel/sub-head/layer rectangles.
Query methods route text through the appropriate slot:

  * `card_query(text)` — regex parse → card tokens → substrate forward →
    decode slot in card vocab range. Verified arithmetic.

  * `hrm_parse(text)` — HRM's char-level tokenizer encodes the prompt,
    substrate forward autoregressively, argmax in HRM vocab range,
    decode back to an expression string like "3+5=". Uses HRM's trained
    weights inside the unified substrate.

  * `hrm_then_card(text)` — Brain+Cards chain: HRM parses NL to a
    math expression, regex pulls out (a, op, b), card computes the
    verified answer. All through the same substrate.

  * `gemma_residual(text)` — feed Gemma-stand-in tokens, run substrate
    forward, return Gemma channel residual. Demonstrates Gemma's slot
    is active in the same forward pass.

Default scale is reduced for dev speed (Gemma as a 128-dim random-init
stand-in). A later call can swap in real Gemma bytes from GGUF without
changing the inference API.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Union

import torch

from calm.hrm.data import _CHAR_TO_ID, _ID_TO_CHAR
from calm.llm_computer.hybrid_substrate import (
    HybridGroupedSmall2DConfig, HybridGroupedSmall2DTransformer,
    install_compiled_card_hybrid,
)
from calm.llm_computer.model import Small2DConfig, Small2DTransformer
from calm.llm_computer.programs.dispatched_v4 import (
    OPCODE_SHIFT, VOCAB as CARD_VOCAB, build_dispatched_v4, decode_output,
)
from calm.llm_computer.substrate_compute import parse_prompt


HRM_CKPT_PATH = (
    "/mnt/c/Users/gabes/projects/claw-code/calm/hrm/checkpoints/"
    "substrate_hrm_nl_best.pt"
)


# Default reduced-scale dimensions for the dev substrate.
# Real Gemma integration is its own demo (Round 14/15).
GEMMA_STANDIN_D_MODEL = 128
GEMMA_STANDIN_D_FFN = 256
GEMMA_STANDIN_VOCAB = 512
GEMMA_STANDIN_N_LAYERS = 2

HRM_D_MODEL = 64
HRM_D_FFN = 128
HRM_VOCAB = 80
HRM_N_LAYERS = 4


def _load_hrm():
    ckpt = torch.load(HRM_CKPT_PATH, weights_only=False, map_location="cpu")
    cfg = Small2DConfig(
        vocab_size=ckpt["config"]["vocab_size"],
        d_model=ckpt["config"]["d_model"],
        n_heads=ckpt["config"]["n_heads"],
        n_layers=ckpt["config"]["n_layers"],
        d_ffn=ckpt["config"]["d_ffn"],
        max_len=ckpt["config"]["max_len"],
        use_hard_max=False,
    )
    m = Small2DTransformer(cfg)
    m.load_state_dict(ckpt["model_state_dict"])
    m.eval()
    return m, ckpt


@dataclass
class SlotMap:
    """Offsets for each specialist's rectangle in the unified substrate."""
    ch_off: int
    sh_off: int
    ffn_off: int
    tok_off: int
    layer_off: int
    n_layers: int
    d_model: int
    vocab_size: int


class UnifiedSubstrateComputer:
    """One substrate hosts Gemma-stand-in + HRM + dispatched_v4. Four
    query methods dispatch text through the right slot."""

    def __init__(self, device: Union[str, torch.device] = "cpu"):
        self.device = torch.device(device)

        # Load real HRM checkpoint
        self.hrm, ckpt = _load_hrm()
        self.hrm_val_acc = ckpt["val_acc"]

        # Build real compiled card
        self.card = build_dispatched_v4()

        # Compute slot allocations
        self.gemma = SlotMap(
            ch_off=0, sh_off=0, ffn_off=0, tok_off=0, layer_off=0,
            n_layers=GEMMA_STANDIN_N_LAYERS,
            d_model=GEMMA_STANDIN_D_MODEL,
            vocab_size=GEMMA_STANDIN_VOCAB,
        )
        self.hrm_slot = SlotMap(
            ch_off=GEMMA_STANDIN_D_MODEL,
            sh_off=GEMMA_STANDIN_D_MODEL // 2,
            ffn_off=GEMMA_STANDIN_D_FFN,
            tok_off=GEMMA_STANDIN_VOCAB,
            layer_off=GEMMA_STANDIN_N_LAYERS,
            n_layers=HRM_N_LAYERS,
            d_model=HRM_D_MODEL,
            vocab_size=HRM_VOCAB,
        )
        self.card_slot = SlotMap(
            ch_off=GEMMA_STANDIN_D_MODEL + HRM_D_MODEL,
            sh_off=(GEMMA_STANDIN_D_MODEL + HRM_D_MODEL) // 2,
            ffn_off=GEMMA_STANDIN_D_FFN + HRM_D_FFN,
            tok_off=GEMMA_STANDIN_VOCAB + HRM_VOCAB,
            layer_off=GEMMA_STANDIN_N_LAYERS + HRM_N_LAYERS,
            n_layers=self.card.config.n_layers,
            d_model=self.card.config.d_model,
            vocab_size=self.card.config.vocab_size,
        )

        d_model = (GEMMA_STANDIN_D_MODEL + HRM_D_MODEL
                   + self.card.config.d_model)
        d_model += d_model % 2
        n_heads = d_model // 2
        d_ffn = (GEMMA_STANDIN_D_FFN + HRM_D_FFN + self.card.config.d_ffn)
        vocab = (GEMMA_STANDIN_VOCAB + HRM_VOCAB + self.card.config.vocab_size)
        n_layers = (GEMMA_STANDIN_N_LAYERS + HRM_N_LAYERS
                    + self.card.config.n_layers)

        # Layer linear types: all fp32 (Gemma stand-in uses fp32 for this
        # dev version; real Gemma would use tq4).
        layer_types = tuple(["fp32"] * n_layers)
        # Attention modes: all single. Hard_max only on card's layers.
        layer_hard_max = tuple(
            [False] * (GEMMA_STANDIN_N_LAYERS + HRM_N_LAYERS)
            + [True] * self.card.config.n_layers
        )
        max_len = max(self.hrm.config.max_len, self.card.config.max_len)

        cfg = HybridGroupedSmall2DConfig(
            vocab_size=vocab, d_model=d_model, n_heads=n_heads,
            n_layers=n_layers, d_ffn=d_ffn, max_len=max_len,
            use_hard_max=False,
            layer_modes=tuple(["single"] * n_layers),
            layer_hard_max=layer_hard_max,
            layer_linear_types=layer_types,
        )
        self.substrate = HybridGroupedSmall2DTransformer(cfg)
        with torch.no_grad():
            for p in self.substrate.parameters():
                p.zero_()

        # Populate Gemma stand-in (random, simulates trained LM weights)
        with torch.no_grad():
            self.substrate.tok.weight[
                :self.gemma.vocab_size,
                self.gemma.ch_off : self.gemma.ch_off + self.gemma.d_model,
            ].normal_(0, 0.02)
            self.substrate.pos.weight[
                :, self.gemma.ch_off : self.gemma.ch_off + self.gemma.d_model,
            ].normal_(0, 0.02)
            # Gemma layers: random weights at its rectangle.
            D_s = cfg.d_model
            for l in range(GEMMA_STANDIN_N_LAYERS):
                G_D = self.gemma.d_model
                G_SH = self.gemma.sh_off
                SH = G_D // 2
                qkv = self.substrate.W_qkv[l].weight
                # Gemma Q/K/V rectangles in GGML orientation (in, out)
                qkv[:G_D, :SH * 2].normal_(0, 0.02)                # Q
                qkv[:G_D, D_s:D_s + SH * 2].normal_(0, 0.02)        # K
                qkv[:G_D, 2 * D_s:2 * D_s + SH * 2].normal_(0, 0.02)  # V
                self.substrate.W_out[l].weight[
                    :SH * 2, :G_D].normal_(0, 0.02)
                self.substrate.ff_in[l].weight[
                    :G_D, :GEMMA_STANDIN_D_FFN].normal_(0, 0.02)
                F_s = cfg.d_ffn
                self.substrate.ff_in[l].weight[
                    :G_D,
                    F_s : F_s + GEMMA_STANDIN_D_FFN].normal_(0, 0.02)
                self.substrate.ff_out[l].weight[
                    :GEMMA_STANDIN_D_FFN, :G_D].normal_(0, 0.02)

        # Install HRM into its slot (hybrid uses GGML orientation)
        install_compiled_card_hybrid(
            self.substrate, self.hrm,
            ch_off=self.hrm_slot.ch_off,
            sh_off=self.hrm_slot.sh_off,
            ffn_off=self.hrm_slot.ffn_off,
            tok_off=self.hrm_slot.tok_off,
            layer_off=self.hrm_slot.layer_off,
        )
        install_compiled_card_hybrid(
            self.substrate, self.card,
            ch_off=self.card_slot.ch_off,
            sh_off=self.card_slot.sh_off,
            ffn_off=self.card_slot.ffn_off,
            tok_off=self.card_slot.tok_off,
            layer_off=self.card_slot.layer_off,
        )

        self.substrate.eval().to(self.device)

    # ---------- card path ----------

    def card_query(self, text: str) -> Optional[Union[int, bool]]:
        """Regex → card dispatch. None if out-of-scope for the card."""
        parsed = parse_prompt(text)
        if parsed is None:
            return None
        opcode, a, b = parsed
        tok_off = self.card_slot.tok_off
        x = torch.tensor(
            [[a + tok_off, b + tok_off,
              opcode + OPCODE_SHIFT + tok_off]],
            dtype=torch.long, device=self.device,
        )
        with torch.no_grad():
            logits = self.substrate(x)[0, 2]
        # Argmax restricted to card's vocab range
        card_logits = logits[tok_off : tok_off + CARD_VOCAB]
        slot = int(card_logits.argmax().item())
        return decode_output(opcode, slot)

    # ---------- HRM path ----------

    def _hrm_encode_prompt(self, text: str) -> torch.Tensor:
        """Encode a prompt as HRM tokens: <bos> + chars + <sep>.
        Maps HRM vocab ids to substrate's absolute vocab range."""
        bos = _CHAR_TO_ID["<bos>"]
        sep = _CHAR_TO_ID["<sep>"]
        ids = [bos]
        for c in text.lower():
            if c in _CHAR_TO_ID:
                ids.append(_CHAR_TO_ID[c])
        ids.append(sep)
        tok_off = self.hrm_slot.tok_off
        return torch.tensor(
            [[tok_off + t for t in ids]],
            dtype=torch.long, device=self.device,
        )

    def hrm_parse(self, text: str, max_new_tokens: int = 30) -> str:
        """Autoregressive HRM decode through the substrate. Returns the
        generated expression string (HRM's parse of the NL prompt)."""
        eos = _CHAR_TO_ID["<eos>"]
        tok_off = self.hrm_slot.tok_off
        x = self._hrm_encode_prompt(text)
        hrm_range = slice(tok_off, tok_off + HRM_VOCAB)

        generated_hrm_ids: list[int] = []
        max_len = self.substrate.config.max_len
        for _ in range(max_new_tokens):
            if x.shape[1] >= max_len:
                break
            with torch.no_grad():
                logits = self.substrate(x)[0, -1]
            hrm_logits = logits[hrm_range]
            next_hrm_id = int(hrm_logits.argmax().item())
            if next_hrm_id == eos:
                break
            generated_hrm_ids.append(next_hrm_id)
            # Append to input for next iter
            next_sub_tok = torch.tensor(
                [[tok_off + next_hrm_id]],
                dtype=torch.long, device=self.device,
            )
            x = torch.cat([x, next_sub_tok], dim=1)

        # Detokenize HRM ids to chars
        chars: list[str] = []
        for i in generated_hrm_ids:
            c = _ID_TO_CHAR.get(i, "")
            if c.startswith("<"):
                continue
            chars.append(c)
        return "".join(chars).strip()

    def hrm_then_card(self, text: str) -> Optional[Union[int, bool]]:
        """Brain+Cards: HRM parses NL → expression → card answers."""
        parsed_expr = self.hrm_parse(text)
        if not parsed_expr:
            return None
        # Strip trailing "=" if present
        expr = parsed_expr.rstrip("=").strip()
        return self.card_query(expr)

    # ---------- Gemma path ----------

    def gemma_residual_std(self, text: str) -> float:
        """Feed Gemma-stand-in tokens to the substrate. Returns the std
        of Gemma's residual channels after forward — confirms Gemma
        processes the input through its layers.

        (Full Gemma decode needs real weights, which this dev substrate
        doesn't have; see real_gemma_q6k_demo.py for the full version.)"""
        tok_off = self.gemma.tok_off
        # Hash text to a deterministic sequence of Gemma-stand-in tokens.
        ids = [tok_off + (ord(c) % self.gemma.vocab_size)
               for c in text[:self.substrate.config.max_len - 1]]
        if not ids:
            ids = [tok_off]
        x = torch.tensor([ids], dtype=torch.long, device=self.device)
        with torch.no_grad():
            from calm.llm_computer.grouped_attention import (
                grouped_attention_single_head_mode,
            )
            import torch.nn.functional as F
            B, S = x.shape
            cfg = self.substrate._hybrid_config
            pos_idx = torch.arange(S, device=self.device)
            res = self.substrate.tok(x) + self.substrate.pos(pos_idx)
            mask = torch.triu(
                torch.ones(S, S, dtype=torch.bool, device=self.device),
                diagonal=1,
            )
            for layer in range(cfg.n_layers):
                qkv = self.substrate.W_qkv[layer](res)
                qkv = qkv.reshape(B, S, 3, cfg.n_heads, cfg.d_head)
                q, k, v = qkv.permute(2, 0, 3, 1, 4)
                qh, kh, vh = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
                attn = grouped_attention_single_head_mode(
                    qh, kh, vh, mask=mask, scale=1.0,
                    hard_max=cfg.layer_hard_max[layer],
                )
                attn = attn.reshape(B, S, cfg.d_model)
                res = res + self.substrate.W_out[layer](attn)
                gate, val = self.substrate.ff_in[layer](res).chunk(2, dim=-1)
                res = res + self.substrate.ff_out[layer](F.relu(gate) * val)
        g_res = res[0, -1,
                    self.gemma.ch_off : self.gemma.ch_off + self.gemma.d_model]
        return float(g_res.std().item())


if __name__ == "__main__":
    import time
    t0 = time.time()
    print("[unified] building substrate...")
    comp = UnifiedSubstrateComputer()
    print(f"  substrate params: {comp.substrate.param_count():,}")
    print(f"  HRM val_acc: {comp.hrm_val_acc:.4f}")
    print(f"  Gemma slot:  ch [{comp.gemma.ch_off}, {comp.gemma.ch_off + comp.gemma.d_model}) "
          f"layers [{comp.gemma.layer_off}, {comp.gemma.layer_off + comp.gemma.n_layers})")
    print(f"  HRM slot:    ch [{comp.hrm_slot.ch_off}, {comp.hrm_slot.ch_off + comp.hrm_slot.d_model}) "
          f"layers [{comp.hrm_slot.layer_off}, {comp.hrm_slot.layer_off + comp.hrm_slot.n_layers})")
    print(f"  Card slot:   ch [{comp.card_slot.ch_off}, {comp.card_slot.ch_off + comp.card_slot.d_model}) "
          f"layers [{comp.card_slot.layer_off}, {comp.card_slot.layer_off + comp.card_slot.n_layers})")
    print(f"  build time: {time.time() - t0:.1f}s")

    print("\n[smoke] card path:")
    for p in ("3 + 5", "7 * 9", "5!", "is 7 prime?", "17 * 23"):
        r = comp.card_query(p)
        print(f"  {p!r:30} → {r!r}")

    # Use HRM's training templates verbatim — no punctuation, lowercase.
    print("\n[smoke] hrm_parse (matching training templates):")
    for p in ("what is 3 plus 5", "gcd of 6 and 9",
              "factorial of 4", "sum of 4 and 7",
              "product of 3 and 6", "is 7 prime",
              "what is 12 times 11"):
        r = comp.hrm_parse(p, max_new_tokens=20)
        print(f"  {p!r:30} → {r!r}")

    print("\n[smoke] hrm_then_card:")
    for p in ("what is 3 plus 5", "what is 7 times 9",
              "factorial of 4", "gcd of 6 and 9", "is 7 prime"):
        r = comp.hrm_then_card(p)
        print(f"  {p!r:30} → {r!r}")

    print("\n[smoke] gemma_residual_std:")
    for p in ("hello world", "3+5", "what is the capital of france?"):
        r = comp.gemma_residual_std(p)
        print(f"  {p!r:40} → std {r:.4f}")
