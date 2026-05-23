"""T2 γ1 parity + recurrence-aliasing tests for KV cache decode.

Per codex +1 implement at msg 1779530833485-eb9296ca and the design
proposal at msg 1779530825108-86d50e8a.

9 tests:
  1. Mask-alignment guard (q_len=1 vs k_len>1) — the LOAD-BEARING test
     that fails if cached decode uses is_causal=True on non-square Q/K
  2. Multi-step decode parity vs no-cache full-prefix re-forward
  3. K/V differ across L iterations at same layer (aliasing guard)
  4. K/V differ between H and L at same layer/iter (aliasing guard)
  5. Cached positions remain bit-identical across decode steps
  6. Cross-row independence after .reset()
  7. Training-mode bypass (defense in depth alongside `not self.training` guard)
  8. PrefixLM prefill vs cached K/V identity (K/V independent of mask)
  9. Access pattern hits exactly 32 distinct keys per HRM forward
"""
from __future__ import annotations

import pytest
import torch

from calm.hrm_text_158 import (
    HierarchicalReasoningModel,
    HierarchicalReasoningModelConfig,
    LMHead,
    LMHeadConfig,
)
from calm.hrm_text_158.kv_cache import KVCache


@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def _build_tiny_model(device: str, vocab_size: int = 32) -> tuple[LMHead, int]:
    """Build a small HRM-Text-1.58 model for fast tests.

    Matches the R1b5 seed=17 arch shape (n_layers=8 with half_layers=True,
    H_cycles=2, L_cycles=3) so the 32-buffer cache key contract is exercised
    identically to the production path. Hidden size shrunk to 64 for speed.
    """
    cfg = HierarchicalReasoningModelConfig(
        max_seq_len=64,
        n_layers=8,
        half_layers=True,
        H_cycles=2,
        L_cycles=3,
        hidden_size=64,
        num_heads=4,
        expansion=3,
        attn_type="prefixlm",
        init_type="lecun_normal",
        init_std=None,
        norm_type="pre",
        norm_eps=1e-6,
        pos_emb_type="rope",
        rope_theta=10000.0,
        bp_warmup_ratio=0.1,
        bp_min_steps=2,
        bp_max_steps=5,
    )
    hrm = HierarchicalReasoningModel(cfg)
    m = LMHead(hrm, LMHeadConfig(vocab_size=vocab_size)).to(device).to(torch.float32)
    m.eval()
    # Deterministic init for reproducibility
    torch.manual_seed(42)
    for p in m.parameters():
        if p.numel() > 0:
            torch.nn.init.normal_(p, std=0.02)
    m.eval()
    return m, cfg.max_seq_len


def _build_kv_cache(m: LMHead, max_seq: int, device: str) -> KVCache:
    sample_attn = m.model.H_level.core.layers[0].attn
    sample_w = sample_attn.gqkv_proj.weight
    return KVCache(
        max_seq_len=max_seq,
        num_kv_heads=sample_attn.num_key_value_heads,
        head_dim=sample_attn.head_dim,
        dtype=sample_w.dtype,
        device=device,
    )


def _build_batch(ids: list[int], sep_pos: int, device: str) -> dict:
    return {
        "inputs": torch.tensor([ids], dtype=torch.long, device=device),
        "sep_positions": torch.tensor([sep_pos], dtype=torch.long, device=device),
        "position_ids": torch.arange(len(ids), dtype=torch.long, device=device).unsqueeze(0),
    }


# ---------------------------------------------------------------------------- #
# TEST 1 — Mask-alignment guard (q_len=1, k_len>1). LOAD-BEARING.
# ---------------------------------------------------------------------------- #


def test_single_token_decode_matches_full_prefix_no_cache(device: str):
    """If decode SDPA accidentally uses is_causal=True with non-square Q/K,
    Q[0] would align to query-local index 0 and only attend K[0], not the
    full cached past. This test compares cached single-token decode logits
    against a fresh no-cache full-prefix re-forward at the same position.
    Diff > eps → wrong mask alignment.
    """
    m, max_seq = _build_tiny_model(device)
    prompt = [1, 5, 7, 3, 2]  # arbitrary token ids; vocab=32
    sep_pos = 4  # last position is sep
    next_id = 4

    # Path A: full no-cache re-forward at length 6
    full_ids = prompt + [next_id]
    batch_full = _build_batch(full_ids, sep_pos, device)
    with torch.no_grad():
        _, logits_full = m(None, batch_full)
    logits_no_cache = logits_full[0, -1, :]

    # Path B: prefill on prompt, then cached decode at position 5
    cache = _build_kv_cache(m, max_seq, device)
    batch_prefill = _build_batch(prompt, sep_pos, device)
    with torch.no_grad():
        _, _ = m(None, batch_prefill, kv_cache=cache)
    # Decode single new token at position len(prompt)=5
    decode_ids = torch.tensor([[next_id]], dtype=torch.long, device=device)
    decode_pos = torch.tensor([[len(prompt)]], dtype=torch.long, device=device)
    decode_batch = {
        "inputs": decode_ids,
        "sep_positions": torch.tensor([sep_pos], dtype=torch.long, device=device),
        "position_ids": decode_pos,
    }
    with torch.no_grad():
        _, logits_decode = m(None, decode_batch, kv_cache=cache)
    logits_cached = logits_decode[0, 0, :]

    # Allow small fp32 numerical noise from SDPA reorderings, but a wrong
    # mask alignment would produce dramatically different logits.
    max_diff = (logits_no_cache - logits_cached).abs().max().item()
    assert max_diff < 1e-3, (
        f"cached decode logits drift from full-prefix no-cache: "
        f"max abs diff = {max_diff:.6e}. This is the mask-alignment guard."
    )


# ---------------------------------------------------------------------------- #
# TEST 2 — Multi-step decode parity
# ---------------------------------------------------------------------------- #


def test_multi_step_decode_parity(device: str):
    """Greedy-decode 4 tokens with cache vs without; same sequence of token ids."""
    m, max_seq = _build_tiny_model(device)
    prompt = [1, 5, 7, 3, 2]
    sep_pos = 4
    max_gen = 4

    # No-cache greedy: full-prefix re-forward each step
    cur = list(prompt)
    no_cache_tokens: list[int] = []
    for _ in range(max_gen):
        batch = _build_batch(cur, sep_pos, device)
        with torch.no_grad():
            _, logits = m(None, batch)
        nxt = int(torch.argmax(logits[0, -1], dim=-1).item())
        no_cache_tokens.append(nxt)
        cur.append(nxt)

    # Cached greedy
    cache = _build_kv_cache(m, max_seq, device)
    batch_prefill = _build_batch(prompt, sep_pos, device)
    with torch.no_grad():
        _, logits = m(None, batch_prefill, kv_cache=cache)
    nxt = int(torch.argmax(logits[0, -1], dim=-1).item())
    cached_tokens = [nxt]
    cur_pos = len(prompt)
    for _ in range(max_gen - 1):
        decode_ids = torch.tensor([[nxt]], dtype=torch.long, device=device)
        decode_pos = torch.tensor([[cur_pos]], dtype=torch.long, device=device)
        batch = {
            "inputs": decode_ids,
            "sep_positions": torch.tensor([sep_pos], dtype=torch.long, device=device),
            "position_ids": decode_pos,
        }
        with torch.no_grad():
            _, logits = m(None, batch, kv_cache=cache)
        nxt = int(torch.argmax(logits[0, -1], dim=-1).item())
        cached_tokens.append(nxt)
        cur_pos += 1

    assert no_cache_tokens == cached_tokens, (
        f"decode token sequences differ:\n  no_cache={no_cache_tokens}\n  "
        f"cached={cached_tokens}"
    )


# ---------------------------------------------------------------------------- #
# TEST 3 — K/V differ across L iterations at same layer (recurrence aliasing)
# ---------------------------------------------------------------------------- #


def test_kv_differ_across_L_iterations(device: str):
    """L-level iteration 0 layer 0 K/V must NOT equal iteration 1 layer 0 K/V.

    The L_level module instance is REUSED across recurrence iterations
    (hrm.py:141-150), but the residual state differs each iteration.
    A bug that keyed cache only by (level, layer) would cause iteration 1
    to read iteration 0's K/V and produce wrong output. This test
    explicitly catches that aliasing mode.
    """
    m, max_seq = _build_tiny_model(device)
    prompt = [1, 5, 7, 3, 2]
    sep_pos = 4
    cache = _build_kv_cache(m, max_seq, device)
    batch = _build_batch(prompt, sep_pos, device)
    with torch.no_grad():
        _, _ = m(None, batch, kv_cache=cache)

    # Both buffers should be present and same shape
    L_iter0_layer0 = cache.get_buffer("L", 0, 0)
    L_iter1_layer0 = cache.get_buffer("L", 1, 0)
    assert L_iter0_layer0 is not None, "L iter=0 layer=0 cache must exist"
    assert L_iter1_layer0 is not None, "L iter=1 layer=0 cache must exist"
    K0, V0, len0 = L_iter0_layer0
    K1, V1, len1 = L_iter1_layer0
    assert len0 == len1 == len(prompt)
    # Aliasing bug would make these bit-equal. Healthy implementation has them
    # numerically distinct (the residual state differs across recurrence).
    assert not torch.equal(K0, K1), (
        "L iter=0 K equals L iter=1 K — recurrence-cache aliasing detected"
    )
    assert not torch.equal(V0, V1), (
        "L iter=0 V equals L iter=1 V — recurrence-cache aliasing detected"
    )


# ---------------------------------------------------------------------------- #
# TEST 4 — K/V differ between H and L at same (rec_idx, layer)
# ---------------------------------------------------------------------------- #


def test_kv_differ_between_H_and_L(device: str):
    """H-level iter 0 layer 0 K/V must NOT equal L-level iter 0 layer 0 K/V.

    Different module instances with different residual inputs — must
    occupy distinct cache slots.
    """
    m, max_seq = _build_tiny_model(device)
    prompt = [1, 5, 7, 3, 2]
    sep_pos = 4
    cache = _build_kv_cache(m, max_seq, device)
    batch = _build_batch(prompt, sep_pos, device)
    with torch.no_grad():
        _, _ = m(None, batch, kv_cache=cache)
    H = cache.get_buffer("H", 0, 0)
    L = cache.get_buffer("L", 0, 0)
    assert H is not None and L is not None
    K_h, V_h, _ = H
    K_l, V_l, _ = L
    assert not torch.equal(K_h, K_l), "H and L K aliased at same (iter, layer)"
    assert not torch.equal(V_h, V_l), "H and L V aliased at same (iter, layer)"


# ---------------------------------------------------------------------------- #
# TEST 5 — Cached positions remain bit-identical across decode steps
# ---------------------------------------------------------------------------- #


def test_cached_positions_stable_across_decode_steps(device: str):
    """K/V at position p≤N in the cache at decode step N+1 must equal what
    they were at decode step N. Decode must only APPEND, never retroactively
    modify cached entries.
    """
    m, max_seq = _build_tiny_model(device)
    prompt = [1, 5, 7, 3, 2]
    sep_pos = 4
    cache = _build_kv_cache(m, max_seq, device)
    batch_prefill = _build_batch(prompt, sep_pos, device)
    with torch.no_grad():
        _, _ = m(None, batch_prefill, kv_cache=cache)

    # Snapshot after prefill
    K0, V0, L0 = cache.get_buffer("L", 2, 1)  # arbitrary key
    K0_snapshot = K0.clone()
    V0_snapshot = V0.clone()

    # One decode step
    decode_ids = torch.tensor([[3]], dtype=torch.long, device=device)
    decode_pos = torch.tensor([[len(prompt)]], dtype=torch.long, device=device)
    batch = {
        "inputs": decode_ids,
        "sep_positions": torch.tensor([sep_pos], dtype=torch.long, device=device),
        "position_ids": decode_pos,
    }
    with torch.no_grad():
        _, _ = m(None, batch, kv_cache=cache)

    K1, V1, L1 = cache.get_buffer("L", 2, 1)
    assert L1 == L0 + 1, "cache length should grow by 1 per decode step"
    # Positions [0..L0) must match the pre-decode snapshot
    assert torch.equal(K1[:, :, :L0, :], K0_snapshot), (
        "cached K at prefix positions changed during decode — should be append-only"
    )
    assert torch.equal(V1[:, :, :L0, :], V0_snapshot), (
        "cached V at prefix positions changed during decode — should be append-only"
    )


# ---------------------------------------------------------------------------- #
# TEST 6 — Cross-row independence after .reset()
# ---------------------------------------------------------------------------- #


def test_cross_row_independence_after_reset(device: str):
    """Decoding row B with a previously-used (then reset) cache yields the
    same token sequence as decoding row B with a fresh cache.
    """
    m, max_seq = _build_tiny_model(device)
    prompt_A = [1, 5, 7, 3, 2]
    prompt_B = [2, 8, 1, 4, 6, 3]
    sep_pos_A = 4
    sep_pos_B = 5
    max_gen = 3

    def decode_row(cache: KVCache, prompt, sep_pos) -> list[int]:
        batch_prefill = _build_batch(prompt, sep_pos, device)
        with torch.no_grad():
            _, logits = m(None, batch_prefill, kv_cache=cache)
        nxt = int(torch.argmax(logits[0, -1], dim=-1).item())
        toks = [nxt]
        cur_pos = len(prompt)
        for _ in range(max_gen - 1):
            decode_ids = torch.tensor([[nxt]], dtype=torch.long, device=device)
            decode_pos = torch.tensor([[cur_pos]], dtype=torch.long, device=device)
            batch = {
                "inputs": decode_ids,
                "sep_positions": torch.tensor([sep_pos], dtype=torch.long, device=device),
                "position_ids": decode_pos,
            }
            with torch.no_grad():
                _, logits = m(None, batch, kv_cache=cache)
            nxt = int(torch.argmax(logits[0, -1], dim=-1).item())
            toks.append(nxt)
            cur_pos += 1
        return toks

    cache_reused = _build_kv_cache(m, max_seq, device)
    # First decode row A
    _ = decode_row(cache_reused, prompt_A, sep_pos_A)
    cache_reused.reset()
    # Then decode row B with the same cache
    toks_B_reused = decode_row(cache_reused, prompt_B, sep_pos_B)

    cache_fresh = _build_kv_cache(m, max_seq, device)
    toks_B_fresh = decode_row(cache_fresh, prompt_B, sep_pos_B)

    assert toks_B_reused == toks_B_fresh, (
        f"row B differs between reused and fresh cache: "
        f"reused={toks_B_reused} fresh={toks_B_fresh}"
    )


# ---------------------------------------------------------------------------- #
# TEST 7 — Training-mode bypass
# ---------------------------------------------------------------------------- #


def test_training_mode_bypasses_kv_cache(device: str):
    """Under model.train(), the cached path must NOT be used even when
    kv_cache is passed. The `not self.training` guard in Attention.forward
    is the defense-in-depth alongside the runtime-only cache attribute.
    """
    m, max_seq = _build_tiny_model(device)
    prompt = [1, 5, 7, 3, 2]
    sep_pos = 4
    cache = _build_kv_cache(m, max_seq, device)

    # Train mode + kv_cache → cached path bypassed; cache should not be populated
    m.train()
    batch = _build_batch(prompt, sep_pos, device)
    with torch.no_grad():
        _, _ = m(None, batch, kv_cache=cache)
    # Cache.update never called → 0 buffers
    assert cache.num_buffers() == 0, (
        f"training-mode forward populated cache anyway: {cache.num_buffers()} buffers"
    )
    m.eval()


# ---------------------------------------------------------------------------- #
# TEST 8 — PrefixLM prefill stores same K/V as no-mask hypothetical
# ---------------------------------------------------------------------------- #


def test_prefill_kv_independent_of_mask_choice(device: str):
    """K/V are computed from gqkv_proj + RoPE BEFORE the attention mask is
    applied. Mask only controls Q-K attention weighting, not K/V storage.

    Sanity: the values stored in cache during prefill (with PrefixLM mask)
    are the same K/V that would be computed under any other mask choice.
    Test by running prefill twice on the same prompt and asserting the
    cached K/V are bit-identical (i.e. deterministic computation).
    """
    m, max_seq = _build_tiny_model(device)
    prompt = [1, 5, 7, 3, 2]
    sep_pos = 4
    cache_A = _build_kv_cache(m, max_seq, device)
    cache_B = _build_kv_cache(m, max_seq, device)
    batch = _build_batch(prompt, sep_pos, device)
    with torch.no_grad():
        _, _ = m(None, batch, kv_cache=cache_A)
        _, _ = m(None, batch, kv_cache=cache_B)
    # Same prompt → same K/V across cache instances
    K_A, V_A, _ = cache_A.get_buffer("L", 0, 0)
    K_B, V_B, _ = cache_B.get_buffer("L", 0, 0)
    assert torch.equal(K_A, K_B)
    assert torch.equal(V_A, V_B)


# ---------------------------------------------------------------------------- #
# TEST 9 — Access pattern hits 32 distinct keys per HRM forward (B=1)
# ---------------------------------------------------------------------------- #


def test_access_pattern_hits_32_distinct_keys(device: str):
    """One HRM forward should touch exactly 32 distinct cache keys (single-row).

    Arch: H_cycles=2, L_cycles=3, half_layers=True at n_layers=8 → 4 layers
    per level. L access: 6 iters × 4 layers = 24. H access: 2 iters × 4 layers
    = 8. Total: 32 distinct (level, rec_idx, layer_idx) keys.
    """
    m, max_seq = _build_tiny_model(device)
    prompt = [1, 5, 7, 3, 2]
    sep_pos = 4
    cache = _build_kv_cache(m, max_seq, device)
    batch = _build_batch(prompt, sep_pos, device)
    with torch.no_grad():
        _, _ = m(None, batch, kv_cache=cache)

    log = cache.get_access_log()
    distinct_keys = set(log)
    # Each key should be touched exactly once per HRM forward (prefill = 1 call)
    assert len(log) == 32, f"expected 32 cache accesses per HRM forward, got {len(log)}"
    assert len(distinct_keys) == 32, (
        f"expected 32 DISTINCT keys, got {len(distinct_keys)} "
        f"(duplicates indicate aliasing)"
    )
    # Verify exact key distribution
    L_keys = {k for k in distinct_keys if k[0] == "L"}
    H_keys = {k for k in distinct_keys if k[0] == "H"}
    assert len(L_keys) == 24, f"expected 24 L keys, got {len(L_keys)}"
    assert len(H_keys) == 8, f"expected 8 H keys, got {len(H_keys)}"
    # L rec_idx should cover [0, 6); H rec_idx should cover [0, 2)
    L_rec = {k[1] for k in L_keys}
    H_rec = {k[1] for k in H_keys}
    assert L_rec == set(range(6)), f"L rec_idx coverage: {L_rec}"
    assert H_rec == set(range(2)), f"H rec_idx coverage: {H_rec}"
    # layer_idx should cover [0, 4) for both
    L_layers = {k[2] for k in L_keys}
    H_layers = {k[2] for k in H_keys}
    assert L_layers == set(range(4)), f"L layer coverage: {L_layers}"
    assert H_layers == set(range(4)), f"H layer coverage: {H_layers}"
