"""Tests for Direction 2 (computation traces), Direction 3 (mixed
geometry attention), and Direction 5 (recurrent substrate).

All extensions must be backward-compatible: calling forward without the
new kwargs / config defaults must produce bit-identical output to the
base Small2DTransformer.
"""

from __future__ import annotations

import torch

from calm.llm_computer.computation_trace import (
    ComputationTrace, TracedSmall2DTransformer, make_trace_collector,
)
from calm.llm_computer.mixed_geometry import (
    GEOMETRY_DISPATCH, MixedGeometryConfig, MixedGeometrySmall2DTransformer,
    euclidean_score, hyperbolic_score, lattice_score, spherical_score,
    toroidal_score,
)
from calm.llm_computer.model import Small2DConfig, Small2DTransformer
from calm.llm_computer.recurrent_substrate import (
    RecurrentConfig, RecurrentSmall2DTransformer,
)


def _tiny_cfg(**kw):
    defaults = dict(
        vocab_size=16, d_model=8, n_heads=4, n_layers=2, d_ffn=8,
        max_len=10, use_hard_max=False,
    )
    defaults.update(kw)
    return Small2DConfig(**defaults)


# ---- Direction 2: computation traces ----

def test_traced_no_trace_matches_parent():
    """TracedSmall2DTransformer with trace=None must equal parent forward."""
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    base = Small2DTransformer(cfg)
    traced = TracedSmall2DTransformer(cfg)
    traced.load_state_dict(base.state_dict())
    x = torch.randint(0, 16, (2, 5))
    with torch.no_grad():
        out_base = base(x)
        out_traced = traced(x)             # no trace argument
        out_traced_none = traced(x, trace=None)
    assert torch.equal(out_base, out_traced), "trace-not-passed should equal parent"
    assert torch.equal(out_base, out_traced_none), "trace=None should equal parent"


def test_traced_populates_trace():
    """When trace is passed, fields are populated and shapes are correct."""
    torch.manual_seed(0)
    cfg = _tiny_cfg(n_layers=3)
    model = TracedSmall2DTransformer(cfg)
    model.eval()
    x = torch.randint(0, 16, (2, 5))
    trace = make_trace_collector()
    with torch.no_grad():
        out = model(x, trace=trace)
    assert out.shape == (2, 5, cfg.vocab_size)
    assert trace.sequence_length == 5
    assert len(trace.layers) == 3
    for i, lt in enumerate(trace.layers):
        assert lt.layer_idx == i
        assert lt.attention_weights.shape == (2, cfg.n_heads, 5, 5)
        assert lt.attention_argmax.shape == (2, cfg.n_heads, 5)
        assert lt.ffn_active_count.shape == (2, 5)
        assert lt.ffn_max_activation.shape == (2, 5)
        assert lt.geometry == "euclidean"


def test_trace_attention_to_helper():
    """attention_to(layer, head, query) returns a valid past position."""
    torch.manual_seed(0)
    model = TracedSmall2DTransformer(_tiny_cfg())
    model.eval()
    x = torch.randint(0, 16, (1, 4))
    trace = make_trace_collector()
    with torch.no_grad():
        model(x, trace=trace)
    sel = trace.attention_to(layer=0, head=0, query_pos=2)
    assert 0 <= sel <= 2, f"causal mask violated: pos 2 attends to {sel}"


# ---- Direction 3: mixed geometry attention ----

def _qk(B=2, H=4, S=5):
    torch.manual_seed(0)
    return torch.randn(B, H, S, 2), torch.randn(B, H, S, 2)


def test_geometry_dispatch_complete():
    """All five geometries are registered."""
    expected = {"euclidean", "hyperbolic", "spherical", "toroidal", "lattice"}
    assert set(GEOMETRY_DISPATCH) == expected


def test_geometry_score_shapes():
    """Every geometry returns (B, H, S, S) given (B, H, S, 2) inputs."""
    q, k = _qk()
    expected_shape = (q.size(0), q.size(1), q.size(2), k.size(2))
    for name, fn in GEOMETRY_DISPATCH.items():
        out = fn(q, k)
        assert out.shape == expected_shape, f"{name} returned {out.shape}"
        assert torch.isfinite(out).all(), f"{name} produced non-finite values"


def test_spherical_magnitude_invariant():
    """Spherical score should be unchanged when Q is scaled."""
    q, k = _qk()
    s1 = spherical_score(q, k)
    s2 = spherical_score(q * 5.0, k)
    assert torch.allclose(s1, s2, atol=1e-5), \
        "spherical attention should ignore Q magnitude"


def test_hyperbolic_diagonal_strongest():
    """When Q and K are identical, hyperbolic distance is 0 — strongest score."""
    q, _ = _qk()
    scores = hyperbolic_score(q, q)
    # Diagonal should be the maximum row-wise (distance 0 = score 0).
    diag = scores.diagonal(dim1=-2, dim2=-1)             # (B, H, S)
    row_max = scores.max(dim=-1).values                   # (B, H, S)
    assert torch.allclose(diag, row_max, atol=1e-4), \
        "hyperbolic q=k should produce maximum self-attention"


def test_lattice_snaps_keys():
    """Lattice score should equal Euclidean on integer-rounded keys."""
    q, k = _qk()
    k_round = k.round()
    expected = torch.einsum("bhid,bhjd->bhij", q, k_round)
    actual = lattice_score(q, k)
    assert torch.allclose(expected, actual)


def test_mixed_geometry_uniform_matches_parent():
    """layer_geometries=None falls back to parent behavior bitwise."""
    torch.manual_seed(0)
    base_cfg = _tiny_cfg()
    mg_cfg = MixedGeometryConfig(
        vocab_size=16, d_model=8, n_heads=4, n_layers=2, d_ffn=8,
        max_len=10, use_hard_max=False,
        layer_geometries=None,
    )
    base = Small2DTransformer(base_cfg)
    mg = MixedGeometrySmall2DTransformer(mg_cfg)
    mg.load_state_dict(base.state_dict())
    x = torch.randint(0, 16, (2, 5))
    with torch.no_grad():
        assert torch.equal(base(x), mg(x))


def test_mixed_geometry_custom_runs():
    """Per-layer geometry selection produces finite output."""
    torch.manual_seed(0)
    cfg = MixedGeometryConfig(
        vocab_size=16, d_model=8, n_heads=4, n_layers=3, d_ffn=8,
        max_len=10, use_hard_max=False,
        layer_geometries=["hyperbolic", "spherical", "euclidean"],
    )
    model = MixedGeometrySmall2DTransformer(cfg)
    model.eval()
    x = torch.randint(0, 16, (2, 5))
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 5, 16)
    assert torch.isfinite(out).all()


def test_mixed_geometry_validates_length():
    """Wrong-length layer_geometries should assert."""
    cfg = MixedGeometryConfig(
        vocab_size=16, d_model=8, n_heads=4, n_layers=3, d_ffn=8,
        max_len=10, use_hard_max=False,
        layer_geometries=["hyperbolic", "spherical"],   # only 2 vs n_layers=3
    )
    raised = False
    try:
        MixedGeometrySmall2DTransformer(cfg)
    except AssertionError:
        raised = True
    assert raised, "should reject layer_geometries with wrong length"


# ---- Direction 5: recurrent substrate ----

def test_recurrent_n1_matches_parent():
    """default_iterations=1 with n_iterations=None must equal parent."""
    torch.manual_seed(0)
    cfg_base = _tiny_cfg()
    cfg_rec = RecurrentConfig(
        vocab_size=16, d_model=8, n_heads=4, n_layers=2, d_ffn=8,
        max_len=10, use_hard_max=False,
        default_iterations=1, max_iterations=8,
    )
    base = Small2DTransformer(cfg_base)
    rec = RecurrentSmall2DTransformer(cfg_rec)
    rec.load_state_dict(base.state_dict())
    x = torch.randint(0, 16, (2, 5))
    with torch.no_grad():
        assert torch.equal(base(x), rec(x))
        assert torch.equal(base(x), rec(x, n_iterations=1))


def test_recurrent_more_iterations_changes_output():
    """n_iterations > 1 must produce different output than n=1."""
    torch.manual_seed(0)
    cfg = RecurrentConfig(
        vocab_size=16, d_model=8, n_heads=4, n_layers=2, d_ffn=8,
        max_len=10, use_hard_max=False, default_iterations=1, max_iterations=8,
    )
    model = RecurrentSmall2DTransformer(cfg)
    model.eval()
    x = torch.randint(0, 16, (2, 5))
    with torch.no_grad():
        o1 = model(x, n_iterations=1)
        o3 = model(x, n_iterations=3)
    assert not torch.allclose(o1, o3, atol=1e-5), \
        "more iterations should evolve the residual stream"


def test_recurrent_max_iterations_clamps():
    """Asking for more than max_iterations should clamp to the cap."""
    torch.manual_seed(0)
    cfg = RecurrentConfig(
        vocab_size=16, d_model=8, n_heads=4, n_layers=2, d_ffn=8,
        max_len=10, use_hard_max=False, default_iterations=1, max_iterations=4,
    )
    model = RecurrentSmall2DTransformer(cfg)
    model.eval()
    x = torch.randint(0, 16, (2, 5))
    with torch.no_grad():
        o4 = model(x, n_iterations=4)
        o100 = model(x, n_iterations=100)   # should clamp to 4 → equal o4
    assert torch.equal(o4, o100), "n_iterations=100 should clamp to max_iterations=4"


if __name__ == "__main__":
    # Direction 2
    test_traced_no_trace_matches_parent()
    print("[ok] D2: traced subclass without trace = parent")
    test_traced_populates_trace()
    print("[ok] D2: trace fields populated correctly")
    test_trace_attention_to_helper()
    print("[ok] D2: causal attention_to helper")
    # Direction 3
    test_geometry_dispatch_complete()
    print("[ok] D3: 5 geometries registered")
    test_geometry_score_shapes()
    print("[ok] D3: all geometries return correct shapes")
    test_spherical_magnitude_invariant()
    print("[ok] D3: spherical is magnitude-invariant")
    test_hyperbolic_diagonal_strongest()
    print("[ok] D3: hyperbolic q=k gives max self-attention")
    test_lattice_snaps_keys()
    print("[ok] D3: lattice = Euclidean on rounded keys")
    test_mixed_geometry_uniform_matches_parent()
    print("[ok] D3: layer_geometries=None matches parent bitwise")
    test_mixed_geometry_custom_runs()
    print("[ok] D3: mixed geometry forward runs cleanly")
    test_mixed_geometry_validates_length()
    print("[ok] D3: rejects mismatched layer_geometries length")
    # Direction 5
    test_recurrent_n1_matches_parent()
    print("[ok] D5: n_iterations=1 matches parent bitwise")
    test_recurrent_more_iterations_changes_output()
    print("[ok] D5: more iterations evolve residual stream")
    test_recurrent_max_iterations_clamps()
    print("[ok] D5: max_iterations clamps runaway requests")
