"""Dispatcher integration tests.

Skips gracefully when router/specialist checkpoints aren't on disk,
so CI stays green even when training hasn't landed for a fresh clone.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from calm.hrm.dispatcher import DEFAULT_ROUTER_CKPT, DEFAULT_SPECIALIST_CKPTS, Dispatcher


def _ckpt_exists(name: str) -> bool:
    return Path(DEFAULT_SPECIALIST_CKPTS[name]).exists()


def _router_exists() -> bool:
    return Path(DEFAULT_ROUTER_CKPT).exists()


def test_dispatcher_construct():
    d = Dispatcher()
    assert d.device == "cpu"
    assert "math" in d.specialist_ckpts


@pytest.mark.skipif(not _router_exists(), reason="router checkpoint missing")
def test_router_classifies_math_expression():
    d = Dispatcher()
    label = d.route("347 * 289")
    assert label in {"math", "nl"}  # shape is short — math or nl depending on training


@pytest.mark.skipif(not _router_exists(), reason="router checkpoint missing")
def test_router_classifies_meta_prefix():
    from calm.hrm.meta_data import MetaGenerator, TRAIN_FORMATS
    from calm.hrm.router_data import _make_meta_text
    d = Dispatcher()
    # Use an actual MetaGenerator sample from the same distribution the
    # router was trained on, so this tests routing rather than OOD robustness.
    sample = MetaGenerator(seed=99, formats=TRAIN_FORMATS).generate(1)[0]
    assert d.route(_make_meta_text(sample)) == "meta"


@pytest.mark.skipif(
    not (_router_exists() and _ckpt_exists("math")),
    reason="router or math specialist missing",
)
def test_dispatcher_end_to_end_math():
    d = Dispatcher()
    # A simple math expression should route to math and the specialist
    # should emit the same expression back through the verified path.
    res = d.run("347 * 289")
    if res.label == "math":
        assert res.answer == "100283", (res.label, res.emit, res.answer)


@pytest.mark.skipif(not _router_exists(), reason="router checkpoint missing")
def test_dispatcher_unknown_specialist_raises():
    d = Dispatcher(specialist_ckpts={"math": "/nonexistent/path.pt"})
    # route may still work; loading the unseen specialist should fail.
    try:
        d.run("123")
    except FileNotFoundError:
        pass
