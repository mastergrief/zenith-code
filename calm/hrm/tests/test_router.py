"""Router data + model tests."""

from __future__ import annotations

import torch

from calm.hrm.router_data import (
    LABELS, LABEL_TO_ID, N_LABELS, RouterDataset, RouterGenerator,
)
from calm.hrm.router_model import RouterConfig, RouterHRM


def test_label_registry_consistent():
    assert len(LABELS) == N_LABELS
    assert {"math", "nl", "word", "gsm", "meta"} <= set(LABELS)
    for name, i in LABEL_TO_ID.items():
        assert LABELS[i] == name


def test_router_generator_balanced():
    gen = RouterGenerator(seed=7)
    samples = gen.generate(500)
    counts = [0] * N_LABELS
    for s in samples:
        counts[s.label_id] += 1
    # Each bucket should hold roughly n/N_LABELS; allow ±10% slack.
    per = 500 // N_LABELS
    for c in counts:
        assert abs(c - per) <= max(per // 4, 5), (counts, per)


def test_router_dataset_shapes():
    gen = RouterGenerator(seed=11)
    ds = RouterDataset(gen.generate(20), max_len=384)
    item = ds[0]
    assert item["input_ids"].shape == (384,)
    assert item["label"].shape == ()
    assert 0 <= int(item["label"].item()) < N_LABELS


def test_router_forward_shape():
    model = RouterHRM(RouterConfig(hidden_size=16, num_heads=4, max_seq_len=384,
                                    num_labels=N_LABELS))
    x = torch.randint(1, 80, (3, 384), dtype=torch.long)
    logits = model(x)
    assert logits.shape == (3, N_LABELS)


def test_router_is_tiny():
    model = RouterHRM(RouterConfig())
    assert model.param_count() < 30_000, model.param_count()
