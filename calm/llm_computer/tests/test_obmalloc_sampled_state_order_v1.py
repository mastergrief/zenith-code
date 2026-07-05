from __future__ import annotations

import pytest

from calm.hrm_text_158.native_full_stack.host_tracemalloc_probe import (
    OBMALLOC_EXPANDED_SAMPLED_STATE_ORDER_ENV,
    OBMALLOC_EXPANDED_SAMPLED_STATES_ENV,
    build_c4_apply_visit_sequence,
    build_order_provenance_fields,
    resolve_obmalloc_expanded_sampled_state_order,
    resolve_obmalloc_expanded_sampled_states,
)
from scripts.hrm_text_158_slice5_v6i_oom_profile_attribution import (
    build_ca_band_counter_confirmation_receipt,
)


def _dense_env(monkeypatch: pytest.MonkeyPatch, *, order: str | None = None) -> None:
    monkeypatch.setenv(
        OBMALLOC_EXPANDED_SAMPLED_STATES_ENV,
        ",".join(str(i) for i in range(10)),
    )
    if order is None:
        monkeypatch.delenv(OBMALLOC_EXPANDED_SAMPLED_STATE_ORDER_ENV, raising=False)
    else:
        monkeypatch.setenv(OBMALLOC_EXPANDED_SAMPLED_STATE_ORDER_ENV, order)


def test_resolve_order_unset_returns_none_and_visit_is_enumerate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(OBMALLOC_EXPANDED_SAMPLED_STATE_ORDER_ENV, raising=False)
    monkeypatch.delenv(OBMALLOC_EXPANDED_SAMPLED_STATES_ENV, raising=False)
    sampled = resolve_obmalloc_expanded_sampled_states(32)
    assert resolve_obmalloc_expanded_sampled_state_order(32, sampled) is None
    assert build_c4_apply_visit_sequence(32, sampled, None) == tuple(range(32))


def test_identity_order_dense_09_preserves_enumerate_semantics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _dense_env(monkeypatch, order=",".join(str(i) for i in range(10)))
    sampled = resolve_obmalloc_expanded_sampled_states(32)
    order = resolve_obmalloc_expanded_sampled_state_order(32, sampled)
    assert order == tuple(range(10))
    visit = build_c4_apply_visit_sequence(32, sampled, order)
    assert visit == tuple(range(32))
    prov = build_order_provenance_fields(32, sampled, order)
    assert prov["order_control_active"] is True
    assert prov["order_perturbation_kind"] == "sampled_block_order_perturbation"
    assert prov["effective_visit_order"] == list(range(32))
    assert prov["order_rank_by_semantic_state"] == {str(i): i for i in range(32)}


def test_invalid_order_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    _dense_env(monkeypatch)
    sampled = resolve_obmalloc_expanded_sampled_states(32)

    monkeypatch.setenv(OBMALLOC_EXPANDED_SAMPLED_STATE_ORDER_ENV, "")
    assert resolve_obmalloc_expanded_sampled_state_order(32, sampled) is None

    monkeypatch.setenv(OBMALLOC_EXPANDED_SAMPLED_STATE_ORDER_ENV, " ,")
    with pytest.raises(ValueError):
        resolve_obmalloc_expanded_sampled_state_order(32, sampled)
    with pytest.raises(ValueError):
        resolve_obmalloc_expanded_sampled_state_order(32, sampled)

    monkeypatch.setenv(OBMALLOC_EXPANDED_SAMPLED_STATE_ORDER_ENV, "0,0,1,2,3,4,5,6,7,8,9")
    with pytest.raises(ValueError):
        resolve_obmalloc_expanded_sampled_state_order(32, sampled)

    monkeypatch.setenv(OBMALLOC_EXPANDED_SAMPLED_STATE_ORDER_ENV, "0,1,2,3,4,5,6,7,8,9,10")
    with pytest.raises(ValueError):
        resolve_obmalloc_expanded_sampled_state_order(32, sampled)

    monkeypatch.setenv(OBMALLOC_EXPANDED_SAMPLED_STATE_ORDER_ENV, "x")
    with pytest.raises(ValueError):
        resolve_obmalloc_expanded_sampled_state_order(32, sampled)


def test_reversed_order_preserves_semantic_ids_and_numeric_tail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _dense_env(monkeypatch, order=",".join(str(i) for i in range(9, -1, -1)))
    sampled = resolve_obmalloc_expanded_sampled_states(32)
    order = resolve_obmalloc_expanded_sampled_state_order(32, sampled)
    assert order == tuple(range(9, -1, -1))
    visit = build_c4_apply_visit_sequence(32, sampled, order)
    assert visit[:10] == tuple(range(9, -1, -1))
    assert visit[10:] == tuple(range(10, 32))
    prov = build_order_provenance_fields(32, sampled, order)
    assert prov["order_rank_by_semantic_state"]["0"] == 9
    assert prov["order_rank_by_semantic_state"]["9"] == 0
    assert prov["effective_visit_order"][0] == 9
    assert prov["effective_visit_order"][9] == 0


def test_slice5_receipt_includes_order_provenance(monkeypatch: pytest.MonkeyPatch) -> None:
    _dense_env(monkeypatch, order=",".join(str(i) for i in range(9, -1, -1)))
    marks_b = [
        {
            "event": "s1d7_band_counter_C4.S1d.7",
            "state_index": state_index,
            "s1d7_band_counters": {
                "byte_proxies": {
                    "band_a_bytes": 1,
                    "band_c_bytes": 2,
                    "band_e_bytes": 0,
                },
                "counts": {"crossing_indices_len": 1 if state_index == 0 else 0},
            },
        }
        for state_index in range(10)
    ]
    receipt = build_ca_band_counter_confirmation_receipt(
        confirmation_root=__import__("pathlib").Path("/tmp/f3b_order_test"),
        n_states=32,
        run_a={"wall_seconds": 1.0},
        run_b={"wall_seconds": 2.0, "eligible_module_limit": 32},
        marks_b=marks_b,
        sampled_states=tuple(range(10)),
    )
    assert receipt["order_control_active"] is True
    assert receipt["order_perturbation_kind"] == "sampled_block_order_perturbation"
    assert receipt["sampled_state_order"] == list(range(9, -1, -1))
    assert receipt["effective_visit_order"][:10] == list(range(9, -1, -1))
    assert receipt["effective_visit_order"][10:] == list(range(10, 32))
    assert all(
        row["semantic_state_id"] == row["state_index"] for row in receipt["per_state"]
    )


def test_legacy_sampled_states_default_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(OBMALLOC_EXPANDED_SAMPLED_STATES_ENV, raising=False)
    monkeypatch.delenv(OBMALLOC_EXPANDED_SAMPLED_STATE_ORDER_ENV, raising=False)
    assert resolve_obmalloc_expanded_sampled_states(32) == frozenset({0, 10, 21, 31})
    assert resolve_obmalloc_expanded_sampled_state_order(
        32,
        resolve_obmalloc_expanded_sampled_states(32),
    ) is None
