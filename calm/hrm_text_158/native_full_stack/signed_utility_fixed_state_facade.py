"""Thin public facade for fixed-state signed-utility diagnostic (PLAN v6 D2)."""
from __future__ import annotations

from typing import Any, Mapping

from calm.hrm_text_158.native_full_stack.signed_utility_fixed_state_authoritative_gpu import (
    call_graph_steps,
    run_authoritative_gpu_call_graph,
)
from calm.hrm_text_158.native_full_stack.signed_utility_fixed_state_driver import (
    developer_check_payload,
    run_authoritative_fixed_state_signed_utility,
    run_developer_check_cpu_static,
)


def developer_check(ff: Mapping[str, Any]) -> dict[str, Any]:
    return developer_check_payload(ff)


def authoritative_proof(packet: Mapping[str, Any]) -> dict[str, Any]:
    return run_authoritative_fixed_state_signed_utility(packet)


def authoritative_gpu_proof(packet: Mapping[str, Any]) -> dict[str, Any]:
    return run_authoritative_gpu_call_graph(packet)


def authoritative_call_graph_steps() -> tuple[str, ...]:
    return tuple(call_graph_steps())


__all__ = [
    "authoritative_call_graph_steps",
    "authoritative_gpu_proof",
    "authoritative_proof",
    "developer_check",
    "run_developer_check_cpu_static",
]
