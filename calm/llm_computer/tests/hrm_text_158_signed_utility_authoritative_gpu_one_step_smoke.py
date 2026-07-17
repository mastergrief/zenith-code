"""One-step GPU smoke for authoritative signed-utility (PLAN v6 D2c2).

Executable under Claude smoke gate. Plan-dev must not acquire GPU or run this under D2.
"""
from __future__ import annotations

import argparse, json, os
from pathlib import Path
from typing import Any, Mapping

from calm.hrm_text_158.native_full_stack.signed_utility_fixed_state_authoritative_gpu import (
    CALL_GRAPH_STEPS_V6, AuthoritativeGpuHooks, run_one_step_smoke,
)
from calm.hrm_text_158.native_full_stack.signed_utility_fixed_state_pin_validation import (
    FORMAL_SOURCE_PIN_BASENAMES, rehash_path,
)
from calm.hrm_text_158.native_full_stack.signed_utility_fixed_state_schema import (
    SCHEMA_SCIENCE, validate_authoritative_result_payload_v3,
)

DEFAULT_PARENT = Path(
    "/mnt/c/Users/gabes/projects/claw-code-hrm-text-158/calm/hrm/checkpoints/"
    "hrm_text_158_phase3_L0c1_seed0017_replay83_n12k_lr7p5e5_pc1p0_rsL0b1math1r1b2_1_anchorsv1r3_from_L0b_final_step01500.pt"
)
DEFAULT_PARENT_SHA = "9b4e311a22787e7d4808bde7bc2953568d767a2ee8ac648942a3f5dbb7b4d5ec"
REPO = Path("/mnt/c/Users/gabes/projects/claw-code-hrm-text-158")
D3A_EXPECTED_HEAD = "c93b68e9ddc3513866adc3f930a17eb80c6f5459"
STACK = REPO / "calm/hrm_text_158/native_full_stack"


def smoke_preflight_cpu() -> dict[str, Any]:
    return {
        "module": __name__,
        "call_graph_steps": list(CALL_GRAPH_STEPS_V6),
        "claim_ceiling": "implementation_correctness_only",
        "invokes_authoritative_entry": True,
        "requires_schema_validate_before_smoke_ok": True,
    }


def build_default_smoke_packet() -> dict[str, Any]:
    pins = {}
    for name in FORMAL_SOURCE_PIN_BASENAMES:
        path = STACK / name
        pins[name] = {"absolute_path": str(path), "sha256": rehash_path(path)}
    pins["watch-wrap"] = {
        "absolute_path": str(REPO / "bin/watch-wrap"),
        "sha256": rehash_path(REPO / "bin/watch-wrap"),
    }
    pins["vote_update.py"] = {
        "absolute_path": str(STACK / "vote_update.py"),
        "sha256": rehash_path(STACK / "vote_update.py"),
    }
    return {
        "authoritative_deferred": False,
        "smoke_mode": True,
        "pin_mode": "smoke",
        "device": "cuda:0",
        "repo_root": str(REPO),
        "expected_head": D3A_EXPECTED_HEAD,
        "parent_checkpoint": {"absolute_path": str(DEFAULT_PARENT), "sha256": DEFAULT_PARENT_SHA},
        "source_pins": pins,
    }


def _oexcl_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(dict(payload), fh, indent=2)
        fh.write("\n")


def _nonauthoritative_compact(result: Mapping[str, Any]) -> dict[str, Any]:
    """Summary only — never authority; full result lives under authoritative_result."""
    keys = (
        "schema", "classifier", "reason", "failed_stage", "eligible_state_key_count",
        "apply_integer_vote_update_from_frozen_plan_calls", "eval_batch_count",
    )
    return {k: result.get(k) for k in keys if k in result}


def run_smoke(*, receipt: Path, packet: Mapping[str, Any] | None = None,
              hooks: AuthoritativeGpuHooks | None = None) -> dict[str, Any]:
    print("SMOKE_BEGIN")
    pkt = dict(packet or build_default_smoke_packet())
    pkt["authoritative_deferred"] = False
    pkt["smoke_mode"] = True
    pkt.setdefault("pin_mode", "smoke")
    result = run_one_step_smoke(pkt, hooks=hooks)
    validate_authoritative_result_payload_v3(result)
    route = list(result.get("route") or [])
    science_ok = (
        result.get("schema") == SCHEMA_SCIENCE
        and result.get("observer_public_apply_calibration", {}).get("ok") is True
        and isinstance(result.get("current_weights_sha256_by_arm"), dict)
        and set(result["current_weights_sha256_by_arm"]) >= {"prod", "inv", "noop", "noop_repeat"}
        and "capture_backward_vote" in route
        and "calibrate_capture_vs_public_apply" in route
        and "eval_nll_three_arm_plus_noop_repeat" in route
        and "emit_in_memory_payload" in route
    )
    status = "SMOKE_OK" if science_ok else "SMOKE_FAIL"
    if status == "SMOKE_FAIL":
        print(
            "SMOKE_FAIL",
            f"schema={result.get('schema')}",
            f"classifier={result.get('classifier')}",
            f"reason={result.get('reason')}",
            f"failed_stage={result.get('failed_stage')}",
            f"route_tail={route[-5:]}",
        )
    else:
        print("SMOKE_OK")
        for step in route:
            print(f"SMOKE_PHASE_{step}")
    out = {
        "status": status,
        "claim_ceiling": "implementation_correctness_only",
        "authoritative_result": dict(result),
        "authoritative_result_compact": _nonauthoritative_compact(result),
        "compact_is_authoritative": False,
        "preflight_cpu": smoke_preflight_cpu(),
    }
    _oexcl_write(receipt, out)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Authoritative GPU one-step smoke")
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--packet", type=Path, default=None)
    args = parser.parse_args(argv)
    packet = json.loads(args.packet.read_text()) if args.packet else None
    try:
        out = run_smoke(receipt=args.receipt, packet=packet, hooks=None)
    except FileExistsError:
        print("SMOKE_FAIL")
        print("receipt_exists_oexcl")
        return 2
    except Exception as exc:  # noqa: BLE001
        print("SMOKE_FAIL")
        try:
            _oexcl_write(args.receipt, {"status": "SMOKE_FAIL", "error": type(exc).__name__, "detail": str(exc)})
        except FileExistsError:
            pass
        return 2
    return 0 if out.get("status") == "SMOKE_OK" else 2


if __name__ == "__main__":
    raise SystemExit(main())
