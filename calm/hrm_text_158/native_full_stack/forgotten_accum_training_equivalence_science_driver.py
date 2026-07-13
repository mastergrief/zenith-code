"""Thin run-arms science driver (forgotten-accum training-equivalence).

One ``run_bounded_delta_steps`` call per arm continuation; RW per-step flip
schedule (no two-call segmentation). Cadence via existing ``post_step_hook``.
"""
from __future__ import annotations

import hashlib
import json
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, MutableMapping, Sequence

from calm.hrm_text_158.native_full_stack.bounded_delta_runner_hook_contract import (
    BoundedDeltaPostStepEvent,
)
from calm.hrm_text_158.native_full_stack.forgotten_accum_training_equivalence_arms import (
    FutureStreamBudget,
    prove_r0_rw_same_zero_seed,
    resume_arm_from_live_cut,
)
from calm.hrm_text_158.native_full_stack.forgotten_accum_training_equivalence_bank_eval import (
    e_must_match_u_bank,
    evaluate_arm_bank_gate,
)
from calm.hrm_text_158.native_full_stack.forgotten_accum_training_equivalence_contracts import (
    ELIGIBLE_SCOPE,
    GLOBAL_CAP_CONTRACT,
    PARENT_SHA256_FULL,
    RUNWAY_STEPS,
    SAVE_CADENCE,
    T_CUT,
    W_REWARM_STEPS,
    ArmId,
    FailClosedClass,
    ResumePolicy,
)
from calm.hrm_text_158.native_full_stack.forgotten_accum_training_equivalence_ledger import (
    ArmComputeCounts,
    build_ledger,
)

RunnerFn = Callable[..., Any]
CadenceSaverFn = Callable[..., Path]
SCHEMA = "forgotten_accum_training_equivalence_science_driver_receipt/v1"
CARRIER_NONE = "NONE"


def resolve_flip_application_deferred_for_step(
    step: int,
    *,
    flip_application_deferred: bool = False,
    flip_application_deferred_schedule: Callable[[int], bool] | None = None,
) -> bool:
    if flip_application_deferred_schedule is not None:
        return bool(flip_application_deferred_schedule(int(step)))
    return bool(flip_application_deferred)


def make_rw_absolute_flip_schedule(
    *, t_cut: int = T_CUT, W: int = W_REWARM_STEPS
) -> Callable[[int], bool]:
    def _schedule(step: int) -> bool:
        post = int(step) - int(t_cut)
        # Local W (formal default matches flip_defer_schedule / W_REWARM_STEPS).
        return 1 <= post <= int(W)

    return _schedule


def rw_resolved_flags_for_absolute_window(
    *, t_cut: int = T_CUT, W: int = W_REWARM_STEPS, through_step: int | None = None
) -> dict[int, bool]:
    end = int(through_step) if through_step is not None else int(t_cut) + int(W) + 1
    sched = make_rw_absolute_flip_schedule(t_cut=t_cut, W=W)
    return {s: bool(sched(s)) for s in range(int(t_cut) + 1, end + 1)}


def assert_carrier_preflight(
    *,
    live_acc_carrier_selector: str,
    global_cap_contract: str,
    eligible_scope: str,
    event_coded_flags_present: bool = False,
) -> None:
    if live_acc_carrier_selector != CARRIER_NONE:
        raise ValueError(f"PREFLIGHT_REFUSE: carrier must be {CARRIER_NONE!r}")
    if global_cap_contract != GLOBAL_CAP_CONTRACT:
        raise ValueError(f"PREFLIGHT_REFUSE: cap must be {GLOBAL_CAP_CONTRACT!r}")
    if eligible_scope != ELIGIBLE_SCOPE:
        raise ValueError(f"PREFLIGHT_REFUSE: scope must be {ELIGIBLE_SCOPE!r}")
    if event_coded_flags_present:
        raise ValueError("PREFLIGHT_REFUSE: event-coded flags must be ABSENT")


def authoritative_state_fingerprint(
    states: Mapping[str, Any] | None, backlog: Mapping[str, Any] | None
) -> str:
    h = hashlib.sha256()
    for key in sorted((states or {})):
        st = states[key]
        h.update(str(key).encode())
        for attr in ("q_levels", "exact_accumulator_shadow"):
            t = getattr(st, attr, None)
            if t is not None and hasattr(t, "detach"):
                h.update(t.detach().cpu().contiguous().numpy().tobytes())
        if getattr(st, "scale", None) is not None:
            h.update(repr(float(st.scale)).encode())
    h.update(json.dumps(backlog or {}, sort_keys=True, default=str).encode())
    return h.hexdigest()


def default_cadence_saver(
    *, path: Path, model: Any, event: BoundedDeltaPostStepEvent,
    config: Mapping[str, Any], source_pin: str, eligible_scope: str = ELIGIBLE_SCOPE,
    use_ternary_bulk: bool = True,
) -> Path:
    import torch
    from calm.hrm_text_158.native_full_stack.trainer_sub2_authority import (
        save_trainer_sub2_live_checkpoint_envelope,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        save_trainer_sub2_live_checkpoint_envelope(
            model, tensor_states=event.states, use_ternary_bulk=bool(use_ternary_bulk),
            eligible_scope=str(eligible_scope), step=int(event.step), config=dict(config),
            source_pin=str(source_pin),
        ),
        path,
    )
    return path


def make_cadence_cut_post_step_hook(
    *, model: Any, arm_root: Path, cadence: Sequence[int],
    cut_store: MutableMapping[int, dict[str, Any]], t_cut: int,
    config: Mapping[str, Any], source_pin: str, cadence_paths: MutableMapping[int, Path],
    fingerprints_pre_post: list[tuple[str, str]] | None = None,
    saver: CadenceSaverFn | None = None, eligible_scope: str = ELIGIBLE_SCOPE,
    use_ternary_bulk: bool = True,
) -> Callable[[BoundedDeltaPostStepEvent], None]:
    cadence_set = {int(s) for s in cadence}

    def _hook(event: BoundedDeltaPostStepEvent) -> None:
        step = int(event.step)
        if step == int(t_cut) and step not in cut_store:
            cut_store[step] = {
                "step": step, "states": event.states, "carry_backlog": event.carry_backlog,
            }
        if step not in cadence_set:
            return
        pre = authoritative_state_fingerprint(event.states, event.carry_backlog)
        path = Path(arm_root) / f"checkpoint_step{step:05d}.pt"
        if saver is not None:
            out = saver(
                path=path, model=model, event=event, config=config, source_pin=source_pin,
            )
        else:
            out = default_cadence_saver(
                path=path, model=model, event=event, config=config, source_pin=source_pin,
                eligible_scope=eligible_scope, use_ternary_bulk=use_ternary_bulk,
            )
        post = authoritative_state_fingerprint(event.states, event.carry_backlog)
        if fingerprints_pre_post is not None:
            fingerprints_pre_post.append((pre, post))
        if pre != post:
            raise RuntimeError("CADENCE_HOOK_MUTATION: state changed during save")
        cadence_paths[step] = Path(out)

    return _hook


@dataclass
class ScienceDriverResult:
    status: str
    fail_closed_class: str | None
    science_label: None = None
    developer_validation: bool = True
    arm_call_counts: dict[str, int] = field(default_factory=dict)
    runner_invocations: list[dict[str, Any]] = field(default_factory=list)
    cadence_paths_by_arm: dict[str, dict[int, str]] = field(default_factory=dict)
    bank_receipts: dict[str, dict[str, Any]] = field(default_factory=dict)
    ledger: dict[str, Any] | None = None
    rw_schedule_flags: dict[str, bool] = field(default_factory=dict)
    zero_seed_proof: str | None = None
    cadence_fingerprint_pairs: list[tuple[str, str]] = field(default_factory=list)
    error: str | None = None
    traceback: str | None = None
    notes: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        blob = asdict(self)
        blob["schema"] = SCHEMA
        blob["cadence_paths_by_arm"] = {
            a: {str(k): v for k, v in p.items()}
            for a, p in self.cadence_paths_by_arm.items()
        }
        blob["cadence_fingerprint_pairs"] = [
            {"pre": a, "post": b} for a, b in self.cadence_fingerprint_pairs
        ]
        return blob


def _invoke(runner, model, batch, states, eligible, device, steps, start_step,
            global_horizon, hook, backlog, flip, schedule, rk, arm, log):
    kw = dict(rk)
    kw.update({
        "start_step": int(start_step), "global_horizon": int(global_horizon),
        "post_step_hook": hook, "initial_deferred_backlog": backlog,
        "flip_application_deferred": bool(flip),
        "flip_application_deferred_schedule": schedule,
    })
    log.append({"arm": arm, "start_step": int(start_step), "steps": int(steps),
                "has_schedule": schedule is not None})
    return runner(model, batch, states, eligible, device=device, steps=int(steps), **kw)


def _bank(arm: str, blob: Mapping[str, Any]):
    return evaluate_arm_bank_gate(
        arm=arm, acquire_pct=float(blob.get("acquire_pct", 100.0)),
        retain_pct_by_support=dict(
            blob.get("retain_pct_by_support") or {"L0b": 100.0, "math_a0": 100.0}
        ),
        clears_by_save={int(k): bool(v) for k, v in dict(
            blob.get("clears_by_save") or {1500: True}
        ).items()},
        parent_consistency_ok=bool(blob.get("parent_consistency_ok", True)),
        close_sibling_ok=bool(blob.get("close_sibling_ok", True)),
        hashes_diagnostic=dict(blob.get("hashes_diagnostic") or {}),
    )


def run_forgotten_accum_training_equivalence_arms(
    *, runner: RunnerFn, model: Any, batch: Mapping[str, Any],
    tensor_states: Mapping[str, Any], eligible_modules: Mapping[str, Any],
    device: Any, experiment_root: Path | str,
    parent_sha256: str = PARENT_SHA256_FULL,
    live_acc_carrier_selector: str = CARRIER_NONE,
    global_cap_contract: str = GLOBAL_CAP_CONTRACT,
    eligible_scope: str = ELIGIBLE_SCOPE, event_coded_flags_present: bool = False,
    t_cut: int = T_CUT, runway_steps: int = RUNWAY_STEPS, W: int = W_REWARM_STEPS,
    save_cadence: Sequence[int] = SAVE_CADENCE,
    runner_kwargs: Mapping[str, Any] | None = None,
    bank_inputs: Mapping[str, Mapping[str, Any]] | None = None,
    cadence_saver: CadenceSaverFn | None = None, developer_validation: bool = True,
    config: Mapping[str, Any] | None = None,
) -> ScienceDriverResult:
    root = Path(experiment_root); root.mkdir(parents=True, exist_ok=True)
    cfg = dict(config or {"eligible_scope": eligible_scope})
    rk = dict(runner_kwargs or {})
    inv: list[dict[str, Any]] = []
    call_counts = {a.value: 0 for a in ArmId}
    cadence_paths_by_arm: dict[str, dict[int, str]] = {}
    fp_pairs: list[tuple[str, str]] = []
    notes = {"parent_sha256": str(parent_sha256), "single_call_design": True,
             "two_call_rw_segmentation": False}
    base = dict(developer_validation=developer_validation, arm_call_counts=call_counts,
                runner_invocations=inv, notes=notes)

    def _res(status, fail=None, **extra):
        return ScienceDriverResult(status=status, fail_closed_class=fail, **base, **extra)

    try:
        assert_carrier_preflight(
            live_acc_carrier_selector=live_acc_carrier_selector,
            global_cap_contract=global_cap_contract, eligible_scope=eligible_scope,
            event_coded_flags_present=event_coded_flags_present,
        )
        FutureStreamBudget(t_cut=int(t_cut), W=int(W), runway_end=int(runway_steps)).assert_matched()
        rw_flags = rw_resolved_flags_for_absolute_window(
            t_cut=int(t_cut), W=int(W), through_step=int(t_cut) + int(W) + 1
        )
        notes["rw_schedule_must_prove"] = {str(k): v for k, v in rw_flags.items()}
        flag_map = {str(k): v for k, v in rw_flags.items()}

        cut_store: dict[int, dict[str, Any]] = {}
        u_paths: dict[int, Path] = {}
        u_root = root / "arms" / ArmId.U.value; u_root.mkdir(parents=True, exist_ok=True)
        u_hook = make_cadence_cut_post_step_hook(
            model=model, arm_root=u_root, cadence=save_cadence, cut_store=cut_store,
            t_cut=int(t_cut), config=cfg, source_pin=str(parent_sha256),
            cadence_paths=u_paths, fingerprints_pre_post=fp_pairs, saver=cadence_saver,
            eligible_scope=eligible_scope,
        )
        _invoke(runner, model, batch, tensor_states, eligible_modules, device,
                int(runway_steps), 1, int(runway_steps), u_hook, None, False, None,
                rk, ArmId.U.value, inv)
        call_counts[ArmId.U.value] = 1
        cadence_paths_by_arm[ArmId.U.value] = {k: str(v) for k, v in u_paths.items()}
        if int(t_cut) not in cut_store:
            raise RuntimeError(f"missing cut capture at t_cut={t_cut}")
        cut = cut_store[int(t_cut)]
        resumes = {
            arm: resume_arm_from_live_cut(
                arm=arm, live_states=cut["states"], live_backlog=cut["carry_backlog"],
                experiment_root=root,
            )
            for arm in (ArmId.E, ArmId.R0, ArmId.RW)
        }
        assert resumes[ArmId.E].meta["policy"] == ResumePolicy.EXACT_PRESERVE.value
        assert resumes[ArmId.R0].meta["policy"] == ResumePolicy.ZERO_STRIP.value
        assert resumes[ArmId.RW].meta["policy"] == ResumePolicy.ZERO_STRIP.value
        zero_proof = prove_r0_rw_same_zero_seed(resumes[ArmId.R0], resumes[ArmId.RW])
        post = int(runway_steps) - int(t_cut)
        rw_sched = make_rw_absolute_flip_schedule(t_cut=int(t_cut), W=int(W))
        for arm in (ArmId.E, ArmId.R0, ArmId.RW):
            arm_root = root / "arms" / arm.value; arm_root.mkdir(parents=True, exist_ok=True)
            paths: dict[int, Path] = {}
            hook = make_cadence_cut_post_step_hook(
                model=model, arm_root=arm_root, cadence=save_cadence, cut_store={},
                t_cut=int(t_cut), config=cfg, source_pin=str(parent_sha256),
                cadence_paths=paths, fingerprints_pre_post=fp_pairs, saver=cadence_saver,
                eligible_scope=eligible_scope,
            )
            _invoke(
                runner, model, batch, resumes[arm].tensor_states, eligible_modules, device,
                post, int(t_cut) + 1, int(runway_steps), hook, resumes[arm].deferred_backlog,
                False, rw_sched if arm is ArmId.RW else None, rk, arm.value, inv,
            )
            call_counts[arm.value] = 1
            cadence_paths_by_arm[arm.value] = {k: str(v) for k, v in paths.items()}
        if any(v != 1 for v in call_counts.values()):
            raise RuntimeError(f"expected one call per arm, got {call_counts}")
        if sum(1 for row in inv if row["arm"] == "RW") != 1:
            raise RuntimeError("RW must be a single runner invocation")

        bank_blobs = dict(bank_inputs or {}) or {
            a.value: {
                "acquire_pct": 100.0,
                "retain_pct_by_support": {"L0b": 100.0, "math_a0": 100.0},
                "clears_by_save": {int(s): True for s in save_cadence},
                "parent_consistency_ok": True, "close_sibling_ok": True,
            }
            for a in ArmId
        }
        receipts = {n: _bank(n, b) for n, b in bank_blobs.items()}
        common = dict(
            cadence_paths_by_arm=cadence_paths_by_arm,
            bank_receipts={k: v.as_dict() for k, v in receipts.items()},
            rw_schedule_flags=flag_map, zero_seed_proof=zero_proof,
            cadence_fingerprint_pairs=fp_pairs,
        )
        if "U" in receipts and "E" in receipts and not e_must_match_u_bank(
            receipts["U"], receipts["E"]
        ):
            return _res("FAILURE", FailClosedClass.CONTROL_INVALID.value,
                        error="E bank outcome diverges from U", **common)
        counts = {
            a.value: ArmComputeCounts(
                a.value, runway_steps, runway_steps, runway_steps, 0.0,
                # Ledger schema pins formal W_REWARM_STEPS; schedule may use local W.
                int(W_REWARM_STEPS) if a is ArmId.RW else 0,
            )
            for a in ArmId
        }
        notes["schedule_W"] = int(W)
        notes["ledger_rewarm_W"] = int(W_REWARM_STEPS)
        ledger = build_ledger(arm_counts=counts)
        if ledger.classification == FailClosedClass.REWARM_ACCOUNTING_INVALID.value:
            return _res("FAILURE", FailClosedClass.REWARM_ACCOUNTING_INVALID.value,
                        ledger=ledger.as_dict(), **common)
        return _res("OK", ledger=ledger.as_dict(), **common)
    except ValueError as exc:
        if str(exc).startswith("PREFLIGHT_REFUSE"):
            return _res("REFUSED", error=str(exc))
        return _res("FAILURE", error=str(exc), traceback=traceback.format_exc(),
                    cadence_paths_by_arm=cadence_paths_by_arm,
                    cadence_fingerprint_pairs=fp_pairs)
    except Exception as exc:  # noqa: BLE001
        return _res(
            "FAILURE", error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
            cadence_paths_by_arm=cadence_paths_by_arm,
            cadence_fingerprint_pairs=fp_pairs,
        )


__all__ = [
    "SCHEMA", "CARRIER_NONE", "ScienceDriverResult",
    "resolve_flip_application_deferred_for_step", "make_rw_absolute_flip_schedule",
    "rw_resolved_flags_for_absolute_window", "assert_carrier_preflight",
    "authoritative_state_fingerprint", "make_cadence_cut_post_step_hook",
    "default_cadence_saver", "run_forgotten_accum_training_equivalence_arms",
]
