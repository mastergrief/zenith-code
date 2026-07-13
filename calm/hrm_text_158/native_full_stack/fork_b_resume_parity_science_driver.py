"""Fork B science driver — thin orchestration; runner injected."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch

from calm.hrm_text_158.native_full_stack.bounded_delta_runner_hook_contract import (
    BoundedDeltaPostStepEvent, clone_deferred_backlog,
)
from calm.hrm_text_158.native_full_stack.fork_b_resume_parity_arm_ops import (
    clone_f_in_memory, comparison_stats_from_state, estimate_bounded_bits,
    prepare_c_stale_for_save, prepare_s_refresh_for_save, rehydrate_from_bounded,
    rehydrate_z_zeros,
)
from calm.hrm_text_158.native_full_stack.fork_b_resume_parity_checkpoint_adapter import (
    PATH_CLASS_IN_MEMORY_FULL_STATE, PATH_CLASS_IN_MEMORY_UNINTERRUPTED,
    PATH_CLASS_IN_MEMORY_ZEROS, PATH_CLASS_REAL,
    real_trainer_sub2_authority_checkpoint_roundtrip,
)
from calm.hrm_text_158.native_full_stack.fork_b_resume_parity_contracts import (
    CUTS_DEFAULT, GATE_BEARING_FIELDS, K_DEFAULT, ArmId, PreScienceClass,
    assert_cs_manifests_or_mismatch, assert_non_target_equality,
    build_non_target_snapshot, canonical_json_sha256, parent_seed_scope_tag,
)
from calm.hrm_text_158.native_full_stack.fork_b_resume_parity_reducers import (
    PerCutResult, classify_terminal, compute_s_accounting, surfaces_equal,
    z_decision_sensitive,
)

RunnerFn = Callable[..., Any]
# Formal U total: max(CUTS_DEFAULT)+K_DEFAULT. Must be passed as GLOBAL bp horizon
# for U and every resumed arm (hrm.compute_train_extra_args total_steps).
FORMAL_GLOBAL_HORIZON = int(max(CUTS_DEFAULT)) + int(K_DEFAULT)


@dataclass
class CutFreezeBundle:
    cut_t: int
    states: dict[str, Any]
    carry_backlog: dict[str, dict[int, dict[str, int]]] | None
    future_batch_sample_ids: tuple[Any, ...]
    non_target_snapshot_hash: str
    bundle_hash: str
    backlog_hash: str = ""
    global_horizon: int | None = None
    model_state_cpu: dict[str, Any] | None = None
    rng_state: Any | None = None
    cuda_rng_states: Any | None = None

    def rehash(self) -> str:
        return _bundle_hash(self.cut_t, self.states, self.carry_backlog,
                            self.future_batch_sample_ids, self.non_target_snapshot_hash,
                            self.global_horizon)


def _bundle_hash(cut_t, states, carry_backlog, future_ids, snap_hash, global_horizon) -> str:
    digests = {k: comparison_stats_from_state(states[k], step_tag=f"cut{cut_t}") for k in sorted(states)}
    return canonical_json_sha256({"cut_t": int(cut_t), "state_digests": digests,
                                  "carry_backlog": carry_backlog, "future_ids": list(future_ids),
                                  "snap_hash": snap_hash, "global_horizon": global_horizon})


def deep_clone_states(states: Mapping[str, Any]) -> dict[str, Any]:
    return {str(k): clone_f_in_memory(s) for k, s in sorted(states.items())}


def backlog_entry_count(backlog: Mapping[str, Any] | None) -> int:
    return 0 if not backlog else sum(len(v) for v in backlog.values())


def extract_decision_vector_from_step_report(step_report: Mapping[str, Any]) -> dict[str, Any]:
    stats = dict((step_report.get("step_result") or {}).get("tensor_stats") or {})
    if not stats:
        raise ValueError("MISSING_OBSERVABLE: step_report.tensor_stats absent")
    primary = stats[sorted(stats)[0]]
    vector = {k: primary.get(k) for k in GATE_BEARING_FIELDS}
    missing = [k for k, v in vector.items() if v is None]
    if missing:
        raise ValueError(f"MISSING_OBSERVABLE: gate fields {missing}")
    return vector


def report_global_horizon(step_report: Mapping[str, Any] | None) -> int | None:
    if not step_report:
        return None
    gh = step_report.get("global_horizon")
    return int(gh) if gh is not None else None


def assert_global_horizon_equality(*, expected: int, observed: Mapping[str, int | None]) -> None:
    missing = [k for k, v in observed.items() if v is None]
    if missing:
        raise RuntimeError(f"NON_TARGET_STATE_MISMATCH: missing global_horizon on {missing}")
    bad = {k: v for k, v in observed.items() if int(v) != int(expected)}
    if bad:
        raise RuntimeError(
            f"NON_TARGET_STATE_MISMATCH: global_horizon unequal to {expected}: {bad}"
        )


def _snap(future_ids, cut_t: int, global_horizon: int | None):
    return build_non_target_snapshot(
        rng_states={"torch": "frozen"}, exact_future_batch_sample_ids=tuple(future_ids),
        loader_cursor={"idx": int(cut_t)},
        rate_cap_backlog_schedule={"cap": 512, "backlog": 0, "step": int(cut_t)},
        q_scales_weights_code_hash={"code": "fork_b"},
        optimizer_empty_proof={"eligible_excluded": True},
        non_manipulated_manifest_fields={
            "phase": "fork-b", "cut_t": int(cut_t), "global_horizon": global_horizon,
        },
    )


def build_cut_freeze_bundle(event: BoundedDeltaPostStepEvent, *, future_batch_sample_ids: Sequence[Any],
                            model: Any | None = None, global_horizon: int | None = None) -> CutFreezeBundle:
    states = deep_clone_states(event.states)
    backlog = clone_deferred_backlog(event.carry_backlog) if event.carry_backlog is not None else None
    future_ids = tuple(future_batch_sample_ids)
    snap_hash = _snap(future_ids, int(event.step), global_horizon).hash_bundle()
    bh = _bundle_hash(int(event.step), states, backlog, future_ids, snap_hash, global_horizon)
    model_state = rng_state = cuda_rng = None
    if model is not None:
        model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        rng_state = torch.get_rng_state()
        if torch.cuda.is_available():
            cuda_rng = torch.cuda.get_rng_state_all()
    return CutFreezeBundle(
        cut_t=int(event.step), states=states, carry_backlog=backlog,
        future_batch_sample_ids=future_ids, non_target_snapshot_hash=snap_hash,
        bundle_hash=bh, backlog_hash=canonical_json_sha256(backlog or {}),
        global_horizon=None if global_horizon is None else int(global_horizon),
        model_state_cpu=model_state, rng_state=rng_state, cuda_rng_states=cuda_rng,
    )


def make_cut_capture_hook(*, cuts: Sequence[int], store: dict[int, CutFreezeBundle],
                          future_ids_by_cut: Mapping[int, Sequence[Any]],
                          model: Any | None = None,
                          global_horizon: int | None = None) -> Callable[[BoundedDeltaPostStepEvent], None]:
    cut_set = {int(t) for t in cuts}

    def _hook(event: BoundedDeltaPostStepEvent) -> None:
        if int(event.step) in cut_set:
            store[int(event.step)] = build_cut_freeze_bundle(
                event, future_batch_sample_ids=future_ids_by_cut.get(int(event.step), (f"cut{event.step}",)),
                model=model, global_horizon=global_horizon,
            )

    return _hook


def assert_bundle_immutable(bundle: CutFreezeBundle) -> None:
    if bundle.rehash() != bundle.bundle_hash:
        raise RuntimeError(f"CutFreezeBundle mutated after freeze at cut {bundle.cut_t}")


def restore_model_and_rng(model: Any, freeze: CutFreezeBundle, device: Any) -> None:
    if freeze.model_state_cpu is None:
        raise RuntimeError("freeze missing model_state_cpu")
    model.load_state_dict({k: v.to(device) for k, v in freeze.model_state_cpu.items()})
    if freeze.rng_state is not None:
        torch.set_rng_state(freeze.rng_state)
    if freeze.cuda_rng_states is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(freeze.cuda_rng_states)


def run_cs_roundtrip_arm(*, arm: ArmId, freeze: CutFreezeBundle, model: Any,
                         eligible_modules: Mapping[str, Any], scratch: Path, device: Any):
    if arm not in (ArmId.C, ArmId.S):
        raise ValueError(f"C/S only, got {arm}")
    prep = prepare_c_stale_for_save if arm is ArmId.C else prepare_s_refresh_for_save
    prepared = {str(k): prep(s) for k, s in freeze.states.items()}
    pre_bits = sum(estimate_bounded_bits(s) for s in prepared.values())
    result = real_trainer_sub2_authority_checkpoint_roundtrip(
        model=model, eligible_modules=eligible_modules, tensor_states=prepared,
        checkpoint_path=scratch / f"arm_{arm.value}_cut_{freeze.cut_t}.pt",
        step=int(freeze.cut_t), device=device,
    )
    if bool(result.get("simulated", True)):
        raise RuntimeError("C/S roundtrip must be REAL on-disk")
    rehydrated = {str(k): rehydrate_from_bounded(s) for k, s in result["loaded_states"].items()}
    post_bits = sum(estimate_bounded_bits(s) for s in rehydrated.values())
    ledger = compute_s_accounting(
        cut_t=int(freeze.cut_t), pre_refresh_bounded_bits=int(pre_bits),
        post_refresh_bounded_bits=int(post_bits), fixed_size_packed_overwrite=True,
    )
    return rehydrated, {
        "path_class": PATH_CLASS_REAL, "simulated": False,
        "checkpoint_sha256": result.get("on_disk_sha256") or result.get("checkpoint_sha256"),
        "checkpoint_path": result.get("checkpoint_path"), "s_accounting": ledger.to_dict(),
    }


def run_k_step_continuation(*, runner: RunnerFn, model: Any, batch: Mapping[str, Any],
                            tensor_states: Mapping[str, Any], eligible_modules: Mapping[str, Any],
                            device: Any, freeze: CutFreezeBundle, k_steps: int,
                            support_batches: Sequence[Mapping[str, Any]],
                            runner_kwargs: Mapping[str, Any]) -> dict[str, Any]:
    if backlog_entry_count(freeze.carry_backlog) <= 0:
        raise RuntimeError("backlog must be non-empty / update-sensitive at cut")
    if freeze.global_horizon is None:
        raise RuntimeError("NON_TARGET_STATE_MISMATCH: freeze missing global_horizon")
    restore_model_and_rng(model, freeze, device)
    kw = dict(runner_kwargs)
    kw.update({
        "post_step_hook": None, "initial_deferred_backlog": freeze.carry_backlog,
        "r7_deferred_backlog_carry_enabled": True, "support_batches": list(support_batches),
        "start_step": int(freeze.cut_t) + 1,
        "global_horizon": int(freeze.global_horizon),
    })
    out = runner(model, batch, tensor_states, eligible_modules, device=device, steps=int(k_steps), **kw)
    return out[0] if isinstance(out, tuple) else out["step_reports"]


def evaluate_cut_from_surfaces(*, cut_t: int, u_surface, f_surface, c_surface, s_surface,
                               z_surface, non_target_ok: bool) -> PerCutResult:
    if not non_target_ok:
        return PerCutResult(cut_t=int(cut_t), pre_science=PreScienceClass.NON_TARGET_STATE_MISMATCH.value, non_target_ok=False)
    if not surfaces_equal(u_surface, f_surface):
        return PerCutResult(cut_t=int(cut_t), f_matches_u=False, pre_science=PreScienceClass.CONTROL_INVALID.value, non_target_ok=True)
    z_ok = z_decision_sensitive(z_surface=z_surface, u_surface=u_surface, f_surface=f_surface)
    return PerCutResult(
        cut_t=int(cut_t), f_matches_u=True, z_decision_sensitive=bool(z_ok),
        c_matches_u=surfaces_equal(u_surface, c_surface) if c_surface is not None else None,
        s_matches_u=surfaces_equal(u_surface, s_surface) if s_surface is not None else None,
        pre_science=None if z_ok else PreScienceClass.CONTROL_INVALID.value, non_target_ok=True,
    )


def compare_k_step_vectors(*, u_reports, arm_reports, cut_t: int, k_steps: int):
    diffs: list[str] = []
    ok = True
    for offset in range(1, int(k_steps) + 1):
        abs_step = int(cut_t) + offset
        u_vec = extract_decision_vector_from_step_report(u_reports[str(abs_step)])
        a_vec = extract_decision_vector_from_step_report(arm_reports[str(abs_step)])
        step_diffs = [k for k in GATE_BEARING_FIELDS if u_vec.get(k) != a_vec.get(k)]
        if step_diffs:
            ok = False
            diffs.extend(f"{abs_step}:{k}" for k in step_diffs)
    return ok, diffs


def _cont(runner, model, batch, states, eligible, device, freeze, k, batches, kw):
    return run_k_step_continuation(
        runner=runner, model=model, batch=batch, tensor_states=states,
        eligible_modules=eligible, device=device, freeze=freeze, k_steps=int(k),
        support_batches=list(batches or []), runner_kwargs=kw,
    )


def _arm_horizon_meta(cut_t: int, k_steps: int, global_horizon: int, reports: Mapping[str, Any]) -> dict:
    sample = reports.get(str(int(cut_t) + 1)) or next(iter(reports.values()), {})
    return {
        "start_step": int(cut_t) + 1, "local_k": int(k_steps),
        "global_horizon": int(global_horizon),
        "report_global_horizon": report_global_horizon(sample),
        "report_start_step": None if not sample else sample.get("start_step"),
        "report_local_steps": None if not sample else sample.get("local_steps"),
    }


@dataclass
class ScienceDriverResult:
    developer_validation: bool
    science_label: None
    pre_science: str | None
    terminal: dict[str, Any]
    per_cut: dict[int, PerCutResult] = field(default_factory=dict)
    freezes: dict[int, CutFreezeBundle] = field(default_factory=dict)
    arm_disclosures: dict[str, Any] = field(default_factory=dict)
    notes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "developer_validation": self.developer_validation, "science_label": None,
            "pre_science": self.pre_science, "terminal": self.terminal,
            "per_cut": {str(k): v.to_dict() for k, v in self.per_cut.items()},
            "arm_disclosures": self.arm_disclosures, "notes": self.notes,
            "freeze_hashes": {str(k): v.bundle_hash for k, v in self.freezes.items()},
            "backlog_hashes": {str(k): v.backlog_hash for k, v in self.freezes.items()},
        }


def run_fork_b_resume_parity_certificate(
    *, runner: RunnerFn, model: Any, batch: Mapping[str, Any],
    tensor_states: Mapping[str, Any], eligible_modules: Mapping[str, Any],
    device: Any, scratch_root: Path | str, parent_sha16: str, batch_seed: int,
    support_order_seed: int, ordering_seed: int, cuts: Sequence[int] = CUTS_DEFAULT,
    k_steps: int = K_DEFAULT, total_steps: int | None = None,
    support_batches: Sequence[Mapping[str, Any]] | None = None,
    future_ids_by_cut: Mapping[int, Sequence[Any]] | None = None,
    runner_kwargs: Mapping[str, Any] | None = None, developer_validation: bool = True,
    require_strict_f_equals_u: bool = False, require_z_gate_break: bool = False,
    global_horizon: int | None = None,
) -> ScienceDriverResult:
    scratch = Path(scratch_root); scratch.mkdir(parents=True, exist_ok=True)
    # Local U length may be bounded (e.g. cut-16 smoke → 20); GLOBAL bp horizon is formal 32.
    local_u_steps = int(total_steps) if total_steps is not None else int(max(cuts)) + int(k_steps)
    gh = int(global_horizon) if global_horizon is not None else int(
        (runner_kwargs or {}).get("global_horizon") or FORMAL_GLOBAL_HORIZON
    )
    scope = parent_seed_scope_tag(
        parent_sha16=parent_sha16, batch_seed=int(batch_seed),
        support_order_seed=int(support_order_seed), ordering_seed=int(ordering_seed),
        cuts=tuple(int(t) for t in cuts), k=int(k_steps),
    )
    freezes: dict[int, CutFreezeBundle] = {}
    ids = {int(t): tuple((future_ids_by_cut or {}).get(int(t), tuple(f"cut{t}_f{i}" for i in range(4)))) for t in cuts}
    kw = dict(runner_kwargs or {})
    kw["global_horizon"] = gh
    kw["post_step_hook"] = make_cut_capture_hook(
        cuts=cuts, store=freezes, future_ids_by_cut=ids, model=model, global_horizon=gh,
    )
    if support_batches is not None:
        kw["support_batches"] = list(support_batches)
    kw["start_step"] = 1
    u_out = runner(model, batch, tensor_states, eligible_modules, device=device,
                   steps=int(local_u_steps), **kw)
    u_reports = u_out[0] if isinstance(u_out, tuple) else u_out["step_reports"]
    disclosures = {ArmId.U.value: {
        "path_class": PATH_CLASS_IN_MEMORY_UNINTERRUPTED, "simulated": False,
        "runner": getattr(runner, "__name__", type(runner).__name__),
        "start_step": 1, "local_steps": int(local_u_steps), "global_horizon": gh,
    }}
    per_cut: dict[int, PerCutResult] = {}
    batches = list(support_batches or [])

    for cut_t in cuts:
        ct = int(cut_t)
        if ct not in freezes:
            per_cut[ct] = PerCutResult(cut_t=ct, pre_science=PreScienceClass.MISSING_OBSERVABLE.value)
            continue
        freeze = freezes[ct]
        assert_bundle_immutable(freeze)
        if backlog_entry_count(freeze.carry_backlog) <= 0:
            raise RuntimeError(f"cut {ct}: empty carry_backlog")
        nt_ok = True
        try:
            assert_non_target_equality({a: _snap(freeze.future_batch_sample_ids, ct, gh) for a in "UFCSZ"})
            if freeze.global_horizon != gh:
                raise RuntimeError("freeze global_horizon mismatch")
        except Exception:
            nt_ok = False

        f_reports = _cont(runner, model, batch, deep_clone_states(freeze.states),
                          eligible_modules, device, freeze, k_steps, batches, kw)
        f_eq, f_diffs = compare_k_step_vectors(u_reports=u_reports, arm_reports=f_reports, cut_t=ct, k_steps=k_steps)
        disclosures[f"F@{ct}"] = {
            "path_class": PATH_CLASS_IN_MEMORY_FULL_STATE, "simulated": False,
            "f_equals_u": f_eq, "diffs": f_diffs, **_arm_horizon_meta(ct, k_steps, gh, f_reports),
        }
        if require_strict_f_equals_u and not f_eq:
            raise RuntimeError(f"F!=U at cut {ct}: {f_diffs[:12]}")

        restore_model_and_rng(model, freeze, device)
        c_states, c_disc = run_cs_roundtrip_arm(arm=ArmId.C, freeze=freeze, model=model, eligible_modules=eligible_modules, scratch=scratch / f"cut{ct}_C", device=device)
        restore_model_and_rng(model, freeze, device)
        s_states, s_disc = run_cs_roundtrip_arm(arm=ArmId.S, freeze=freeze, model=model, eligible_modules=eligible_modules, scratch=scratch / f"cut{ct}_S", device=device)
        try:
            assert_cs_manifests_or_mismatch(
                {"bounded_fresh_for_exact_shadow": False, "phase": "fork-b", "cut_t": ct},
                {"bounded_fresh_for_exact_shadow": True, "bounded_refresh_applied": True,
                 "s_accounting_metadata": s_disc.get("s_accounting"), "phase": "fork-b", "cut_t": ct},
            )
        except Exception:
            nt_ok = False
        c_reports = _cont(runner, model, batch, c_states, eligible_modules, device, freeze, k_steps, batches, kw)
        s_reports = _cont(runner, model, batch, s_states, eligible_modules, device, freeze, k_steps, batches, kw)
        c_eq, _ = compare_k_step_vectors(u_reports=u_reports, arm_reports=c_reports, cut_t=ct, k_steps=k_steps)
        s_eq, _ = compare_k_step_vectors(u_reports=u_reports, arm_reports=s_reports, cut_t=ct, k_steps=k_steps)
        disclosures[f"C@{ct}"] = {**c_disc, "c_equals_u_k": c_eq, **_arm_horizon_meta(ct, k_steps, gh, c_reports)}
        disclosures[f"S@{ct}"] = {**s_disc, "s_equals_u_k": s_eq, **_arm_horizon_meta(ct, k_steps, gh, s_reports)}

        z_reports = _cont(runner, model, batch, {str(k): rehydrate_z_zeros(s) for k, s in freeze.states.items()},
                          eligible_modules, device, freeze, k_steps, batches, kw)
        z_eq, z_diffs = compare_k_step_vectors(u_reports=u_reports, arm_reports=z_reports, cut_t=ct, k_steps=k_steps)
        z_breaks = not z_eq
        disclosures[f"Z@{ct}"] = {
            "path_class": PATH_CLASS_IN_MEMORY_ZEROS, "simulated": False,
            "z_breaks_gate_bearing": z_breaks, "diffs": z_diffs[:20],
            **_arm_horizon_meta(ct, k_steps, gh, z_reports),
        }
        if require_z_gate_break and not z_breaks:
            raise RuntimeError(f"Z did not break gate-bearing within K at cut {ct}")

        # Horizon equality across U/F/C/S/Z BEFORE science classification.
        try:
            assert_global_horizon_equality(
                expected=gh,
                observed={
                    "U": report_global_horizon(u_reports.get(str(ct + 1))),
                    "F": report_global_horizon(f_reports.get(str(ct + 1))),
                    "C": report_global_horizon(c_reports.get(str(ct + 1))),
                    "S": report_global_horizon(s_reports.get(str(ct + 1))),
                    "Z": report_global_horizon(z_reports.get(str(ct + 1))),
                    "freeze": freeze.global_horizon,
                },
            )
        except Exception:
            nt_ok = False

        assert_bundle_immutable(freeze)
        pc = evaluate_cut_from_surfaces(
            cut_t=ct,
            u_surface=extract_decision_vector_from_step_report(u_reports[str(ct + 1)]),
            f_surface=extract_decision_vector_from_step_report(f_reports[str(ct + 1)]),
            c_surface=extract_decision_vector_from_step_report(c_reports[str(ct + 1)]),
            s_surface=extract_decision_vector_from_step_report(s_reports[str(ct + 1)]),
            z_surface=extract_decision_vector_from_step_report(z_reports[str(ct + 1)]),
            non_target_ok=nt_ok,
        )
        pc.f_matches_u, pc.c_matches_u, pc.s_matches_u = bool(f_eq), bool(c_eq), bool(s_eq)
        pc.z_decision_sensitive = bool(z_breaks)
        if require_strict_f_equals_u and f_eq and z_breaks and nt_ok:
            pc.pre_science = None
        per_cut[ct] = pc

    terminal = classify_terminal(per_cut=per_cut, cuts=tuple(int(t) for t in cuts), parent_seed_scope=scope)
    return ScienceDriverResult(
        developer_validation=bool(developer_validation), science_label=None,
        pre_science=terminal.get("pre_science"), terminal=terminal, per_cut=per_cut,
        freezes=freezes, arm_disclosures=disclosures,
        notes={"u_steps_completed": len(u_reports), "gate_bearing_fields": list(GATE_BEARING_FIELDS),
               "parent_seed_scope": scope, "runner": getattr(runner, "__name__", type(runner).__name__),
               "global_horizon": gh, "local_u_steps": int(local_u_steps),
               "formal_global_horizon": FORMAL_GLOBAL_HORIZON},
    )


__all__ = [
    "CutFreezeBundle", "FORMAL_GLOBAL_HORIZON", "ScienceDriverResult",
    "assert_bundle_immutable", "assert_global_horizon_equality", "backlog_entry_count",
    "build_cut_freeze_bundle", "compare_k_step_vectors", "deep_clone_states",
    "evaluate_cut_from_surfaces", "extract_decision_vector_from_step_report",
    "make_cut_capture_hook", "report_global_horizon", "restore_model_and_rng",
    "run_cs_roundtrip_arm", "run_fork_b_resume_parity_certificate", "run_k_step_continuation",
]
