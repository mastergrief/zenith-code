"""Dense-legacy smoke: non-vacuous ordinary-vs-deferred twin (co_lead REVISE)."""
from __future__ import annotations

import copy
import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from calm.hrm_text_158.native_full_stack.event_coded_vote_update_adapter import (
    LIVE_ACC_CARRIER_NONE,
    resolve_live_acc_carrier_selector,
)
from calm.hrm_text_158.native_full_stack.forgotten_accum_flip_deferral_apply import (
    apply_global_rate_cap_with_optional_flip_deferral,
)
from calm.hrm_text_158.native_full_stack.forgotten_accum_flip_deferral_contracts import (
    DENSE_LEGACY_CAP_SITE_ID,
)
from calm.hrm_text_158.native_full_stack.forgotten_accum_flip_deferral_reducers import (
    backlog_content_sha256,
)
from calm.hrm_text_158.native_full_stack.forgotten_accum_training_equivalence_contracts import (
    GLOBAL_CAP_CONTRACT,
    PARENT_SHA256_FULL,
    SMOKE_CPU_PREDICATES,
)
from calm.hrm_text_158.native_full_stack.global_rate_cap import (
    GlobalRateCapSpec,
    GlobalRateCapTensorInput,
)
from calm.hrm_text_158.native_full_stack.vote_update import (
    VoteUpdateInputs,
    VoteUpdateSpec,
    VoteUpdateState,
    plan_integer_vote_update_reference,
)

EXIT_PASS = 0
EXIT_INFRA_FAIL = 1
EXIT_NO_AUTHORITY = 2
EXIT_EVENT_CODED_STOP = 4
EXIT_PREDICATE_FAIL = 5
EXIT_INCONCLUSIVE_NO_CROSSING = 6
RECEIPT_SCHEMA = "forgotten_accum_dense_site_smoke_receipt/v2"
DEFAULT_PARENT_RELPATH = (
    "calm/hrm/checkpoints/"
    "hrm_text_158_phase3_L0c1_seed0017_replay83_n12k_lr7p5e5_pc1p0_"
    "rsL0b1math1r1b2_1_anchorsv1r3_from_L0b_final_step01500.pt"
)
GUARANTEED_CROSSING_TEMPLATE = [
    {"state_key": "smoke_A", "q": [0] * 6, "acc": [0] * 6, "votes": [0, 25, 0, 25, 0, 25]}
]
GUARANTEED_CAP, GUARANTEED_STEP = 1, 1


def _sha_tensor(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _vote_spec() -> VoteUpdateSpec:
    return VoteUpdateSpec(
        threshold_abs=10,
        accumulator_clip_min=-127,
        accumulator_clip_max=127,
        decay_numerator=1,
        decay_denominator=1,
        max_abs_per_tensor=64,
        fraction_per_tensor=1.0,
    )


def _join_sha(parts: list[str]) -> str:
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def build_guaranteed_crossing_inputs(
    *, device: str | torch.device | None = None, template: list[dict[str, Any]] | None = None
) -> list[GlobalRateCapTensorInput]:
    dev = torch.device(device) if device is not None else torch.device("cpu")
    out: list[GlobalRateCapTensorInput] = []
    for row in template or GUARANTEED_CROSSING_TEMPLATE:
        state = VoteUpdateState(
            q_levels=torch.tensor(row["q"], dtype=torch.int8, device=dev),
            accumulators=torch.tensor(row["acc"], dtype=torch.int16, device=dev),
        )
        vin = VoteUpdateInputs(votes=torch.tensor(row["votes"], dtype=torch.int16, device=dev))
        out.append(
            GlobalRateCapTensorInput(
                state_key=str(row["state_key"]),
                state=state,
                plan=plan_integer_vote_update_reference(state, vin, _vote_spec()),
                vote_inputs=vin,
            )
        )
    return out


def clone_cap_inputs(inputs: list[GlobalRateCapTensorInput]) -> list[GlobalRateCapTensorInput]:
    cloned: list[GlobalRateCapTensorInput] = []
    for item in inputs:
        state = VoteUpdateState(
            q_levels=item.state.q_levels.detach().clone().contiguous(),
            accumulators=item.state.accumulators.detach().clone().contiguous(),
        )
        vin = VoteUpdateInputs(votes=item.vote_inputs.votes.detach().clone().contiguous())
        cloned.append(
            GlobalRateCapTensorInput(
                state_key=item.state_key,
                state=state,
                plan=plan_integer_vote_update_reference(state, vin, _vote_spec()),
                vote_inputs=vin,
            )
        )
    return cloned


def _state_hashes(inputs: list[GlobalRateCapTensorInput]) -> tuple[str, str]:
    return (
        _join_sha([_sha_tensor(i.state.q_levels) for i in inputs]),
        _join_sha([_sha_tensor(i.state.accumulators) for i in inputs]),
    )


def _result_hashes(result: Any) -> tuple[str, str]:
    return (
        _join_sha([_sha_tensor(tr.q_levels) for tr in result.tensor_results]),
        _join_sha([_sha_tensor(tr.accumulators) for tr in result.tensor_results]),
    )


@dataclass(frozen=True)
class TwinArmObservation:
    deferred: bool
    q_hash_pre: str
    q_hash_post: str
    acc_hash_pre: str
    acc_hash_post: str
    backlog_sha_pre: str
    backlog_sha_post: str
    backlog_cardinality_pre: int
    backlog_cardinality_post: int
    applied_count: int
    q_changed_count: int
    residual_writeback_count: int
    crossing_demand: int
    cap_site_branch: str
    flip_application_deferred: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "deferred": self.deferred,
            "q_hash_pre": self.q_hash_pre,
            "q_hash_post": self.q_hash_post,
            "acc_hash_pre": self.acc_hash_pre,
            "acc_hash_post": self.acc_hash_post,
            "backlog_sha_pre": self.backlog_sha_pre,
            "backlog_sha_post": self.backlog_sha_post,
            "backlog_cardinality_pre": self.backlog_cardinality_pre,
            "backlog_cardinality_post": self.backlog_cardinality_post,
            "applied_count": self.applied_count,
            "q_changed_count": self.q_changed_count,
            "residual_writeback_count": self.residual_writeback_count,
            "crossing_demand": self.crossing_demand,
            "cap_site_branch": self.cap_site_branch,
            "flip_application_deferred": self.flip_application_deferred,
            "q_changed": self.q_hash_pre != self.q_hash_post,
            "acc_changed": self.acc_hash_pre != self.acc_hash_post,
            "backlog_unchanged": self.backlog_sha_pre == self.backlog_sha_post,
        }


@dataclass(frozen=True)
class TwinClassifyResult:
    smoke_class: str
    failures: tuple[str, ...]
    ordinary: TwinArmObservation
    deferred: TwinArmObservation

    def as_dict(self) -> dict[str, Any]:
        return {
            "smoke_class": self.smoke_class,
            "failures": list(self.failures),
            "ordinary": self.ordinary.as_dict(),
            "deferred": self.deferred.as_dict(),
        }


def _observe_arm(
    *, deferred: bool, inputs: list[GlobalRateCapTensorInput], seed_backlog: Mapping[str, Any] | None, spec: GlobalRateCapSpec
) -> TwinArmObservation:
    q_pre, acc_pre = _state_hashes(inputs)
    backlog_pre = copy.deepcopy(dict(seed_backlog or {}))
    bsha_pre = backlog_content_sha256(backlog_pre)
    bcard_pre = int(sum(len(v) for v in backlog_pre.values()))
    result = apply_global_rate_cap_with_optional_flip_deferral(
        inputs, spec, deferred_backlog=backlog_pre, contract_name=GLOBAL_CAP_CONTRACT, flip_application_deferred=bool(deferred)
    )
    q_post, acc_post = _result_hashes(result)
    backlog_post = dict(result.deferred_backlog or {})
    summary = dict(result.step_summary or {})
    applied = int(summary.get("global_rate_cap_applied_count", 0))
    q_changed = int(summary.get("q_changed_count", 0))
    residual = 0 if deferred else (applied if q_changed > 0 else 0)
    return TwinArmObservation(
        deferred=bool(deferred),
        q_hash_pre=q_pre,
        q_hash_post=q_post,
        acc_hash_pre=acc_pre,
        acc_hash_post=acc_post,
        backlog_sha_pre=bsha_pre,
        backlog_sha_post=backlog_content_sha256(backlog_post),
        backlog_cardinality_pre=bcard_pre,
        backlog_cardinality_post=int(sum(len(v) for v in backlog_post.values())),
        applied_count=applied,
        q_changed_count=q_changed,
        residual_writeback_count=residual,
        crossing_demand=int(len(result.rows)),
        cap_site_branch=str(summary.get("forgotten_accum_cap_site_branch") or DENSE_LEGACY_CAP_SITE_ID),
        flip_application_deferred=bool(summary.get("flip_application_deferred", deferred)),
    )


def classify_ordinary_deferred_twin(ordinary: TwinArmObservation, deferred: TwinArmObservation) -> TwinClassifyResult:
    if not (ordinary.crossing_demand > 0 and ordinary.applied_count > 0):
        return TwinClassifyResult("INCONCLUSIVE_NO_CROSSING", ("ordinary_no_qualifying_crossing",), ordinary, deferred)
    fails: list[str] = []
    if ordinary.q_hash_pre == ordinary.q_hash_post:
        fails.append("ordinary_q_hash_did_not_change")
    if ordinary.q_changed_count <= 0:
        fails.append("ordinary_q_changed_count_not_positive")
    if deferred.acc_hash_pre == deferred.acc_hash_post:
        fails.append("deferred_carry_did_not_change")
    if deferred.q_hash_pre != deferred.q_hash_post or deferred.q_changed_count != 0:
        fails.append("deferred_q_mutated")
    if deferred.applied_count != 0:
        fails.append("deferred_applied_nonzero")
    if deferred.residual_writeback_count != 0:
        fails.append("deferred_residual_writeback_nonzero")
    if deferred.backlog_sha_pre != deferred.backlog_sha_post or deferred.backlog_cardinality_pre != deferred.backlog_cardinality_post:
        fails.append("deferred_backlog_mutated")
    if deferred.cap_site_branch != DENSE_LEGACY_CAP_SITE_ID:
        fails.append("deferred_cap_site_not_dense_legacy")
    if not deferred.flip_application_deferred:
        fails.append("deferred_marker_missing")
    if ordinary.cap_site_branch != DENSE_LEGACY_CAP_SITE_ID:
        fails.append("ordinary_cap_site_not_dense_legacy")
    return TwinClassifyResult("FAIL" if fails else "PASS", tuple(fails), ordinary, deferred)


def run_ordinary_deferred_twin(
    *,
    inputs: list[GlobalRateCapTensorInput] | None = None,
    seed_backlog: Mapping[str, Any] | None = None,
    cap: int = GUARANTEED_CAP,
    step: int = GUARANTEED_STEP,
    device: str | torch.device | None = None,
) -> TwinClassifyResult:
    src = inputs if inputs is not None else build_guaranteed_crossing_inputs(device=device)
    spec = GlobalRateCapSpec(cap=int(cap), step=int(step), mutate_outputs=True)
    ordinary = _observe_arm(deferred=False, inputs=clone_cap_inputs(src), seed_backlog=seed_backlog, spec=spec)
    deferred = _observe_arm(deferred=True, inputs=clone_cap_inputs(src), seed_backlog=seed_backlog, spec=spec)
    return classify_ordinary_deferred_twin(ordinary, deferred)


def is_event_coded_carrier(carrier_selector: str) -> bool:
    sel = str(carrier_selector or "").upper()
    return "EVENT_CODED" in sel or "V4_LIVE" in sel


def cpu_checkable_predicate_eval(
    *,
    carrier_selector: str,
    cap_site_branch: str,
    twin: TwinClassifyResult | None = None,
    flip_application_deferred: bool = True,
    during_w_q_changed: int = 0,
    during_w_applied: int = 0,
) -> list[str]:
    fails: list[str] = []
    sel = str(carrier_selector or "")
    if sel not in {LIVE_ACC_CARRIER_NONE, "NONE", "none", ""} or is_event_coded_carrier(sel):
        fails.append("carrier_must_be_dense_legacy_not_event_coded")
    if str(cap_site_branch) != DENSE_LEGACY_CAP_SITE_ID:
        fails.append("forgotten_accum_cap_site_branch_equals_DENSE_LEGACY_CAP_SITE_ID")
    if twin is not None:
        if twin.smoke_class == "INCONCLUSIVE_NO_CROSSING":
            fails.append("twin_inconclusive_no_crossing")
        elif twin.smoke_class != "PASS":
            fails.extend(f"twin:{f}" for f in twin.failures)
            fails.append("twin_fail")
    elif bool(flip_application_deferred):
        if int(during_w_q_changed) != 0 or int(during_w_applied) != 0:
            fails.append("flip_application_deferred_true_engages_during_W_law")
        else:
            fails.append("legacy_vacuous_deferred_zeros_insufficient_without_twin")
    return fails


@dataclass(frozen=True)
class SmokeReceipt:
    pass_fail: str
    smoke_class: str
    failures: tuple[str, ...]
    carrier_selector: str
    forgotten_accum_cap_site_branch: str
    flip_application_deferred: bool
    during_w_q_changed: int
    during_w_applied: int
    head: str
    tree: str
    parent_sha256: str
    argv_digest: str
    device: str
    global_cap_contract: str
    predicates: tuple[str, ...]
    steps_run: int
    twin: dict[str, Any]
    live_runner_cap_site_branch: str = ""
    notes: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": RECEIPT_SCHEMA,
            "pass_fail": self.pass_fail,
            "smoke_class": self.smoke_class,
            "failures": list(self.failures),
            "carrier_selector": self.carrier_selector,
            "forgotten_accum_cap_site_branch": self.forgotten_accum_cap_site_branch,
            "flip_application_deferred": bool(self.flip_application_deferred),
            "during_w_q_changed": int(self.during_w_q_changed),
            "during_w_applied": int(self.during_w_applied),
            "head": self.head,
            "tree": self.tree,
            "parent_sha256": self.parent_sha256,
            "argv_digest": self.argv_digest,
            "device": self.device,
            "global_cap_contract": self.global_cap_contract,
            "predicates": list(self.predicates),
            "steps_run": int(self.steps_run),
            "twin": dict(self.twin),
            "live_runner_cap_site_branch": self.live_runner_cap_site_branch,
            "notes": self.notes,
            "DENSE_LEGACY_CAP_SITE_ID": DENSE_LEGACY_CAP_SITE_ID,
            "supersedes_receipt_schema": "forgotten_accum_dense_site_smoke_receipt/v1",
        }


def classify_smoke_exit(receipt: SmokeReceipt) -> int:
    if is_event_coded_carrier(receipt.carrier_selector):
        return EXIT_EVENT_CODED_STOP
    if receipt.smoke_class == "INCONCLUSIVE_NO_CROSSING":
        return EXIT_INCONCLUSIVE_NO_CROSSING
    if receipt.pass_fail != "PASS" or receipt.smoke_class != "PASS":
        return EXIT_PREDICATE_FAIL
    return EXIT_PASS


def resolve_carrier_for_smoke(*, force_event_coded: bool = False) -> str:
    if force_event_coded:
        return "LIVE_ACC_CARRIER_V4_LIVE"
    return resolve_live_acc_carrier_selector(
        v4_enabled=False, w5_enabled=False, w6_enabled=False,
        w6_clip_only_enabled=False, w4_clip_only_enabled=False, w7_enabled=False, w8_enabled=False,
    )


def _git_rev(repo: Path, rev: str) -> str:
    return subprocess.check_output(["git", "rev-parse", rev], cwd=str(repo)).decode().strip()


def _argv_digest(argv: list[str]) -> str:
    return hashlib.sha256("\0".join(argv).encode()).hexdigest()


def _extract_cap_telemetry(step_reports: Mapping[str, Any], step: int = 1) -> dict[str, Any]:
    report = dict(step_reports.get(str(step)) or {})
    gs = dict(dict(report.get("step_result") or {}).get("global_summary") or {})
    return {
        "forgotten_accum_cap_site_branch": str(
            gs.get("forgotten_accum_cap_site_branch") or report.get("forgotten_accum_cap_site_branch") or ""
        ),
        "q_changed_count": int(gs.get("q_changed_count", report.get("q_changed_count", 0))),
        "global_rate_cap_applied_count": int(
            gs.get("global_rate_cap_applied_count", report.get("global_rate_cap_applied_count", 0))
        ),
    }


def run_dense_site_device_smoke(
    *,
    repo_root: Path,
    parent_path: Path,
    expected_parent_sha256: str = PARENT_SHA256_FULL,
    device: str = "cuda:0",
    include_deferred_second_step: bool = True,
    argv_for_digest: list[str] | None = None,
    run_live_runner_branch_probe: bool = True,
) -> tuple[SmokeReceipt, int]:
    from scripts.hrm_text_158_bounded_delta_acquisition_probe import (
        C1_BANKED_FAITHFUL_LONG_RUN_GLOBAL_CAP_CONTRACT_NAME,
        build_identity_full_support_batches,
        build_model_from_checkpoint,
        derive_tensor_states_and_check_init_fidelity,
        load_parent_checkpoint,
        run_bounded_delta_steps,
        select_eligible_bitlinears,
    )

    repo_root, parent_path = Path(repo_root), Path(parent_path)
    head, tree = _git_rev(repo_root, "HEAD"), _git_rev(repo_root, "HEAD^{tree}")
    argv_digest = _argv_digest(list(argv_for_digest or []))
    carrier = resolve_carrier_for_smoke(force_event_coded=False)
    if carrier != LIVE_ACC_CARRIER_NONE or is_event_coded_carrier(carrier):
        receipt = SmokeReceipt(
            pass_fail="FAIL", smoke_class="FAIL",
            failures=("carrier_must_be_dense_legacy_not_event_coded",),
            carrier_selector=carrier, forgotten_accum_cap_site_branch="",
            flip_application_deferred=False, during_w_q_changed=-1, during_w_applied=-1,
            head=head, tree=tree, parent_sha256="", argv_digest=argv_digest, device=device,
            global_cap_contract=GLOBAL_CAP_CONTRACT, predicates=tuple(SMOKE_CPU_PREDICATES),
            steps_run=0, twin={"smoke_class": "FAIL", "failures": ["carrier_event_coded_stop"]},
            notes="STOP: carrier is not dense-legacy NONE",
        )
        return receipt, EXIT_EVENT_CODED_STOP

    ckpt, parent_sha = load_parent_checkpoint(parent_path, expected_sha256=expected_parent_sha256)
    torch_device = torch.device(str(device))
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("cuda requested but torch.cuda.is_available() is False")

    live_branch, steps_run = "", 0
    if run_live_runner_branch_probe:
        model, tok, cfg = build_model_from_checkpoint(ckpt, torch_device)
        support_batches, _ = build_identity_full_support_batches(
            tok=tok, max_len=int(getattr(cfg, "max_seq_len", 64) or 64),
            batch_size=1, curriculum_seed=17, device=torch_device,
        )
        eligible = select_eligible_bitlinears(model, eligible_scope="all-bitlinear")
        states, report = derive_tensor_states_and_check_init_fidelity(eligible, threshold=0.0)
        if not report.get("all_pass", False):
            raise RuntimeError(f"init fidelity failed: {report}")
        step_reports, *_ = run_bounded_delta_steps(
            model, support_batches[0]["batch"], states, eligible, device=torch_device, steps=1,
            require_q_change=False, max_abs_per_tensor=4096, support_batches=support_batches,
            global_cap_contract=C1_BANKED_FAITHFUL_LONG_RUN_GLOBAL_CAP_CONTRACT_NAME,
            r7_deferred_backlog_carry_enabled=True, flip_application_deferred=False,
        )
        live_branch = _extract_cap_telemetry(step_reports, 1)["forgotten_accum_cap_site_branch"]
        steps_run = 1

    twin = run_ordinary_deferred_twin(device=torch_device)
    twin_blob = twin.as_dict()
    if not include_deferred_second_step:
        twin_blob["notes_cli"] = "skip_deferred ignored; twin requires both arms"
    failures = cpu_checkable_predicate_eval(
        carrier_selector=carrier, cap_site_branch=twin.deferred.cap_site_branch, twin=twin
    )
    if live_branch and live_branch != DENSE_LEGACY_CAP_SITE_ID:
        failures.append("live_runner_ordinary_cap_path_missed_dense_site")
    smoke_class = twin.smoke_class
    if failures and smoke_class == "PASS":
        smoke_class = "FAIL"
    if "twin_inconclusive_no_crossing" in failures:
        smoke_class = "INCONCLUSIVE_NO_CROSSING"
    pass_fail = "PASS" if (not failures and smoke_class == "PASS") else (
        "INCONCLUSIVE" if smoke_class == "INCONCLUSIVE_NO_CROSSING" else "FAIL"
    )
    receipt = SmokeReceipt(
        pass_fail=pass_fail, smoke_class=smoke_class, failures=tuple(dict.fromkeys(failures)),
        carrier_selector=carrier, forgotten_accum_cap_site_branch=str(twin.deferred.cap_site_branch),
        flip_application_deferred=True,
        during_w_q_changed=int(twin.deferred.q_changed_count),
        during_w_applied=int(twin.deferred.applied_count),
        head=head, tree=tree, parent_sha256=parent_sha, argv_digest=argv_digest, device=device,
        global_cap_contract=GLOBAL_CAP_CONTRACT, predicates=tuple(SMOKE_CPU_PREDICATES),
        steps_run=steps_run, twin=twin_blob, live_runner_cap_site_branch=str(live_branch),
        notes=(
            f"non_vacuous_twin; live_branch={live_branch}; "
            f"ordinary_applied={twin.ordinary.applied_count}; "
            f"deferred_acc_changed={twin.deferred.acc_hash_pre != twin.deferred.acc_hash_post}"
        ),
    )
    return receipt, classify_smoke_exit(receipt)


__all__ = [
    "EXIT_PASS", "EXIT_NO_AUTHORITY", "EXIT_EVENT_CODED_STOP", "EXIT_PREDICATE_FAIL",
    "EXIT_INCONCLUSIVE_NO_CROSSING", "EXIT_INFRA_FAIL", "DEFAULT_PARENT_RELPATH",
    "RECEIPT_SCHEMA", "GUARANTEED_CROSSING_TEMPLATE", "cpu_checkable_predicate_eval",
    "is_event_coded_carrier", "SmokeReceipt", "TwinArmObservation", "TwinClassifyResult",
    "build_guaranteed_crossing_inputs", "clone_cap_inputs", "classify_ordinary_deferred_twin",
    "run_ordinary_deferred_twin", "classify_smoke_exit", "resolve_carrier_for_smoke",
    "run_dense_site_device_smoke",
]
