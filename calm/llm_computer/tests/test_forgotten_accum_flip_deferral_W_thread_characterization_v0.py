"""Multi-step W-thread characterization for forgotten-accum flip deferral (dense-legacy).

Repo: /mnt/c/Users/gabes/projects/claw-code-hrm-text-158
Branch: feature/hrm-text-1.58
Pinned HEAD: 370025c99f0939beb56dffe17ad257fb8e7103c2

W=4 derivation: decay 1/1 + vote 25 → carry 25·k until clip ±127; first hash-stall
at k=7 (127→127); W=4 sits three steps below that stall. Fixture pin guards geometry.

Counterfactual validity (equal-margin fixture): producer MARGIN order key is
`highest_abs_new_acc_then_lower_global_flat_index` (global_rate_cap.py). With all
five crossings at equal margin, the true selection takes the two lowest
global_flat_index candidates under cap=2; reversed take-cap takes the two highest,
so both accepted and deferred index shas must move.

Exact argv (from hrm-text-158 root):
  PYTHONPATH=. python3 -m pytest \\
    calm/llm_computer/tests/test_forgotten_accum_flip_deferral_W_thread_characterization_v0.py -q
Baseline re-run (untouched known-good):
  PYTHONPATH=. python3 -m pytest \\
    calm/llm_computer/tests/test_forgotten_accum_flip_deferral_characterization_v0.py -q

Preregistered outcomes (branch consequences):
  (i) HOLDS — every during-W conjunct holds across threaded k=1..W with acc changing,
      backlog fixed at pre-loop seed_sha, chain continuous, spec invariant, row count==W,
      key presence true; at W+1 applied_count in [1, ordinary_cap], special_backlog_flush
      false, true R0 five-key vector equals RW; pure-R0 trajectory ≠ RW trajectory;
      counterfactual index-pair differs on both shas.
      → RW arm design valid AT THIS CHARACTERIZATION GEOMETRY only. Never canonical W.
  (ii) VIOLATED at step k / phase — named conjunct fails; or W+1 applied_count==0
      (degenerate release is (ii), not (i)).
      → RW arm as designed defective; fix precedes any GPU arm.
  (iii) APPARATUS — harness cannot express W>1 threading with preconditions intact,
      or counterfactual leaves either index sha equal to true.
      → STOP; no science claim about the law.

Per-property negative-path rows (pre-run forms only; results belong to the run receipt).
Scope of this block: the module's own assertions plus the one borrowed conjunct that has an authored tamper (flip_applied_count); the remaining borrowed conjuncts in the pinned reducers are not enumerated here.
  window ran exactly W steps: negative-path test at line 716 — result pending run
  key presence global_rate_cap_applied_count: negative-path test at line 727 — result pending run
  key presence threshold_residual_writeback_count: no negative-path test — decision: same shape as applied-count presence; no separate tamper authored
  q-constancy across deferred step: no negative-path test — decision: ENFORCED-UPSTREAM producer RuntimeError on q change; no assertion-side tamper authored
  harness spec_cap invariance across k: negative-path test at line 708 — result pending run
  harness spec_step tracks k: no negative-path test — decision: configuration-side claim that the loop passed step=k; no separate tamper authored
  producer contract_name emission present: no negative-path test — decision: fail-closed presence on step_summary key; absence fails without a local-constant compare
  producer global_rate_cap_cap equals ordinary cap: no negative-path test — decision: value compare against ORDINARY_CAP; absence yields None and fails rather than defaulting
  borrowed flip_applied_count==0: negative-path test at line 686 — result pending run
  borrowed accumulator carry must CHANGE: known-bad calibration at line 673 (module `_bundle_sha` over post as pre); known-good is the threaded table — result pending run
  chain continuity: known-bad negative-path test at line 697; known-good is the threaded table (same module↔telemetry pair as B14) — result pending run
  pins match on disk: no negative-path test — decision: pin mismatch is the failing world of the pin test itself
  cost (raw-byte domain): dtype-or-shape-only change with identical content is not distinguished by `_sha_tensor` / `_bundle_sha`; the surface that detects a geometry change is the six pins plus test_pins_match_on_disk, not this domain
  W+1 non-degenerate release: negative-path test at line 647 — result pending run
  true RW five-key equals R0: no negative-path test — decision: the reversed-selector counterfactual binds index-sha order sensitivity only and cannot make the five-key comparison fail
  counterfactual accepted_index_sha differs: no negative-path test — decision: failing world is either sha matching true (asserted in main witness)
  counterfactual deferred_index_sha differs: no negative-path test — decision: failing world is either sha matching true (asserted in main witness)
  pure-R0 trajectory differs from RW: no negative-path test — decision: known-bad requires a geometry where paths coincide ((iii) world); no construction authored
  ENFORCED-UPSTREAM (not a deciding row): mutate_outputs false on deferred path raises ValueError at apply.py:95 — deleted as a harness assert

ADVISOR: consulted 1786472680952-f3cb055c (mandatory defect-class audit, ADVISOR_WAIVER
unavailable; normalized class = converting a finding into a permission by weighing the
document's claims instead of returning the artifact, with self-claims-mismatch as its
surface; substantiated bounces 1786471748703-614ac99a blocker 1 and
1786472500285-8cf62081 blocker 1). CLAUDE_REDERIVATION: adopted the higher-counter
normalization test; adopted the finding-IS-the-bounce decision rule as a control-plane
change owed its own slice; adopted prereg-IN-MODULE over claude's own 90-line document
form after the interest-check; routed the plan/diff gate-structure question to the
gate-2 owner.

ADVISOR: consulted 1786473880067-563da0d2 (mandatory defect-class audit #3 this lineage,
ADVISOR_WAIVER unavailable; normalized class = apparent verification surface exceeds
effective surface, gap unmarked; substantiated bounces 1786472500285-8cf62081 and
1786473552451-3668aaf6). CLAUDE_REDERIVATION: (a) adopted — habitat-scoped renaming would
reset the counter through the back door; (b) adopted after independent verification — a
subtractive cure needs a denominator, so the 24-row property-to-check map (15 deciding
properties + 9 calibration rows, none unmapped, no property double-mapped) was performed
at gate-1; (c) not adopted as a separate action — codifying the authoring constraint is a
control-plane slice owed its own gate; (d) counterfactual silent-world refuted at the
bytes (one reversed() copy at the cf helper, producer-side true operands, fixture is pin
six), non-degeneracy serialization independently verified and it produced the
_tensor_sha256 borrow.

ADVISOR: consulted 1786475029952-483ab73a (mandatory defect-class audit, ADVISOR_WAIVER
unavailable; record correction 1786475039277-7bde107e fixes a garbled bounce id in its
first line; substantiated bounces 1786473552451-3668aaf6 and 1786474779119-ef201547).
CLAUDE_REDERIVATION: (a) adopted — a second, verifier-side generator: an instrument
authored in the same pass as the verdict it licenses binds the verdict's shape, so the
aggregate coverage verdict is composed at gate-2 and gate-1's map is disclosed working
data carrying its producing commands; (b) adopted — the class is cross-head, pre-run prose
is its habitat, and execution is the one calibration instrument that cannot overstate;
(c) adopted after independent verification at the pinned bytes — an AssertionError-only
grep misses four ENFORCED-UPSTREAM arms (forgotten_accum_flip_deferral_apply.py:75 and
:95 ValueError, :167 and :169 RuntimeError), one borrowed call carries six conjuncts, and
line-anchoring is itself spelling-keyed; (d) adopted — pre-run calibration prose is
restricted to two row forms and execution-dependent claims belong to the run receipt.

ADVISOR: consulted 1786476135664-2ce64d75 (mandatory defect-class audit, ADVISOR_WAIVER
unavailable; substantiated bounces 1786475618220-3be02c55 and 1786475882682-ac1adb44).
CLAUDE_REDERIVATION: (a) adopted, no new rule — the standing "a modified check is a new
check and its calibrations reset" already catches the defect, and the generalizing form is
that a cure touching one operand of a cross-producer compare touches the compare, so
borrowing applies to comparisons and never to halves of one; the generator is any pass that
asserts while building, now in its third sub-shape (author, verifier-map, cure). (b) adopted
— borrow-a-stronger-instrument is not a spent shape but a pairwise-constrained one. (c)
adopted after independent verification — the calibration's silent-false world is operand-
provenance collapse, so the pre operand comes from the module's own bundle and never from
the telemetry digest path or a copy of the emitted field; its fires-false world is tensor
layout, closed for this pair because contiguity is normalized inside apply.py:47. (d)
adopted — instrument strength is meaningful only relative to the comparisons it feeds, so
the compatible domain is taken and the dtype/shape gap is disclosed at its row together
with its detecting surface, the six pins. REVERSION NOTE: the `_tensor_sha256` borrow named
in the second disposition above was reverted at this version as domain-incompatible with
the telemetry producer; that mention is historical, not current.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import torch

from calm.hrm_text_158.native_full_stack.forgotten_accum_flip_deferral_apply import (
    apply_global_rate_cap_with_optional_flip_deferral,
    build_W_plus_1_release_record,
    build_during_W_telemetry,
    _sha_tensor,
)
from calm.hrm_text_158.native_full_stack.forgotten_accum_flip_deferral_contracts import (
    DENSE_LEGACY_CAP_SITE_ID,
    DuringWTelemetry,
)
from calm.hrm_text_158.native_full_stack.forgotten_accum_flip_deferral_reducers import (
    assert_W_plus_1_anti_burst,  # release_path_id arm: NON-COVERAGE (unfirable by builder default on W+1)
    assert_during_W,  # backlog-fixed + q path: ENFORCED-UPSTREAM producer raises; not sole coverage
    assert_pre_W_seed_invariant,
    backlog_content_sha256,
)
from calm.hrm_text_158.native_full_stack.global_rate_cap import (
    GlobalRateCapResult,
    GlobalRateCapRow,
    GlobalRateCapSpec,
    GlobalRateCapTensorInput,
    apply_global_rate_cap_reference,
    _row_global_index_sha,
)
from calm.hrm_text_158.native_full_stack.vote_update import (
    VoteUpdateInputs,
    VoteUpdateSpec,
    VoteUpdateState,
    plan_integer_vote_update_reference,
)

# ---------------------------------------------------------------------------
# Pins (executable) + geometry constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[3]
FIX = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "forgotten_accum_flip_deferral_characterization_v0"
)

PINNED_HEAD = "370025c99f0939beb56dffe17ad257fb8e7103c2"

PINNED_FILE_SHA256: dict[str, str] = {
    "calm/hrm_text_158/native_full_stack/forgotten_accum_flip_deferral_apply.py": (
        "fe1547d03a178f1f32d6cae26264e5acf949788877fefccb4ca1d2c5928baeb3"
    ),
    "calm/hrm_text_158/native_full_stack/forgotten_accum_flip_deferral_contracts.py": (
        "5e95fa6eb839e84c5accfb65c90318bc3b74b586b71e51aae314179f3e309378"
    ),
    "calm/hrm_text_158/native_full_stack/forgotten_accum_flip_deferral_reducers.py": (
        "15ebff85fe601f6c6c24eba1fb518098cc1e90fa70ed55b680dda3a78e244fa8"
    ),
    "calm/hrm_text_158/native_full_stack/global_rate_cap.py": (
        "46bb9db87cb426937278bcfa508b6b5c3a4bc3e20fc1a509a72ac5b51554ec70"
    ),
    "calm/llm_computer/tests/test_forgotten_accum_flip_deferral_characterization_v0.py": (
        "cf78b84124e249928e149be9b704c96ddfd813ab9a95d0059e83de40e3801594"
    ),
    "calm/llm_computer/tests/fixtures/forgotten_accum_flip_deferral_characterization_v0/"
    "cap_on_ordinary_saturated.json": (
        "2e77246df19fd76b56dfaabaf3a9640bf529133c409f94506a2e6246ab158867"
    ),
}

W = 4
ORDINARY_CAP = 2
CONTRACT_NAME = "c1_banked_faithful_long_run_global_cap"
FIXTURE_NAME = "cap_on_ordinary_saturated.json"


# ---------------------------------------------------------------------------
# Local helpers (do not import from the existing characterization test module)
# ---------------------------------------------------------------------------


def _load_fixture() -> dict[str, Any]:
    return json.loads((FIX / FIXTURE_NAME).read_text(encoding="utf-8"))


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


def _bundle_sha(tensors: list[torch.Tensor], keys: list[str]) -> str:
    """Join per-tensor digests; each digest is apply-module `_sha_tensor` (raw bytes).

    Same per-tensor function and state_key-sorted b"|" join shape as
    ``build_during_W_telemetry``'s acc_hash_post / q_hash so the module pre and
    the telemetry post live in one domain. Contiguity is normalized inside
    ``_sha_tensor`` (apply.py:47). The joins themselves remain separately
    implemented — B14 binds them.
    """
    ordered = sorted(zip(keys, tensors), key=lambda kv: kv[0])
    return hashlib.sha256(
        b"|".join(_sha_tensor(t).encode() for _, t in ordered)
    ).hexdigest()


def _inputs_from_rows(rows: list[dict[str, Any]]) -> list[GlobalRateCapTensorInput]:
    out: list[GlobalRateCapTensorInput] = []
    spec = _vote_spec()
    for row in rows:
        state = VoteUpdateState(
            q_levels=torch.tensor(row["q"], dtype=torch.int8),
            accumulators=torch.tensor(row["acc"], dtype=torch.int16),
        )
        vin = VoteUpdateInputs(votes=torch.tensor(row["votes"], dtype=torch.int16))
        plan = plan_integer_vote_update_reference(state, vin, spec)
        out.append(
            GlobalRateCapTensorInput(
                state_key=row["state_key"],
                state=state,
                plan=plan,
                vote_inputs=vin,
            )
        )
    return out


def _rebuild_inputs_from_result(
    result: GlobalRateCapResult,
    fixture_rows: list[dict[str, Any]],
) -> list[GlobalRateCapTensorInput]:
    by_key = {tr.state_key: tr for tr in result.tensor_results}
    rebuilt: list[GlobalRateCapTensorInput] = []
    spec = _vote_spec()
    for row in fixture_rows:
        tr = by_key[row["state_key"]]
        state = VoteUpdateState(
            q_levels=tr.q_levels.clone(),
            accumulators=tr.accumulators.clone(),
        )
        vin = VoteUpdateInputs(votes=torch.tensor(row["votes"], dtype=torch.int16))
        plan = plan_integer_vote_update_reference(state, vin, spec)
        rebuilt.append(
            GlobalRateCapTensorInput(
                state_key=row["state_key"],
                state=state,
                plan=plan,
                vote_inputs=vin,
            )
        )
    return rebuilt


def _clone_inputs(inputs: list[GlobalRateCapTensorInput]) -> list[GlobalRateCapTensorInput]:
    """Independent clone so R0/RW do not share storage."""
    out: list[GlobalRateCapTensorInput] = []
    for item in inputs:
        state = VoteUpdateState(
            q_levels=item.state.q_levels.clone(),
            accumulators=item.state.accumulators.clone(),
        )
        vin = VoteUpdateInputs(votes=item.vote_inputs.votes.clone())
        plan = plan_integer_vote_update_reference(state, vin, _vote_spec())
        out.append(
            GlobalRateCapTensorInput(
                state_key=item.state_key,
                state=state,
                plan=plan,
                vote_inputs=vin,
            )
        )
    return out


def _cap_spec(*, step: int, cap: int = ORDINARY_CAP) -> GlobalRateCapSpec:
    return GlobalRateCapSpec(cap=int(cap), step=int(step), mutate_outputs=True)


def _q_acc_keys(inputs_or_results: list[Any]) -> tuple[list[str], list[torch.Tensor], list[torch.Tensor]]:
    keys: list[str] = []
    qs: list[torch.Tensor] = []
    accs: list[torch.Tensor] = []
    for item in inputs_or_results:
        if isinstance(item, GlobalRateCapTensorInput):
            keys.append(item.state_key)
            qs.append(item.state.q_levels)
            accs.append(item.state.accumulators)
        else:
            keys.append(item.state_key)
            qs.append(item.q_levels)
            accs.append(item.accumulators)
    return keys, qs, accs


def five_key_vector(result: GlobalRateCapResult) -> dict[str, Any]:
    """True-path decision vector from producer-emitted fields only.

    Index shas are producer keys, not fallbacks: absence must raise.
    """
    keys, qs, accs = _q_acc_keys(result.tensor_results)
    return {
        "applied_count": int(result.step_summary["global_rate_cap_applied_count"]),
        "accepted_index_sha": str(result.step_summary["exact_shadow_accepted_sha256"]),
        "deferred_index_sha": str(result.step_summary["exact_shadow_deferred_sha256"]),
        "post_step_q_hash": _bundle_sha(qs, keys),
        "post_step_acc_hash": _bundle_sha(accs, keys),
    }


def counterfactual_index_pair(
    demand_rows: list[GlobalRateCapRow],
    *,
    cap: int,
) -> dict[str, str]:
    """Declared-UNSAFE selection: reversed candidate order, take first `cap`.

    Exactly two produced hashes — never a five-key 'decision vector'. Nothing
    executes the unsafe selector; applied_count / post-q / post-acc are not built.
    """
    reversed_rows = list(reversed(list(demand_rows)))
    unsafe_accepted = reversed_rows[: int(cap)]
    unsafe_deferred = reversed_rows[int(cap) :]
    return {
        "accepted_index_sha": _row_global_index_sha(unsafe_accepted),
        "deferred_index_sha": _row_global_index_sha(unsafe_deferred),
    }


def capture_during_W_window(
    fixture_rows: list[dict[str, Any]],
    *,
    w: int = W,
) -> tuple[list[dict[str, Any]], str, list[GlobalRateCapTensorInput], Any]:
    """Capture-only: emit row table. Does not assert.

    Threaded backlog: {} at k=1, then res_{k-1}.deferred_backlog.
    seed_sha computed ONCE pre-loop.
    Presence flags recorded without failing.
    """
    seed_backlog: dict = {}
    seed_sha = backlog_content_sha256(seed_backlog)
    backlog = seed_backlog
    inputs = _inputs_from_rows(fixture_rows)
    rows: list[dict[str, Any]] = []

    for k in range(1, int(w) + 1):
        keys, qs, accs = _q_acc_keys(inputs)
        acc_hash_pre = _bundle_sha(accs, keys)
        q_hash_pre = _bundle_sha(qs, keys)
        spec_k = _cap_spec(step=k)
        res = apply_global_rate_cap_with_optional_flip_deferral(
            inputs,
            spec_k,
            deferred_backlog=backlog,
            contract_name=CONTRACT_NAME,
            flip_application_deferred=True,
        )
        summary = res.step_summary
        # Capture-only booleans — no fail here (C4 subtractive).
        has_applied = "global_rate_cap_applied_count" in summary
        has_writeback = "threshold_residual_writeback_count" in summary
        tel = build_during_W_telemetry(acc_hash_pre=acc_hash_pre, result=res)
        rkeys, rqs, raccs = _q_acc_keys(res.tensor_results)
        rows.append(
            {
                "k": k,
                "acc_hash_pre": acc_hash_pre,
                "acc_hash_post": tel.acc_hash_post,
                "q_hash_pre": q_hash_pre,
                "q_hash_post": _bundle_sha(rqs, rkeys),
                "backlog_hash": tel.backlog_hash,
                "flip_applied_count": int(tel.flip_applied_count),
                "threshold_residual_writeback_count": int(
                    tel.threshold_residual_writeback_count
                ),
                "flip_application_deferred": bool(tel.flip_application_deferred),
                "cap_site_branch": str(tel.cap_site_branch),
                "has_global_rate_cap_applied_count": has_applied,
                "has_threshold_residual_writeback_count": has_writeback,
                "has_global_rate_cap_contract_name": (
                    "global_rate_cap_contract_name" in summary
                ),
                "producer_contract_name": summary.get("global_rate_cap_contract_name"),
                "producer_cap": summary.get("global_rate_cap_cap"),
                "spec_cap": int(spec_k.cap),
                "spec_step": int(spec_k.step),
                "tel": tel,
                "result": res,
            }
        )
        backlog = res.deferred_backlog
        inputs = _rebuild_inputs_from_result(res, fixture_rows)

    return rows, seed_sha, inputs, backlog


def assert_during_W_table(rows: list[dict[str, Any]], *, seed_sha: str, w: int = W) -> None:
    """Pure assertion layer over captured rows."""
    # Deciding property: threaded window actually ran W steps.
    assert len(rows) == int(w), (
        f"during_W row count must equal W={w}, got {len(rows)}"
    )

    for i, row in enumerate(rows):
        # Deciding property: zero-defaulted counters must be PRESENT (not merely ==0).
        assert row["has_global_rate_cap_applied_count"] is True, (
            f"key presence: global_rate_cap_applied_count missing at k={row['k']}"
        )
        assert row["has_threshold_residual_writeback_count"] is True, (
            f"key presence: threshold_residual_writeback_count missing at k={row['k']}"
        )

        # Borrowed during-W law (six conjuncts).
        assert_during_W(row["tel"], seed_backlog_sha=seed_sha)

        # Deciding property: q unchanged across deferred step (also ENFORCED-UPSTREAM).
        assert row["q_hash_pre"] == row["q_hash_post"], (
            f"q-constancy: q hash moved at k={row['k']}"
        )

        # Deciding property: harness loop did not drift its own GlobalRateCapSpec.cap across k
        # (configuration-side claim; optional producer bind of global_rate_cap_cap).
        assert row["spec_cap"] == ORDINARY_CAP, (
            f"spec invariance: harness cap drifted at k={row['k']}: {row['spec_cap']}"
        )
        assert row["producer_cap"] == ORDINARY_CAP, (
            f"producer cap emission: global_rate_cap_cap={row['producer_cap']} "
            f"at k={row['k']}"
        )
        # Deciding property: harness loop passed step=k (configuration-side threading claim).
        assert row["spec_step"] == row["k"], (
            f"spec invariance: harness step {row['spec_step']} != k {row['k']}"
        )
        # Deciding property: producer emitted contract_name (fail-closed presence; no local re-default).
        assert row["has_global_rate_cap_contract_name"] is True, (
            f"producer contract_name presence: global_rate_cap_contract_name "
            f"missing at k={row['k']}"
        )
        assert row["producer_contract_name"] == CONTRACT_NAME, (
            f"producer contract_name emission: got {row['producer_contract_name']!r} "
            f"at k={row['k']}"
        )
        # mutate_outputs: ENFORCED-UPSTREAM — deferred path raises ValueError when false
        # (apply.py:95); not a harness deciding row.

        if i == 0:
            continue
        prev = rows[i - 1]
        # Deciding property: chain continuity — pre[k] equals post[k-1].
        assert row["acc_hash_pre"] == prev["acc_hash_post"], (
            f"chain continuity: acc_hash_pre at k={row['k']} "
            f"!= acc_hash_post at k={prev['k']}"
        )


def _trajectory_fingerprint(
    fixture_rows: list[dict[str, Any]],
    *,
    steps: int,
    deferred: bool,
) -> list[tuple[str, str, int]]:
    inputs = _inputs_from_rows(fixture_rows)
    backlog: dict = {}
    fp: list[tuple[str, str, int]] = []
    for k in range(1, int(steps) + 1):
        res = apply_global_rate_cap_with_optional_flip_deferral(
            inputs,
            _cap_spec(step=k),
            deferred_backlog=backlog,
            contract_name=CONTRACT_NAME,
            flip_application_deferred=bool(deferred),
        )
        keys, qs, accs = _q_acc_keys(res.tensor_results)
        fp.append(
            (
                _bundle_sha(qs, keys),
                _bundle_sha(accs, keys),
                int(res.step_summary["global_rate_cap_applied_count"]),
            )
        )
        backlog = res.deferred_backlog
        inputs = _rebuild_inputs_from_result(res, fixture_rows)
    return fp


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_pins_match_on_disk():
    """Deciding property: pinned file bytes still match plan pins."""
    for rel, expected in PINNED_FILE_SHA256.items():
        path = REPO_ROOT / rel
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == expected, f"pin mismatch for {rel}: {digest} != {expected}"


def test_W_thread_law_holds_at_characterization_geometry():
    """Main witness: maps to outcome (i)/(ii)/(iii) via assertions below."""
    fx = _load_fixture()
    fixture_rows = fx["inputs"]

    # Pre-W seed identity (R0 vs RW construction from same fixture).
    r0_seed = _inputs_from_rows(fixture_rows)
    rw_seed = _inputs_from_rows(fixture_rows)
    assert_pre_W_seed_invariant(
        r0_acc_sha=_sha_tensor(r0_seed[0].state.accumulators),
        rw_acc_sha=_sha_tensor(rw_seed[0].state.accumulators),
        r0_backlog_sha=backlog_content_sha256({}),
        rw_backlog_sha=backlog_content_sha256({}),
        r0_backlog_cardinality=0,
        rw_backlog_cardinality=0,
    )

    rows, seed_sha, post_w_inputs, post_w_backlog = capture_during_W_window(
        fixture_rows, w=W
    )
    assert_during_W_table(rows, seed_sha=seed_sha, w=W)

    # W+1 ordinary release + true R0 reference (independent clones).
    rw_inputs = _clone_inputs(post_w_inputs)
    r0_inputs = _clone_inputs(post_w_inputs)
    spec_w1 = _cap_spec(step=W + 1)

    rw = apply_global_rate_cap_with_optional_flip_deferral(
        rw_inputs,
        spec_w1,
        deferred_backlog=post_w_backlog,
        contract_name=CONTRACT_NAME,
        flip_application_deferred=False,
    )
    r0 = apply_global_rate_cap_reference(
        r0_inputs,
        spec_w1,
        deferred_backlog=post_w_backlog,
        contract_name=CONTRACT_NAME,
    )

    pre_carry = _bundle_sha(
        [inp.state.accumulators for inp in _clone_inputs(post_w_inputs)],
        [inp.state_key for inp in post_w_inputs],
    )
    # No assignment to release_path_id anywhere — NON-COVERAGE arm on borrow.
    record = build_W_plus_1_release_record(pre_vote_carry_hash=pre_carry, result=rw)
    assert_W_plus_1_anti_burst(record)

    # Deciding property: non-degenerate release at W+1 (applied_count == 0 is outcome (ii)).
    applied = int(rw.step_summary["global_rate_cap_applied_count"])
    assert 1 <= applied <= ORDINARY_CAP, (
        f"W+1 non-degenerate release: applied_count={applied} not in [1, {ORDINARY_CAP}]"
    )

    # Deciding property: true R0 five-key vector equals RW (matched selector).
    rw_vec = five_key_vector(rw)
    r0_vec = five_key_vector(r0)
    assert rw_vec == r0_vec, (
        f"R0 differential: RW vector != R0 vector: {rw_vec} vs {r0_vec}"
    )

    # Deciding property: counterfactual index pair differs on BOTH shas.
    # Exactly {accepted_index_sha, deferred_index_sha} — not a five-key vector.
    # (iii) if either matches: apparatus; no knob substitute.
    cf = counterfactual_index_pair(r0.rows, cap=ORDINARY_CAP)
    assert cf["accepted_index_sha"] != r0_vec["accepted_index_sha"], (
        "outcome (iii) apparatus: counterfactual accepted_index_sha matches true "
        f"(cf={cf['accepted_index_sha']}, true={r0_vec['accepted_index_sha']})"
    )
    assert cf["deferred_index_sha"] != r0_vec["deferred_index_sha"], (
        "outcome (iii) apparatus: counterfactual deferred_index_sha matches true "
        f"(cf={cf['deferred_index_sha']}, true={r0_vec['deferred_index_sha']})"
    )

    # Deciding property: pure-R0 trajectory ≠ RW trajectory (non-degeneracy).
    pure_r0_fp = _trajectory_fingerprint(fixture_rows, steps=W + 1, deferred=False)
    rw_deferred_fp = _trajectory_fingerprint(fixture_rows, steps=W, deferred=True)
    rw_traj = list(rw_deferred_fp) + [
        (
            rw_vec["post_step_q_hash"],
            rw_vec["post_step_acc_hash"],
            int(rw_vec["applied_count"]),
        )
    ]
    assert pure_r0_fp != rw_traj, (
        "non-degeneracy: pure-R0 (W+1 ordinary) fingerprint equals "
        "RW (W deferred + 1 ordinary) fingerprint; "
        "equality does not discriminate outcome (ii) law defect from "
        "outcome (iii) apparatus (cap non-binding at this geometry) — "
        "read whether the ordinary cap bound under the pinned fixture"
    )


def test_capture_KB1_emits_non_deferred_path():
    """CAPTURE-side known-bad: flip_application_deferred=False applies flips."""
    fx = _load_fixture()
    inputs = _inputs_from_rows(fx["inputs"])
    res = apply_global_rate_cap_with_optional_flip_deferral(
        inputs,
        _cap_spec(step=1),
        deferred_backlog={},
        contract_name=CONTRACT_NAME,
        flip_application_deferred=False,
    )
    assert int(res.step_summary["global_rate_cap_applied_count"]) >= 1
    assert bool(res.step_summary.get("flip_application_deferred", False)) is False


def test_capture_KB_cap0_degenerate_release_at_W_plus_1():
    """CAPTURE-side known-bad: cap=0 at W+1 yields applied_count==0 (outcome ii world)."""
    fx = _load_fixture()
    rows, _seed, post_w_inputs, post_w_backlog = capture_during_W_window(
        fx["inputs"], w=W
    )
    assert len(rows) == W
    res = apply_global_rate_cap_with_optional_flip_deferral(
        _clone_inputs(post_w_inputs),
        _cap_spec(step=W + 1, cap=0),
        deferred_backlog=post_w_backlog,
        contract_name=CONTRACT_NAME,
        flip_application_deferred=False,
    )
    assert int(res.step_summary["global_rate_cap_applied_count"]) == 0


def test_calibration_module_acc_bundle_domain_matches_telemetry():
    """B14: module `_bundle_sha` over RESULT post accs, as acc_hash_pre, must fire CHANGE.

    Cross-producer bind: pre is produced by the module's OWN `_bundle_sha` over the
    result's post accumulators — never by calling the telemetry digest path and
    never by copying tel.acc_hash_post. When the two joins agree, pre==post and
    assert_during_W raises "accumulator carry must CHANGE". When the domains
    diverge (v10 configuration), pre!=post on identical tensors and this raise
    does not fire — so the calibration binds domain agreement itself.
    """
    fx = _load_fixture()
    inputs = _inputs_from_rows(fx["inputs"])
    res = apply_global_rate_cap_with_optional_flip_deferral(
        inputs,
        _cap_spec(step=1),
        deferred_backlog={},
        contract_name=CONTRACT_NAME,
        flip_application_deferred=True,
    )
    rkeys, _rqs, raccs = _q_acc_keys(res.tensor_results)
    # Module's OWN helper over the RESULT's post tensors — not tel.acc_hash_post.
    module_post_as_pre = _bundle_sha(raccs, rkeys)
    tel = build_during_W_telemetry(acc_hash_pre=module_post_as_pre, result=res)
    with pytest.raises(AssertionError, match="accumulator carry must CHANGE"):
        assert_during_W(tel, seed_backlog_sha=backlog_content_sha256({}))


def test_assertion_tamper_applied_count_fires():
    """ASSERTION-side: flip_applied_count=1 on a good row fires borrowed during_W."""
    fx = _load_fixture()
    rows, seed_sha, _, _ = capture_during_W_window(fx["inputs"], w=W)
    bad = dict(rows[1])
    tel: DuringWTelemetry = bad["tel"]
    bad_tel = DuringWTelemetry(
        **{**tel.__dict__, "flip_applied_count": 1}
    )
    with pytest.raises(AssertionError, match="flip_applied_count"):
        assert_during_W(bad_tel, seed_backlog_sha=seed_sha)


def test_assertion_tamper_chain_continuity_fires():
    """ASSERTION-side: broken chain fires chain-continuity message."""
    fx = _load_fixture()
    rows, seed_sha, _, _ = capture_during_W_window(fx["inputs"], w=W)
    tampered = [dict(r) for r in rows]
    tampered[2] = dict(tampered[2])
    tampered[2]["acc_hash_pre"] = "0" * 64
    with pytest.raises(AssertionError, match="chain continuity"):
        assert_during_W_table(tampered, seed_sha=seed_sha, w=W)


def test_assertion_tamper_spec_invariance_fires():
    """ASSERTION-side: drifted cap fires spec-invariance message."""
    fx = _load_fixture()
    rows, seed_sha, _, _ = capture_during_W_window(fx["inputs"], w=W)
    tampered = [dict(r) for r in rows]
    tampered[1] = dict(tampered[1])
    tampered[1]["spec_cap"] = ORDINARY_CAP + 1
    with pytest.raises(AssertionError, match="spec invariance: harness cap drifted"):
        assert_during_W_table(tampered, seed_sha=seed_sha, w=W)


def test_assertion_tamper_row_count_fires():
    """ASSERTION-side: dropped row fails len==W."""
    fx = _load_fixture()
    rows, seed_sha, _, _ = capture_during_W_window(fx["inputs"], w=W)
    with pytest.raises(AssertionError, match="row count must equal W"):
        assert_during_W_table(rows[:-1], seed_sha=seed_sha, w=W)


def test_assertion_tamper_key_presence_fires():
    """ASSERTION-side: missing key flag fails sole presence check."""
    fx = _load_fixture()
    rows, seed_sha, _, _ = capture_during_W_window(fx["inputs"], w=W)
    tampered = [dict(r) for r in rows]
    tampered[0] = dict(tampered[0])
    tampered[0]["has_global_rate_cap_applied_count"] = False
    with pytest.raises(AssertionError, match="key presence: global_rate_cap_applied_count"):
        assert_during_W_table(tampered, seed_sha=seed_sha, w=W)

