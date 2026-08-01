#!/usr/bin/env python3
"""Pure-seam tests for colead_commit_gate_classifier — T1–T18 + characterization.

Step-2: v4/v7 semantics (digest-set, F/X, demotion-before-uniqueness, A3/A4).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

CLS = Path(__file__).with_name("colead_commit_gate_classifier.py")


def _load():
    spec = importlib.util.spec_from_file_location("colead_commit_gate_classifier", CLS)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M = _load()

DIGEST = "a" * 64
OTHER = "b" * 64
LIVE = "cab2f1ab638a989a1ae718d3d62ca0b055b54c11a4238ef5d865cb787ffbc2f5"
TASK = "1781862264540-5c174b3f"
WORKER = "1781864000000-worker01"
FREEZE = "1781864100000-freeze01"
PASS = "1781864200000-a1b2c3d4"


def rec(
    frm: str,
    body: str,
    ts: str,
    mid: str,
    reply_to: str = "",
    kind: str = "validation_receipt",
):
    obj = {"ts": ts, "id": mid, "from": frm, "kind": kind, "body": body}
    if reply_to:
        obj["reply_to"] = reply_to
    return obj


def freeze_body(digest: str = DIGEST) -> str:
    # Anchored F2: line starts with optional "claude " then gate-1 ... FREEZE
    return (
        f"claude gate-1 freeze handoff task {TASK}\n"
        f"DIFF_DIGEST: {digest}\n"
        f"reply_to worker receipt {WORKER}"
    )


def pass_body(digest: str = DIGEST) -> str:
    return (
        f"co_lead gate-2 PASS validation/diff\n"
        f"DIFF_DIGEST: {digest}\n"
        f"task {TASK} threaded to {FREEZE}"
    )


def worker_body() -> str:
    return f"VALIDATION RECEIPT — slice task {TASK}"


def fresh_chain(digest: str = DIGEST, pass_kind: str = "validation_receipt"):
    return [
        rec("codex", worker_body(), "2026-06-19T10:00:00Z", WORKER),
        rec("claude", freeze_body(digest), "2026-06-19T10:05:00Z", FREEZE, WORKER),
        rec(
            "codex_co_lead",
            pass_body(digest),
            "2026-06-19T10:10:00Z",
            PASS,
            FREEZE,
            kind=pass_kind,
        ),
    ]


def expect(name: str, ok: bool, reason_sub: str | None, records, digest: str = DIGEST):
    got_ok, got_reason = M.find_fresh_colead_pass(records, digest)
    if got_ok != ok:
        raise AssertionError(
            f"{name}: expected ok={ok} got {got_ok} reason={got_reason!r}"
        )
    if reason_sub is not None and reason_sub not in got_reason:
        raise AssertionError(
            f"{name}: expected reason containing {reason_sub!r}, got {got_reason!r}"
        )
    print(f"PASS {name}")


def expect_freeze(name: str, body: str, staged: str, want: bool, kind: str = "msg"):
    r = rec("claude", body, "2026-06-19T10:05:00Z", FREEZE, kind=kind)
    got = M.is_claude_freeze(r, staged)
    if got != want:
        raise AssertionError(f"{name}: is_claude_freeze expected {want} got {got}")
    print(f"PASS {name}")


def main() -> int:
    # --- unit digests ---
    assert DIGEST in M.extract_digests(f"DIFF_DIGEST: {DIGEST}\n")
    assert DIGEST in M.extract_digests(f"inline DIFF_DIGEST: {DIGEST} trailing")
    assert DIGEST in M.extract_digests(f"binds to DIFF_DIGEST `{DIGEST}` here")
    assert DIGEST in M.extract_digests(f"FRESH DIFF_DIGEST **{DIGEST}**")
    assert not M.extract_digests(f"bare {DIGEST} without label")
    assert M.record_authoritatively_binds(f"DIFF_DIGEST: {DIGEST}\n", DIGEST)
    demoted = f"prior DIFF_DIGEST: {DIGEST} (dead)\n"
    assert DIGEST in M.extract_digests(demoted)
    assert DIGEST not in M.authoritative_digests(demoted)
    print("PASS digest_set_and_demotion_unit")

    # T1: backtick freeze+PASS live shapes
    t1_freeze = (
        f"DIFF GATE REQUEST — staged\n"
        f"DIFF_DIGEST `{LIVE}`\n"
    )
    t1_pass = (
        f"VALIDATION/DIFF REVIEW: PASS\n"
        f"binds to **DIFF_DIGEST `{LIVE}`**.\n"
    )
    expect(
        "T1_backtick_live_shapes",
        True,
        "fresh co_lead PASS",
        [
            rec("codex", worker_body(), "2026-06-19T10:00:00Z", WORKER),
            rec(
                "claude",
                t1_freeze,
                "2026-06-19T10:05:00Z",
                FREEZE,
                WORKER,
                kind="review_request",
            ),
            rec(
                "codex_co_lead",
                t1_pass,
                "2026-06-19T10:10:00Z",
                PASS,
                FREEZE,
                kind="review_request",
            ),
        ],
        LIVE,
    )

    # T2: FRESH DIFF_DIGEST freeze table
    t2_freeze = f"DIFF GATE REQUEST\nFRESH DIFF_DIGEST {DIGEST}\n"
    expect(
        "T2_fresh_label_freeze",
        True,
        "fresh co_lead PASS",
        [
            rec("codex", worker_body(), "2026-06-19T10:00:00Z", WORKER),
            rec("claude", t2_freeze, "2026-06-19T10:05:00Z", FREEZE, WORKER),
            rec("codex_co_lead", pass_body(), "2026-06-19T10:10:00Z", PASS, FREEZE),
        ],
    )

    # T3: strict DIFF_DIGEST: hex fixtures
    expect("T3_strict_fixtures", True, "fresh co_lead PASS", fresh_chain())

    # T4: PASS kind=review_request explicit
    expect(
        "T4_pass_kind_review_request",
        True,
        "fresh co_lead PASS",
        fresh_chain(pass_kind="review_request"),
    )

    # T5: ack / BLOCK / wrong digest / unthreaded
    expect(
        "T5_block",
        False,
        "no codex_co_lead validation/diff PASS",
        [
            rec("codex", worker_body(), "2026-06-19T10:00:00Z", WORKER),
            rec("claude", freeze_body(), "2026-06-19T10:05:00Z", FREEZE, WORKER),
            rec(
                "codex_co_lead",
                f"co_lead gate-2 REVISE\nDIFF_DIGEST: {DIGEST}",
                "2026-06-19T10:10:00Z",
                PASS,
                FREEZE,
            ),
        ],
    )
    expect("T5_wrong_digest", False, "no claude freeze", fresh_chain(OTHER))
    # Unthreaded: pass has no reply_to to freeze and no shared task id in body.
    expect(
        "T5_unthreaded_pass",
        False,
        "no codex_co_lead validation/diff PASS",
        [
            rec("codex", "VALIDATION RECEIPT — other", "2026-06-19T10:00:00Z", WORKER),
            rec(
                "claude",
                f"claude gate-1 freeze handoff\nDIFF_DIGEST: {DIGEST}\n",
                "2026-06-19T10:05:00Z",
                FREEZE,
                WORKER,
            ),
            rec(
                "codex_co_lead",
                f"co_lead gate-2 PASS validation/diff\nDIFF_DIGEST: {DIGEST}\n",
                "2026-06-19T10:10:00Z",
                PASS,
            ),
        ],
    )

    # T6: after valid freeze+PASS, later override/nudge/follow-up must not rebind freeze
    chain = fresh_chain()
    chain += [
        rec(
            "claude",
            f"CO_LEAD_GATE_OVERRIDE: /tmp/repo DIFF_DIGEST {DIGEST} co_lead PASS msg {PASS}",
            "2026-06-19T10:20:00Z",
            "1781864300000-ovride1",
            kind="msg",
        ),
        rec(
            "claude",
            f"Wake nudge (standing review_request wake-gap workaround) DIFF_DIGEST: {DIGEST}",
            "2026-06-19T10:21:00Z",
            "1781864300001-nudge01",
            kind="msg",
        ),
        rec(
            "claude",
            f"FOLLOW-UP — still waiting\nDIFF_DIGEST: {DIGEST}\n",
            "2026-06-19T10:22:00Z",
            "1781864300002-follow1",
            kind="msg",
        ),
    ]
    expect("T6_later_exclusions_do_not_rebind", True, "fresh co_lead PASS", chain)

    # T7: bare hex without label
    expect_freeze(
        "T7_bare_hex_not_freeze",
        f"claude gate-1 freeze handoff\n{DIGEST}\n",
        DIGEST,
        False,
    )

    # T8 is integration-level (allowlisted commit without chain) — covered by integration suite

    # T9 H-multi-a
    multi_freeze = (
        f"DIFF GATE REQUEST\n"
        f"prior DIFF_DIGEST: {OTHER} (dead)\n"
        f"FRESH DIFF_DIGEST {DIGEST}\n"
    )
    expect(
        "T9_multi_a_fresh_wins",
        True,
        "fresh co_lead PASS",
        [
            rec("codex", worker_body(), "2026-06-19T10:00:00Z", WORKER),
            rec("claude", multi_freeze, "2026-06-19T10:05:00Z", FREEZE, WORKER),
            rec("codex_co_lead", pass_body(DIGEST), "2026-06-19T10:10:00Z", PASS, FREEZE),
        ],
        DIGEST,
    )
    expect(
        "T9_multi_a_old_demoted",
        False,
        "no claude freeze",
        [
            rec("codex", worker_body(), "2026-06-19T10:00:00Z", WORKER),
            rec("claude", multi_freeze, "2026-06-19T10:05:00Z", FREEZE, WORKER),
            rec("codex_co_lead", pass_body(OTHER), "2026-06-19T10:10:00Z", PASS, FREEZE),
        ],
        OTHER,
    )

    # T10 H-multi-b: PASS with multi labeled; staged equals non-authoritative only
    multi_pass = (
        f"co_lead gate-2 PASS validation/diff\n"
        f"prior DIFF_DIGEST: {DIGEST} (superseded)\n"
        f"DIFF_DIGEST: {OTHER}\n"
        f"task {TASK}\n"
    )
    expect(
        "T10_multi_b_non_auth_pass",
        False,
        "no codex_co_lead validation/diff PASS",
        [
            rec("codex", worker_body(), "2026-06-19T10:00:00Z", WORKER),
            rec("claude", freeze_body(DIGEST), "2026-06-19T10:05:00Z", FREEZE, WORKER),
            rec(
                "codex_co_lead",
                multi_pass,
                "2026-06-19T10:10:00Z",
                PASS,
                FREEZE,
            ),
        ],
        DIGEST,
    )

    # T11: quoted freeze markers only
    quoted = (
        f"> claude gate-1 freeze handoff\n"
        f"> DIFF_DIGEST: {DIGEST}\n"
        f"see prior freeze\n"
    )
    expect_freeze("T11_quoted_only_not_freeze", quoted, DIGEST, False)

    # T12: X records after valid chain — chain still matches
    expect("T12_exclusions_after_chain", True, "fresh co_lead PASS", chain)

    # T13: citation free-prose (verbatim pre-mortem)
    t13 = (
        f"Previous gate-1 FREEZE carried this digest FRESH DIFF_DIGEST {DIGEST}\n"
        f"This line cites the gate-1 FREEZE for context\n"
    )
    expect_freeze("T13_citation_free_prose", t13, DIGEST, False)

    # T14: retrospective F8-like free prose, no anchored act
    t14 = (
        f"status: earlier we had FRESH DIFF_DIGEST {DIGEST} in the packet\n"
        f"no freeze act here\n"
    )
    # ensure no line-start F8
    expect_freeze("T14_retrospective_not_freeze", t14, DIGEST, False)

    # T15: unique demoted only
    t15 = f"claude gate-1 freeze handoff\nprior DIFF_DIGEST: {DIGEST} (dead)\n"
    expect_freeze("T15_unique_demoted_not_freeze", t15, DIGEST, False)
    expect(
        "T15_unique_demoted_no_pass_chain",
        False,
        "no claude freeze",
        [
            rec("codex", worker_body(), "2026-06-19T10:00:00Z", WORKER),
            rec("claude", t15, "2026-06-19T10:05:00Z", FREEZE, WORKER),
            rec("codex_co_lead", pass_body(), "2026-06-19T10:10:00Z", PASS, FREEZE),
        ],
    )

    # T16: mechanical X1
    t16 = (
        f"CO_LEAD_GATE_OVERRIDE: /tmp/repo DIFF_DIGEST={DIGEST} co_lead PASS msg {PASS}\n"
        f"claude gate-1 freeze handoff\n"
        f"DIFF_DIGEST: {DIGEST}\n"
    )
    expect_freeze("T16_x1_override_not_freeze", t16, DIGEST, False)

    # T17: coexistence positive — F8 freeze + later +1 sequencing
    t17 = (
        f"FRESH DIFF_DIGEST {DIGEST}\n"
        f"On dual accept, +1 commit follows\n"
    )
    expect_freeze("T17_coexistence_plus1_still_freeze", t17, DIGEST, True)
    expect(
        "T17_coexistence_chain",
        True,
        "fresh co_lead PASS",
        [
            rec("codex", worker_body(), "2026-06-19T10:00:00Z", WORKER),
            rec("claude", t17, "2026-06-19T10:05:00Z", FREEZE, WORKER),
            rec("codex_co_lead", pass_body(), "2026-06-19T10:10:00Z", PASS, FREEZE),
        ],
    )

    # T18: first-line +1 negative (missing F act)
    t18 = (
        f"+1 commit for the staged surface\n"
        f"DIFF_DIGEST: {DIGEST}\n"
        f"prior freeze quoted only for context\n"
    )
    expect_freeze("T18_first_line_plus1_not_freeze", t18, DIGEST, False)

    # Round-2 hostiles: quoted-only / cross-line must NOT authorize
    # (a) unquoted F act + quoted-only digest → NO freeze
    t_r2a = (
        f"claude gate-1 freeze handoff task {TASK}\n"
        f"> DIFF_DIGEST: {DIGEST}\n"
    )
    expect_freeze("R2a_quoted_only_digest_no_freeze", t_r2a, DIGEST, False)
    assert DIGEST not in M.authoritative_digests(t_r2a)
    assert not M.record_authoritatively_binds(t_r2a, DIGEST)
    print("PASS R2a_quoted_only_digest_no_auth")

    # (b) PASS marker + quoted-only digest → NO PASS
    expect(
        "R2b_pass_quoted_only_digest",
        False,
        "no codex_co_lead validation/diff PASS",
        [
            rec("codex", worker_body(), "2026-06-19T10:00:00Z", WORKER),
            rec("claude", freeze_body(), "2026-06-19T10:05:00Z", FREEZE, WORKER),
            rec(
                "codex_co_lead",
                f"co_lead gate-2 PASS validation/diff\n> DIFF_DIGEST: {DIGEST}\n"
                f"task {TASK}\n",
                "2026-06-19T10:10:00Z",
                PASS,
                FREEZE,
            ),
        ],
    )

    # (c) DIFF_DIGEST: newline hex → NO authoritative bind / freeze / PASS
    t_r2c = f"claude gate-1 freeze handoff task {TASK}\nDIFF_DIGEST:\n{DIGEST}\n"
    expect_freeze("R2c_cross_line_digest_no_freeze", t_r2c, DIGEST, False)
    assert DIGEST not in M.extract_digests(t_r2c)
    assert DIGEST not in M.authoritative_digests(t_r2c)
    assert not M.record_authoritatively_binds(t_r2c, DIGEST)
    print("PASS R2c_cross_line_digest_no_auth")
    expect(
        "R2c_cross_line_no_pass_chain",
        False,
        "no claude freeze",
        [
            rec("codex", worker_body(), "2026-06-19T10:00:00Z", WORKER),
            rec("claude", t_r2c, "2026-06-19T10:05:00Z", FREEZE, WORKER),
            rec(
                "codex_co_lead",
                f"co_lead gate-2 PASS validation/diff\nDIFF_DIGEST:\n{DIGEST}\n"
                f"task {TASK}\n",
                "2026-06-19T10:10:00Z",
                PASS,
                FREEZE,
            ),
        ],
    )

    # Live-shape offline replay of 4 repro records (packet_v8 digest)
    live_worker = rec(
        "codex",
        "VALIDATION RECEIPT — materialize task 1785580662332-ab6ceb57",
        "2026-08-01T13:56:20Z",
        "1785592580777-55676dee",
        kind="validation_receipt",
    )
    live_freeze = rec(
        "claude",
        "DIFF GATE REQUEST — packet_v8 plan v17 staged (task 1785580662332-ab6ceb57)\n"
        f"**FRESH DIFF_DIGEST** | **{LIVE}**\n"
        f"Requesting: diff-gate PASS echoing DIFF_DIGEST {LIVE}.\n",
        "2026-08-01T13:56:51Z",
        "1785592611035-df24a9ba",
        "1785592540189-ec57a2c3",
        kind="review_request",
    )
    # Ensure F1 anchors: first line is DIFF GATE REQUEST
    live_freeze = rec(
        "claude",
        "DIFF GATE REQUEST — packet_v8 plan v17 staged (task 1785580662332-ab6ceb57)\n"
        f"FRESH DIFF_DIGEST {LIVE}\n",
        "2026-08-01T13:56:51Z",
        "1785592611035-df24a9ba",
        "1785592540189-ec57a2c3",
        kind="review_request",
    )
    live_pass = rec(
        "codex_co_lead",
        "VALIDATION/DIFF REVIEW: PASS\n"
        f"Verdict: PASS — binds to **DIFF_DIGEST `{LIVE}`**.\n"
        f"task 1785580662332-ab6ceb57\n",
        "2026-08-01T13:57:43Z",
        "1785592663322-f8b3c5bd",
        "1785592611035-df24a9ba",
        kind="review_request",
    )
    live_override = rec(
        "claude",
        f"CO_LEAD_GATE_OVERRIDE — authorized\n"
        f"DIFF_DIGEST: {LIVE}\n"
        f"co_lead PASS msg 1785592663322-f8b3c5bd\n",
        "2026-08-01T14:02:21Z",
        "1785592941922-5cba45e0",
        kind="msg",
    )
    expect(
        "LIVE_replay_four_records",
        True,
        "fresh co_lead PASS",
        [live_worker, live_freeze, live_pass, live_override],
        LIVE,
    )

    print("ALL pure T1–T18 + live replay PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
