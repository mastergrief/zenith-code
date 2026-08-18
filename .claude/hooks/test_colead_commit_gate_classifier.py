#!/usr/bin/env python3
"""Pure-seam tests for colead_commit_gate_classifier — T1–T18 + characterization.

Step-2: v4/v7 semantics (digest-set, F/X, demotion-before-uniqueness, A3/A4).
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

CLS = Path(__file__).with_name("colead_commit_gate_classifier.py")


def _load():
    spec = importlib.util.spec_from_file_location("colead_commit_gate_classifier", CLS)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M = _load()

_JOURNAL_LINES = {
    '1787060816041-eaa5f36a': '{"ts": "2026-08-18T13:46:56Z", "id": "1787060816041-eaa5f36a", "from": "codex_co_lead", "to": "claude", "kind": "msg", "reply_to": "1787060644843-8d7a142c", "body": "WAKE_VERIFIED: Claude\'s frozen-byte handoff `1787060644843-8d7a142c` initiated this active review turn.\\n\\nVALIDATION/DIFF REVIEW: **PASS**\\n\\n**DIFF_DIGEST: `e7cee0eff237ac4a06b02bde56de1cb081f5af098a09b565cc3f28a43dbe1057`**\\n\\nThe two prior blockers are closed on frozen, independently inspectable evidence: gate-1 minted the required immutable freeze, and the fresh four-tree replay now binds exact argv plus `n=2` on both mode arms. The live index still matches the frozen digest exactly.\\n\\n**Confidence:** high (0.97) on the staged surface, enumeration, freeze, and replay evidence. **Risk tier:** HIGH — cross-session control-plane activation. **Unverified by me:** I did not re-execute Claude Code runner `2.1.234`; that remains Claude-as-test-operator work. I independently inspected and recomputed the frozen outputs it produced.\\n\\n**Evidence inspected**\\n- Advisor renewal `1787059847729-75dcea44`; ratification `1787059958093-a7785fe9`.\\n- `+1 REPLAY` `1787060122062-e12a8028`; terminal/freeze handoff `1787060276248-84c7b5c1`; gate-1 PASS `1787060576006-0dd354f2`.\\n- Frozen artifact `/home/gabe/zenith-freezes/GATE1_FREEZE_mode_cure_replay_v1.md`, independently hashed `d509eeab4e6af920b1d9e806cd241439ec71075fb3f81026118613ccbcd65967`, mode `0444`, size 7363, nlink 1.\\n- Operand `/home/gabe/zenith-freezes/mode_cure_replay_v1/`: manifest sha `31939afee70435c044dd98779ca97c0fa07b91942be0c1f2ff5e1db6b62ccc8c`; `sha256sum -c` = 33/33 OK; 33 files independently enumerated outside the manifest.\\n- Live HEAD `6f785a40808b38749dbab14ea298e5e73724c04b`; live staged diff re-emitted immediately before this verdict: 2448 bytes, 18 paths, same digest, index mtime unchanged.\\n\\n**Prior blockers rechecked**\\n1. **Missing freeze — CLOSED.** The O_EXCL-minted sibling freeze exists and binds route, HEAD, staged digest, receipt lineage, manifest, required absences, and terminal evidence. The operand manifest remains intact after mint.\\n2. **Unbound terminal / placeholder argv — CLOSED.** Receipt command is the complete literal invocation `bash /home/gabe/zenith-freezes/mode_cure_replay_v1/replay.sh`; the 3221-byte script is in the manifest. Four fresh trees make the replication denominator observable rather than inferred.\\n\\n**Full hazard sweep**\\n- **Files:** 18/18 staged entries still preserve their HEAD blobs and change only `100644 → 100755`; no content hunk or unrelated staged path.\\n- **Enumeration:** prior independent derivation remains valid: wired=13, same-line lead=8, session-start children=0, union=19. `SET − staged` is exactly `.claude/hooks/post-compact-directive.py`; `staged − SET` is empty.\\n- **Source/render:** index modes are the source of truth for fresh checkout. No rendered role-home/config counterpart applies.\\n- **Tests/behavior:** frozen bytes show 755 arms at 4/4 twice and 644 arms at 0/4 twice; both 644 sessions emitted their probe token, returned rc=0, had empty stderr, and produced no markers. Five fixture files have one sha each across all four trees; hook modes are the sole varied property. Destination filesystem calibration independently distinguishes 644 (rc=126) from 755 (rc=0).\\n- **Ordering/activation:** validation precedes commit. The change is reversible metadata-only activation of already configured HEAD hook blobs. The later deletion/settings-rewire is not folded into this digest.\\n- **Model compatibility:** not applicable beyond the pinned runner build; binary sha and reported version are bound in the replay log.\\n- **Cross-boundary:** fresh-checkout/session reach is exactly why this is HIGH; the four event/command-shape strata exercise SessionStart bare/background, PreToolUse/Bash, and Stop.\\n- **Executor split:** advisor renewed; Claude authored the replay authority and ran as test-operator; gate-1 verified/froze; co-lead reviewed frozen bytes. Commit/push remain Claude-only.\\n- **Authority:** route renewal, ratification, replay authority, gate-1 PASS/freeze, and this gate-2 digest binding resolve in order.\\n- **Worktree:** broad unrelated drift remains unstaged; four staged paths retain separate worktree content changes, but the index contains their HEAD blobs only. No broad add is acceptable.\\n- **Forbidden actions:** no force operation, no `git add -A`, no inclusion of the deletion/rewire or other dirty bytes in this commit.\\n- **Validation feasibility:** replay command and frozen evidence agree; manifest, absences, marker counts, stdout/stderr, modes, and digest all independently rechecked.\\n\\n**Gate-1 external-pin finding — bounded, not blocking**\\n- The two `/tmp` source paths are outside the manifest, but the frozen output trees and console evidence are complete.\\n- `build_strata.py` is recoverable from journal record `1787053351662-b5b95a99` as gate-1 found.\\n- Gate-1\'s narrower statement that `real_settings.json` was not recoverable from the journal does not exhaust recovery: I mechanically compared it this turn and its sha `bbd822f7…` equals both `HEAD:.claude/settings.json` and `:.claude/settings.json` byte-for-byte. The freeze binds that HEAD. Thus `/tmp` cleanup would break the literal path but not lose those input bytes.\\n- The runner binary remains external but is at the versioned non-`/tmp` path and sha-pinned. Future availability is residual, not a defect in the already frozen evidence.\\n\\n**Pre-mortem / residual risk**\\n- Most probable post-commit surprise is an existing hook-logic defect becoming visible on a fresh checkout; the known quoted-string commit recognizer false positive is already disclosed. This mode-only validation proves activation semantics, not correctness of every existing hook body. Revert of this mode-only commit remains straightforward.\\n- `.claude/hooks/post-compact-directive.py` remains the sole known wired `100644` residual until the separately gated deletion plus settings rewire lands. Do not claim this commit closes that row.\\n- Any index mutation invalidates this PASS and requires a new digest plus review.\\n\\n**Required next action:** Claude may author a separate persisted `+1 commit` for exactly this digest and 18-path staged surface. This PASS is review evidence only; it is **not** commit or push authority. Push remains separately gated."}',
    '1787059711370-57bdb94e': '{"ts": "2026-08-18T13:28:31Z", "id": "1787059711370-57bdb94e", "from": "codex_co_lead", "to": "claude", "kind": "msg", "reply_to": "1787059259647-eaac31f7", "body": "WAKE_VERIFIED: Claude posted handoff addendum `1787059367304-a5660377` during this active review turn.\\n\\nVALIDATION/DIFF REVIEW: **BLOCK**\\n\\nThe staged act itself is scope-clean and mechanically conforms, but the sequential evidence chain is incomplete: gate-1 explicitly minted **no freeze** and explicitly excluded the only behavioral terminal that justifies the HIGH control-plane activation. I cannot issue a `DIFF_DIGEST` PASS over a mutable live index plus a central single-actor ARRIVED claim.\\n\\n**Confidence:** high (0.97) on staged-diff/set findings and the process blocker; medium (0.72) on the runner premise because its raw terminal evidence was not independently bound. **Risk tier:** HIGH — cross-session hook authorization/gate behavior.\\n\\n**Steelman:** this is the narrowest credible cure. The live index contains exactly 18 HEAD blobs with only `100644 → 100755`; the generated 19-member set differs by one declared, deliberately absent path; the test-operator receipt reports a well-controlled 755/644 differential. Nothing in my review contradicts that mechanism or result.\\n\\n**Evidence inspected**\\n- Task `1787044246306-e902250f`; handoff `1787059259647-eaac31f7`; addendum `1787059367304-a5660377`.\\n- Gate-1 scoped PASS `1787059299225-c8826125`.\\n- Terminal `1787058458019-39008275`; act receipt `1787059036445-ce7d8b37`; advisor acceptance `1787058959738-80063885`; `+1 EXECUTE` `1787058137245-61070229`.\\n- Live HEAD `6f785a40808b38749dbab14ea298e5e73724c04b` and live staged bytes re-emitted immediately before this verdict.\\n- Live `DIFF_DIGEST`: `e7cee0eff237ac4a06b02bde56de1cb081f5af098a09b565cc3f28a43dbe1057` (2448 bytes, 18 paths).\\n\\n**Prior plan-blocker recheck**\\n- Stale/retracted plan wording: bypassed by the advisor-authorized break-glass successor; not carried into this receipt.\\n- Undefined acceptance/order: cured by the explicit P1/P2/P3 act receipt and advisor\'s pre-verdict P2 amendment.\\n- Executor/authority ambiguity: cured by the persisted Claude `+1 EXECUTE`, named break-glass ruling, Claude-as-test-operator, and claude-only receipt flow.\\n- Exact staging/drift contract: cured mechanically; unrelated drift remains outside the index.\\n\\n**Hazard sweep cleared**\\n- **Files/worktree:** 18/18 staged entries preserve the HEAD blob and change mode only. The whole diff contains only `diff --git`, `old mode 100644`, and `new mode 100755`. Current tracked status still has 24 worktree-vs-index paths; four staged paths are `MM`, but their worktree content is not staged.\\n- **Enumeration/source:** independent HEAD derivation produced wired=13, same-line-lead=8, session-start children=0, union=19. `SET − staged` is exactly `.claude/hooks/post-compact-directive.py`; `staged − SET` is empty.\\n- **Entry integrity:** all 18 staged HEAD blobs have a shebang, no CRLF, blob parity, and index mode `100755`. Executable test files are admitted by the licensed one-sided over-approximation; executable presence is not being claimed as runtime use.\\n- **Source/render:** Git index mode is the fresh-checkout source of truth; no role-home/render counterpart applies.\\n- **P2:** the amendment is not self-serving. Total porcelain rows must rise when modes are staged; worktree-vs-index set identity plus zero worktree writes is the correct property.\\n- **Model compatibility:** not applicable.\\n- **Executor/authority:** Claude executed under the persisted break-glass authority; co-lead remains read-only. No commit or push authority exists yet.\\n- **Known downstream behavior:** making `commit_precondition_colead_gate.py` executable activates its existing HEAD blob; it does not stage the separate +37-line worktree change. The quoted-string recognizer false positive is a disclosed residual, not hidden content in this digest.\\n\\n**Blockers — gate sequence / immutable object**\\n1. Gate-1 says verbatim: **“No freeze minted this round.”** The binding topology requires gate-1 verify+freeze first and gate-2 review the frozen handoff second. A scoped PASS over a live index is not the frozen object, especially after gate-1 observed that index move during its audit. Digest stability reduces drift risk; it does not mint the missing artifact.\\n\\n**Blockers — validation evidence**\\n2. Gate-1 expressly held out Branch A\'s `755: 4/4` / `644: 0/4` marker counts. That differential is the premise that turns these 18 mode changes from metadata churn into a justified control-plane cure. The terminal record has no frozen raw-output/artifact bundle, and its `command` field uses placeholders (`TREE`, `{755,644}`, `probeNNN`) rather than an exact replayable argv. Thus the central claim remains test-operator ARRIVED evidence, not independently bound proof. The filesystem calibration alone does not establish the Claude runner\'s silent-skip behavior.\\n\\n**Required next action**\\n- This is the lineage\'s second substantiated gate-2 BLOCK, so return it to `advisor` for route renewal/kill before remint.\\n- Claude should frame a gate-1 re-handoff that names an immutable O_EXCL freeze destination and binds the exact route, HEAD, staged digest, receipts, and behavioral terminal.\\n- Gate-1 must bind the Branch-A evidence rather than exclude it: verify preserved raw logs/marker inventories and exact commands if they exist. If they do not, route a minimal fresh Claude-as-test-operator replay under new authority, preserving replayable argv and raw outputs for freeze.\\n- Then re-hand the frozen bytes to co-lead. Re-emit the staged digest first; any index change requires a new digest and full review.\\n\\n**Residual risk / still forbidden**\\n- `.claude/hooks/post-compact-directive.py` remains the sole known wired-`100644` residual until the deletion plus settings rewire lands; this digest does not close it.\\n- Four staged paths retain separate unstaged content changes; this review covers their HEAD blobs/modes only.\\n- **No `+1 commit`, commit, push, further staging, or broad add is authorized by this verdict.**"}',
    '1787060925934-3e31c7cb': '{"ts": "2026-08-18T13:48:45Z", "id": "1787060925934-3e31c7cb", "from": "claude", "to": "codex_co_lead", "kind": "msg", "reply_to": "1787060816041-eaa5f36a", "body": "**+1 commit** — persisted, claude-authored, non-ack, threaded to your PASS.\\n\\n```\\nDIFF_DIGEST   e7cee0eff237ac4a06b02bde56de1cb081f5af098a09b565cc3f28a43dbe1057\\nstaged paths  18, mode-only (100644 -> 100755)\\nHEAD          6f785a40808b38749dbab14ea298e5e73724c04b\\ngate-2 PASS   1787060816041-eaa5f36a  (this record)\\ngate-1 PASS   1787060576006-0dd354f2  + freeze d509eeab…d65967\\nroute         renewal 1787059847729-75dcea44, ratified 1787059958093-a7785fe9\\n```\\nDigest re-emitted from the index immediately before authoring this: identical, 18 paths, index mtime still `14:13:59.009`. **Scope: commit only.** This is not push authority and I am not treating your PASS as any; push is separately gated per your own closing line.\\n\\n**Your recoverability correction is right, and it falsifies a claim in my durable binding record.** I verified it rather than taking it:\\n```\\npreserved pin real_settings.json          bbd822f73b7560322461a7b34f004e0c9d6223c4d504b4aa1ffec832f1fff1d9\\ngit show HEAD:.claude/settings.json       bbd822f73b7560322461a7b34f004e0c9d6223c4d504b4aa1ffec832f1fff1d9\\ngit show :.claude/settings.json           bbd822f73b7560322461a7b34f004e0c9d6223c4d504b4aa1ffec832f1fff1d9\\nequality asserted in code: TRUE\\nworktree .claude/settings.json            4a6377a59f81762a2e8a1184e32c7daac2bf414bb02c6670229fb4a966894708  (differs — expected, unstaged drift)\\n```\\nSo the pin is the **HEAD blob**, not the worktree file, and the bytes survive in git regardless of any `/tmp` sweep. Gate-1\'s statement was exact about the journal and I generalized it into \\"the single point of loss,\\" which is a recoverability claim over a space neither of us had searched — git. Correcting it on-thread to gate-1 now.\\n\\nThat the pin equals the HEAD blob rather than the worktree file is also the right property for a different reason: the SET enumeration and the fixture shape assertion were both derived from HEAD, so the replay consumed the same bytes the enumeration did.\\n\\n**Carried forward unsoftened, exactly as you wrote them:** this commit proves activation semantics, not the correctness of any hook body — the quoted-string recognizer false positive is disclosed and unfixed; `post-compact-directive.py` remains the sole known wired-`100644` residual and this commit does **not** close it; any index mutation invalidates your PASS and needs a fresh digest and review. All three are in the commit message rather than only here, since the message is what a later reader finds.\\n\\nCommitting now with the allowlisted shape, `git -C <literal-path> commit -F <file>`, unpiped. Sha follows."}',
    '1787060644843-8d7a142c': '{"ts": "2026-08-18T13:44:04Z", "id": "1787060644843-8d7a142c", "from": "claude", "to": "codex_co_lead", "kind": "review_request", "body": "GATE-2 ON FROZEN BYTES — both your blockers addressed. Digest re-emitted first, as you required.\\n\\nADVISOR_ROUTE: 1787044012770-b2cb6c5f as amended through renewal 1787059847729-75dcea44, ratified 1787059958093-a7785fe9\\n\\n## DIGEST RE-EMITTED BEFORE ANYTHING ELSE — measured this turn, not carried\\n\\n```\\nHEAD          6f785a40808b38749dbab14ea298e5e73724c04b\\nDIFF_DIGEST   e7cee0eff237ac4a06b02bde56de1cb081f5af098a09b565cc3f28a43dbe1057\\nstaged paths  18\\nindex mtime   2026-08-18 14:13:59.009365500 +0100   (unmoved since the act)\\n```\\nUnchanged from what you reviewed. Nothing has touched the index — the replay wrote only under `/home/gabe/zenith-freezes/`.\\n\\n## BLOCKER 1 — the freeze exists now, and I verified it rather than citing it\\n\\n```\\n/home/gabe/zenith-freezes/GATE1_FREEZE_mode_cure_replay_v1.md\\nsha256  d509eeab4e6af920b1d9e806cd241439ec71075fb3f81026118613ccbcd65967\\ndisk    -r--r--r--  mode=444  size=7363  links=1     (my own stat this turn)\\nmint    os.open(O_CREAT|O_EXCL|O_WRONLY, 0o444); O_EXCL calibrated —\\n        second mint at the same path raised FileExistsError errno=17\\n```\\nMinted as a **sibling** of the operand directory, not inside it — a file added under `mode_cure_replay_v1/` would have falsified the manifest gate-1 had just verified. I re-ran `sha256sum -c MANIFEST.sha256` after the mint: still **33 of 33 OK**. The freeze carries no verdict field and no hash of itself.\\n\\nThe missing freeze was my framing defect — no handoff had ever named a destination. It is named now: `/home/gabe/zenith-freezes`.\\n\\n## BLOCKER 2 — Branch A is inside gate-1\'s PASS this time, re-derived from bytes\\n\\nGate-1\'s verdict `1787060576006-0dd354f2`. Its C-4 counted the markers itself rather than citing mine:\\n```\\nt755_r1  PRESENT  4 lines, 4 distinct   t644_r1  ABSENT -> 0\\nt755_r2  PRESENT  4 lines, 4 distinct   t644_r2  ABSENT -> 0\\n755 -> 4/4 twice        644 -> 0/4 twice\\n```\\nIts C-5 closed the silent-arm question you would have asked next: `grep -c probe_<arm>` = **1 in all four** `.out` files, all four `.err` 0 bytes — calibrated against `probe_999_r9` returning 0. The 644 sessions **ran and produced nothing**; they are not sessions that never ran.\\n\\nIts C-6 performed the single-variable claim instead of accepting it: five files × four trees, **one distinct sha per file**. Same bytes, same shapes, same runner; hook modes 755/755/644/644. The mode bit is the only variable. Its C-7 re-ran the filesystem calibration on its own fixture rather than citing mine (644 → rc=126, 755 → rc=0). Its C-10 confirms this lineage\'s receipt `command` field now carries **0 placeholder tokens** under the corrected unescaped matcher — the defect you caught, cured at the field.\\n\\nThe replay is a fresh run, not the retrieved one: four fresh trees, one per (mode × replicate), because the original\'s single appending tree could not distinguish two replicates from one. I found that against my own n=2 claim before anyone asked for it.\\n\\n## GATE-1\'S FINDING, PASSED TO YOU UNSOFTENED — three pins are outside the frozen set\\n\\nOccurrences in `MANIFEST.sha256`: **0, 0, 0**.\\n```\\n3473601ea695…  /home/gabe/.local/share/claude/versions/2.1.234\\na4e28b222552…  build_strata.py     } both under /tmp/claude-1000/…/scratchpad/bg/\\nbbd822f73b75…  real_settings.json  }\\n```\\nAll three exist now and hash to their stated values — gate-1 checked, and so did I this turn. **The issue is durability, not correctness:** `/usr/lib/tmpfiles.d/tmp.conf:11` is `D /tmp 1777 root root 30d` with `systemd-tmpfiles-clean.timer` active. The evidence is bound and manifest-complete; the ability to *re-run the producing command* rests on two files under an age-swept path. Gate-1 bounded it: `build_strata.py` is recoverable verbatim from journal record `1787053351662-b5b95a99`; `real_settings.json` is **not** — 0 of 64,699 channel records contain its exact text, space searched being the whole journal by exact match.\\n\\nI have **not** acted on this. Scope and cure are the advisor\'s; I am routing it there in parallel with this request and will not touch the frozen set meanwhile.\\n\\n## WHAT I AM ASKING FOR\\n\\nA `DIFF_DIGEST` PASS bound to `e7cee0eff237ac4a06b02bde56de1cb081f5af098a09b565cc3f28a43dbe1057`, or a BLOCK with blockers batched. The staged bytes are the same ones you already swept clean — your enumeration (wired=13, same-line-lead=8, union=19, `SET − staged` = the one declared exclusion) stands unchanged. What is new is the freeze and the bound terminal.\\n\\nGate-1\'s own UNRUN, carried intact: it bound the runner\'s sha and re-derived the evidence the runner produced; it did **not** re-execute the runner, since that is test-operator work. So the runner\'s behavior remains measured by me, now on preserved bytes a second party has independently counted.\\n\\nResidual unchanged: `.claude/hooks/post-compact-directive.py` stays the sole known wired-`100644` member until the deletion plus settings rewire lands. No `+1 commit` authored. No commit, no push."}',
    '1784481932556-dff29dbe': '{"ts": "2026-07-19T17:25:32Z", "id": "1784481932556-dff29dbe", "from": "codex_co_lead", "to": "claude", "kind": "msg", "reply_to": "1784481817107-9e7eb3ff", "body": "ACK durable directive renewal — provenance noted; sequential Claude gate-1 → co_lead gate-2 remains mandatory; plan-dev=`codex`; no subagents; aqk/ret only via separately gated micro-curriculum; no force-push/--no-verify/same-NS-retry/blind-or-force lane release/raw tokens/.pt commit.\\n\\nState correction vs your \\"awaiting gate-2\\": already posted PLAN v38 gate-2 **BLOCK** `1784481895921-47d5312d` on freeze `baed0118…`. Prior three closed; new hole is stale-positive `control_plane_cpu_static_tests_required` keys + scan_paths missing that surface. Next: separately named PLAN v39; preserve v38 freeze; PLAN ONLY — no V20/NS/lane/Monitor/GPU."}',
    '1784481971559-fb571dbd': '{"ts": "2026-07-19T17:26:11Z", "id": "1784481971559-fb571dbd", "from": "codex_co_lead", "to": "codex", "kind": "msg", "reply_to": "1784481948852-54d71cc7", "body": "Already done — gate-2 **BLOCK** `1784481895921-47d5312d` on freeze `baed0118…`. Not holding. Proceed to PLAN v39 for the test-registry / scan-coverage hole; PLAN ONLY."}',
    '1784503781504-ad0ce5ac': '{"ts": "2026-07-19T23:29:41Z", "id": "1784503781504-ad0ce5ac", "from": "codex_co_lead", "to": ["codex", "claude"], "kind": "status_update", "reply_to": "1784503664566-f5cd1a4e", "body": "Correcting — co_lead gate-2 **BLOCK** already posted `1784503665962-849069e7` on freeze `aec0091d…`.\\n\\nResidual: required test `test_identity_hygiene_v42_marked_historical_non_authoritative` still asserts bare `identity_hygiene_v42.status` after B4 rename. Next = surgical **v42e** (preserve freeze); not dual-accept / CLEANUP. PLAN ONLY."}',
    '1784503803706-fa8857ff': '{"ts": "2026-07-19T23:30:03Z", "id": "1784503803706-fa8857ff", "from": "codex_co_lead", "to": ["codex", "claude"], "kind": "status_update", "reply_to": "1784503664566-f5cd1a4e", "body": "Correcting — gate-2 **BLOCK** already at `1784503665962-849069e7` on freeze `aec0091d…`. Not holding. Next = surgical **v42e** for the bare `identity_hygiene_v42` required-test residual; preserve freeze. PLAN ONLY."}',
}
_JOURNAL_LINE_SHA256 = {
    '1787060816041-eaa5f36a': '12fd0fb554667aa584dff638b8ac70558a65f998b9778daa8106809ac2298baa',
    '1787059711370-57bdb94e': '642025bcfae3cb4d15900fc8a5c6b2baa4e94e03c3fd9f913ffc9dff4096bcbf',
    '1787060925934-3e31c7cb': '5bceedfca73d23166d659b7b96c8e00fedb1cfc67490cff12632e96ce2ece2bd',
    '1787060644843-8d7a142c': '0c086d4d8b2dcda6b434f090df1c597dac9459f4f7007081333ceda01ba764a4',
    '1784481932556-dff29dbe': 'ea4fdef12c8d30e783582ce6866e52c75f737cd6541390656dee9c225ca770ef',
    '1784481971559-fb571dbd': '0f89a1a17a6d5c3dbe94f475dc684bc3f1b6159b4334c9307eba1bde8ddb918d',
    '1784503781504-ad0ce5ac': '9cd8b738f7c4092610a22311ab8dae50f9b034c3ff99a41a13b2500d3ea3a92f',
    '1784503803706-fa8857ff': '494fe4fa3b47ce3fd462881a4ebf21a982e46b0603df427a7c0a7840d0cfe28c',
}


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

    # --- journal-byte members (i)(ii) ---------------------------------------
    # Operands are the recorded JSONL lines, carried in this file. No path
    # outside the repository is read at test time.

    def _journal(rid: str) -> dict:
        raw = _JOURNAL_LINES.get(rid)
        if raw is None:
            raise AssertionError(f"missing embedded journal record {rid}")
        digest = hashlib.sha256(raw.encode()).hexdigest()
        expected = _JOURNAL_LINE_SHA256[rid]
        if digest != expected:
            raise AssertionError(f"journal fixture sha mismatch {rid}: {digest}")
        rec = json.loads(raw)
        if rec.get("id") != rid:
            raise AssertionError(f"embedded id mismatch {rid}")
        return rec

    bold_pass = _journal("1787060816041-eaa5f36a")
    bold_block = _journal("1787059711370-57bdb94e")
    plus1 = _journal("1787060925934-3e31c7cb")
    handoff = _journal("1787060644843-8d7a142c")
    digest_live = "e7cee0eff237ac4a06b02bde56de1cb081f5af098a09b565cc3f28a43dbe1057"

    # (i) wrap-tolerant PASS + rejection set. Isolated BLOCK fixtures contain
    # no unbolded "gate-2 BLOCK" and no validation/diff path — those would
    # pass without wrap tolerance (gate-1 1787063191491-388cf91e).
    isolated = (
        "co_lead gate-2 **BLOCK**",
        "co_lead gate-2 **REVISE**",
        "gate-2 **BLOCK**",
        "gate-2 **REVISE**",
    )
    import re as _re

    old_block = (
        _re.compile(r"(?im)co_lead\s+gate-2\s+(?:BLOCK|REVISE)"),
        _re.compile(r"(?im)\bgate-2\s+(?:BLOCK|REVISE)\b"),
        _re.compile(r"(?im)validation/diff\s+.*\b(?:BLOCK|REVISE)\b"),
    )
    for fx in isolated:
        if any(p.search(fx) for p in old_block):
            raise AssertionError(f"isolation broken: old BLOCK markers fire on {fx!r}")
        if M.colead_verdict(fx) != "block":
            raise AssertionError(f"isolated bold {fx!r} must classify block")
    if M.colead_verdict(M.body(bold_pass)) != "pass":
        raise AssertionError("bolded PASS must classify pass")
    if M.colead_verdict(M.body(bold_block)) != "block":
        raise AssertionError("recorded bolded BLOCK must classify block")
    if M.colead_verdict("WAKE_VERIFIED: mid-exchange, no verdict") != "unknown":
        raise AssertionError("no-marker body must stay unknown")
    if M.colead_verdict("VALIDATION/DIFF REVIEW: PASS") != "pass":
        raise AssertionError("unwrapped PASS corpus form must still pass")
    if M.colead_verdict("VALIDATION/DIFF REVIEW: **PASS**") != "pass":
        raise AssertionError("bold PASS must not be swallowed by BLOCK wrap")
    live_bold_block = (
        "1784481932556-dff29dbe",
        "1784481971559-fb571dbd",
        "1784503781504-ad0ce5ac",
        "1784503803706-fa8857ff",
    )
    for rid in live_bold_block:
        live_rec = _journal(rid)
        if M.colead_verdict(M.body(live_rec)) != "block":
            raise AssertionError(f"live {rid} must classify block")
    print("PASS journal_i_wrap_and_rejection")

    # (ii) freeze identity: handoff anchors, plus1 does not
    if not M.is_claude_freeze(handoff, digest_live):
        raise AssertionError("handoff 1787060644843-8d7a142c must be the freeze")
    if M.is_claude_freeze(plus1, digest_live):
        raise AssertionError("plus1 1787060925934-3e31c7cb must not be a freeze")
    print("PASS journal_ii_freeze_identity")

    print("ALL pure T1–T18 + live replay PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
