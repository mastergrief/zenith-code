# PIPELINE - RUN THE DEFAULT IMPLEMENTATION LOOP ON A TASK

Execute the non-trivial work described in [$ARGUMENTS] through the full ai-room
Default Implementation Loop. This command LAUNCHES the loop defined in
`.claude/rules/workflow.md` and the ai-room collaboration rules in
`.claude/CLAUDE.md` / `.claude/rules/AI_ROOM_COLLAB.md`. It does not define new
policy. On any conflict those files win.

**STEP 1 - SCOPE** (Claude direct)
- CAPTURE INTENT FIRST (compulsory): Before restate/board-task, fire
  `AskUserQuestion` to capture intent for [$ARGUMENTS]. Relay locked answers to
  the room as a persisted non-ack record threaded to source where possible.
- Restate smallest faithful intent; name candidate seams/files and risk class.
- Create ai-room board task with provenance + structured decision_contract.

**STEP 2 - SCOPE GATE** (codex_co_lead - HARD BLOCKING)
- Post scope/direction to codex_co_lead with requires_response_from.
- **Passive-wait** for reply — do not poll.

**STEP 3 - DEEPEN** (plan-dev read-only grounding — mandatory)
- `plan-dev` investigates agreed seams read-only with live measurement +
  file:line cites; never edits without `+1 implement`.
- Material scope expansion re-runs STEP 2.

**STEP 4 - PLAN** (Claude direct)
- Author implementation spec on the board task: file scope, invariants,
  validation commands, acceptance criteria, stop conditions, gates.

**STEP 5 - PLAN GATE** (codex_co_lead - HARD BLOCKING)
- Post spec to codex_co_lead; passive-wait. No dispatch until co_lead replies.

**STEP 6 - DELEGATE** (Claude -> plan-dev)
- Dispatch fresh `plan-dev` citing task id. Worker acks grounding + short
  approach BEFORE editing; Claude sanity-checks and gives `+1 implement`.

**STEP 7 - IMPLEMENT + TEST** (plan-dev)
- `plan-dev` implements AND runs focused validation; posts validation_receipt to
  **claude ONLY** (`REPORT_TO: [claude]` — sink + framing). Worker NEVER commits.

**STEP 8 - REVIEW GATE** (sequential: gate1_audit gate-1 → co_lead gate-2)
- Claude frames the handoff; `gate1_audit` verifies, freezes, or bounces.
- Cross-thread frozen handoff + `DIFF_DIGEST` to codex_co_lead; passive-wait
  for PASS/REVISE. co_lead gate-2 is the LAST gate per changed diff.
- On REVISE/FAIL: redispatch (STEP 10) WITHOUT eval.

**STEP 9 - EVAL** (Claude, incl. as `test-operator` for formal runs — NOT co_lead)
- Run artifact smoke AFTER review PASS and BEFORE commit/push when a runtime
  surface exists. Formal training/proof runs execute Claude-side after `+1 launch`.
- SKIP with stated reason when no runtime surface (docs/governance/hooks).

**STEP 10 - REDISPATCH** (defects from review OR eval)
- Defects return to same worker (`RETAIN OVERRIDE: defect-cycle`).
- New scope or context pressure → recycle and spawn fresh.

**STEP 11 - COMMIT / PUSH** (Claude + Gabe; commit-precondition hook enforces co_lead PASS)
- Gabe standing auto-research directive satisfies AUQ for routine LOW commits.
- Stage EXACT files only; commit; push only on explicit `+1 push` or
  `+1 commit+push`. co_lead is NOT a push gate.

**IMPORTANT**:
- Thinking stays parallel; artifact review gates are sequential.
- Three co_lead gates (scope, plan, validation/diff) are HARD BLOCKING;
  validation/diff is co_lead's last gate per changed diff.
- Passive-wait-don't-poll at gates.
- Roles: **plan-dev** (plan+implement) is the only spawnable codex worker;
  **test-operator** (formal runs) is Claude-carried, not spawned — NOT
  legacy codex-explore/dev/tmux-tester names from reference repos.
