#!/usr/bin/env python3
"""Fixture tests for commit_precondition_colead_gate.py."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HOOK = Path(__file__).with_name("commit_precondition_colead_gate.py")


def _load_hook_module():
    spec = importlib.util.spec_from_file_location("commit_precondition_colead_gate", HOOK)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


HOOK_MOD = _load_hook_module()
resolve_target_repo = HOOK_MOD.resolve_target_repo

DIGEST = "a" * 64
OTHER_DIGEST = "b" * 64
TASK = "1781862264540-5c174b3f"
WORKER_RCPT = "1781864000000-worker01"
FREEZE_ID = "1781864100000-freeze01"
PASS_ID = "1781864200000-a1b2c3d4"
FULL_OVERRIDE_REASON = (
    f"/mnt/c/Users/gabes/projects/claw-code-hrm-text-158 "
    f"DIFF_DIGEST={DIGEST} co_lead gate-2 PASS msg {PASS_ID}"
)


def rec(frm: str, body: str, ts: str, mid: str, reply_to: str = "") -> str:
    obj = {"ts": ts, "id": mid, "from": frm, "kind": "validation_receipt", "body": body}
    if reply_to:
        obj["reply_to"] = reply_to
    return json.dumps(obj)


def worker_body() -> str:
    return f"VALIDATION RECEIPT — slice task {TASK}"


def freeze_body(digest: str = DIGEST) -> str:
    return (
        f"claude gate-1 freeze handoff task {TASK}\n"
        f"DIFF_DIGEST: {digest}\n"
        f"reply_to worker receipt {WORKER_RCPT}"
    )


def pass_body(digest: str = DIGEST) -> str:
    return (
        f"co_lead gate-2 PASS validation/diff\n"
        f"DIFF_DIGEST: {digest}\n"
        f"task {TASK} threaded to {FREEZE_ID}"
    )


def fresh_chain(digest: str = DIGEST) -> list[str]:
    return [
        rec("codex", worker_body(), "2026-06-19T10:00:00Z", WORKER_RCPT),
        rec("claude", freeze_body(digest), "2026-06-19T10:05:00Z", FREEZE_ID, WORKER_RCPT),
        rec("codex_co_lead", pass_body(digest), "2026-06-19T10:10:00Z", PASS_ID, FREEZE_ID),
    ]


def run(command: str, log_lines: list[str] | None, cwd: Path | None = None) -> int:
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as fh:
        path = fh.name
        if log_lines is not None:
            fh.write("\n".join(log_lines) + ("\n" if log_lines else ""))
    env = {
        "PATH": "/usr/bin:/bin",
        "AI_ROOM_CHANNEL_LOG": path,
    }
    payload = {"tool_input": {"command": command}}
    proc = subprocess.run(
        [sys.executable, str(HOOK)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        env=env,
        cwd=str(cwd) if cwd else None,
    )
    Path(path).unlink(missing_ok=True)
    return proc.returncode


def setup_git_repo_with_staged(content: str = "hello\n") -> Path:
    tmp = Path(tempfile.mkdtemp())
    subprocess.run(["git", "init"], cwd=tmp, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=tmp,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "test"],
        cwd=tmp,
        check=True,
        capture_output=True,
    )
    f = tmp / "file.txt"
    f.write_text(content, encoding="utf-8")
    subprocess.run(["git", "add", "file.txt"], cwd=tmp, check=True, capture_output=True)
    return tmp


def staged_digest(repo: Path) -> str:
    proc = subprocess.run(
        ["git", "diff", "--cached"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )
    return hashlib.sha256(proc.stdout.encode("utf-8")).hexdigest()


CASES = []


def case(name: str, expected: int, command: str, log_lines: list[str] | None, repo: Path | None):
    CASES.append((name, expected, command, log_lines, repo))


repo = setup_git_repo_with_staged()
digest = staged_digest(repo)

case("non_git_allow", 0, "echo hello", None, None)
case("commit_no_log_block", 2, "git commit -m test", None, repo)
case("commit_unreadable_log_block", 2, "git commit -m test", ["{not json}"], repo)
case("commit_fresh_pass_allow", 0, "git commit -m test", fresh_chain(digest), repo)
case("commit_stale_digest_block", 2, "git commit -m test", fresh_chain(OTHER_DIGEST), repo)
case("commit_block_revise_block", 2, "git commit -m test", [
    rec("codex", worker_body(), "2026-06-19T10:00:00Z", WORKER_RCPT),
    rec("claude", freeze_body(digest), "2026-06-19T10:05:00Z", FREEZE_ID, WORKER_RCPT),
    rec(
        "codex_co_lead",
        f"co_lead gate-2 REVISE\nDIFF_DIGEST: {digest}",
        "2026-06-19T10:10:00Z",
        PASS_ID,
        FREEZE_ID,
    ),
], repo)
case("commit_pre_receipt_pass_block", 2, "git commit -m test", [
    rec("codex_co_lead", pass_body(digest), "2026-06-19T09:00:00Z", PASS_ID),
    rec("codex", worker_body(), "2026-06-19T10:00:00Z", WORKER_RCPT),
    rec("claude", freeze_body(digest), "2026-06-19T10:05:00Z", FREEZE_ID, WORKER_RCPT),
], repo)
case("push_allow", 0, "git push origin feature/test", fresh_chain(digest), repo)
case("push_force_block", 2, "git push --force origin feature/test", fresh_chain(digest), repo)
case("push_plus_refspec_block", 2, "git push origin +feature/test", fresh_chain(digest), repo)
case("commit_deferral_echo_block", 2, "git commit -m test", [
    rec("codex", worker_body(), "2026-06-19T10:00:00Z", WORKER_RCPT),
    rec("claude", freeze_body(digest), "2026-06-19T10:05:00Z", FREEZE_ID, WORKER_RCPT),
    rec(
        "codex_co_lead",
        f"Receipt noted. DIFF_DIGEST: {digest}\n"
        f"dual-accept pending — no co_lead approval yet; holding until Claude gate-1",
        "2026-06-19T10:10:00Z",
        PASS_ID,
        FREEZE_ID,
    ),
], repo)
case("commit_unrelated_later_worker_allow", 0, "git commit -m test", fresh_chain(digest) + [
    rec(
        "codex",
        "VALIDATION RECEIPT — unrelated task 1781999999999-deadbeef",
        "2026-06-19T11:00:00Z",
        "1781864999999-unrel01",
    ),
], repo)
case("override_trivial_block", 2, "git commit -m test\nCO_LEAD_GATE_OVERRIDE: x", None, repo)
case(
    "override_docs_only_block",
    2,
    "git commit -m test\nCO_LEAD_GATE_OVERRIDE: docs-only emergency",
    None,
    repo,
)
case(
    "override_full_allow",
    0,
    f"git commit -m test\nCO_LEAD_GATE_OVERRIDE: {FULL_OVERRIDE_REASON}",
    None,
    repo,
)
case(
    "override_missing_digest_block",
    2,
    f"git commit -m test\nCO_LEAD_GATE_OVERRIDE: /tmp/repo co_lead gate-2 PASS msg {PASS_ID}",
    None,
    repo,
)


def test_resolve_target_repo_unit() -> None:
    cwd = "/zenith/repo"
    repo, explicit = resolve_target_repo("git commit -m test", cwd=cwd)
    assert repo == str(Path(cwd).resolve())
    assert explicit is False

    hrm = "/mnt/c/Users/gabes/projects/claw-code-hrm-text-158"
    repo, explicit = resolve_target_repo(f'git -C "{hrm}" commit -m test', cwd=cwd)
    assert repo == str(Path(hrm).resolve())
    assert explicit is True

    repo, explicit = resolve_target_repo(f"cd {hrm} && git commit -m test", cwd=cwd)
    assert repo == str(Path(hrm).resolve())
    assert explicit is True

    other = "/other/repo"
    repo, explicit = resolve_target_repo(f"cd {hrm} && git -C {other} commit -m test", cwd=cwd)
    assert repo == str(Path(other).resolve())
    assert explicit is True

    base = tempfile.mkdtemp()
    sub = Path(base) / "sub"
    sub.mkdir()
    repo, explicit = resolve_target_repo(f"cd {base} && git -C sub commit -m test", cwd=cwd)
    assert repo == str(sub.resolve())
    assert explicit is True

    blocked_commands = [
        "git -C /hrm add . && git commit -m test",
        "cd /other; git commit -m test",
        "cd /other || exit; git commit -m test",
        "git --git-dir=/other/.git --work-tree=/other commit -m test",
        "git --bare commit -m test",
        "GIT_DIR=/other/.git git commit -m test",
        "env GIT_WORK_TREE=/other git commit -m test",
        "cd /a && cd /b && git commit -m test",
        "git -c user.name=x -C /target commit -m test",
        "git -c user.name=x commit -m test",
        "git ci -m test",
        "git commit -m test && git push",
        "cd $HOME && git commit -m test",
        "( cd /other && git commit -m test )",
        "{ cd /other && git commit -m test; }",
    ]
    for blocked in blocked_commands:
        repo, explicit = resolve_target_repo(blocked, cwd=cwd)
        assert repo is None, blocked
        assert explicit is True


def test_cross_repo_git_c_pass() -> None:
    zenith = setup_git_repo_with_staged("zenith\n")
    hrm = setup_git_repo_with_staged("hrm\n")
    hrm_digest = staged_digest(hrm)
    rc = run(
        f'git -C "{hrm}" commit -m test',
        fresh_chain(hrm_digest),
        zenith,
    )
    assert rc == 0, "git -C should digest HRM index while hook cwd is zenith"


def test_cross_repo_cd_prefix_pass() -> None:
    zenith = setup_git_repo_with_staged("zenith\n")
    hrm = setup_git_repo_with_staged("hrm\n")
    hrm_digest = staged_digest(hrm)
    rc = run(
        f'cd "{hrm}" && git commit -m test',
        fresh_chain(hrm_digest),
        zenith,
    )
    assert rc == 0, "cd prefix should digest target repo index"


def test_cross_repo_invalid_target_block() -> None:
    zenith = setup_git_repo_with_staged("zenith\n")
    digest = staged_digest(zenith)
    rc = run(
        'git -C /nonexistent/path/for/hook-test commit -m test',
        fresh_chain(digest),
        zenith,
    )
    assert rc == 2, "invalid explicit target must fail-closed"


def test_same_repo_unchanged() -> None:
    repo_path = setup_git_repo_with_staged("same-repo\n")
    digest = staged_digest(repo_path)
    rc = run("git commit -m test", fresh_chain(digest), repo_path)
    assert rc == 0


def test_cd_then_git_c_resolves_to_c_target_not_cd_repo() -> None:
    approved = setup_git_repo_with_staged("approved\n")
    other = setup_git_repo_with_staged("other\n")
    approved_digest = staged_digest(approved)
    other_digest = staged_digest(other)
    zenith = setup_git_repo_with_staged("zenith\n")
    rc_stale = run(
        f'cd "{approved}" && git -C "{other}" commit -m test',
        fresh_chain(approved_digest),
        zenith,
    )
    assert rc_stale == 2, "stale PASS for cd repo must not authorize commit -C other"
    rc_ok = run(
        f'cd "{approved}" && git -C "{other}" commit -m test',
        fresh_chain(other_digest),
        zenith,
    )
    assert rc_ok == 0, "digest must match commit -C target, not cd prefix"


def test_git_c_only_global_allowed() -> None:
    zenith = setup_git_repo_with_staged("zenith\n")
    target = setup_git_repo_with_staged("target\n")
    target_digest = staged_digest(target)
    rc = run(
        f'git -C "{target}" commit -m test',
        fresh_chain(target_digest),
        zenith,
    )
    assert rc == 0


def test_git_c_with_disallowed_global_blocks() -> None:
    repo_path = setup_git_repo_with_staged("repo\n")
    digest = staged_digest(repo_path)
    target = setup_git_repo_with_staged("target\n")
    rc = run(
        f'git -c user.name=test -C "{target}" commit -m test',
        fresh_chain(digest),
        repo_path,
    )
    assert rc == 2


def test_force_push_with_override_still_blocks() -> None:
    repo_path = setup_git_repo_with_staged("repo\n")
    rc = run(
        f"git push --force origin feature/test\nCO_LEAD_GATE_OVERRIDE: {FULL_OVERRIDE_REASON}",
        None,
        repo_path,
    )
    assert rc == 2


def test_push_override_without_commit_blocks() -> None:
    repo_path = setup_git_repo_with_staged("repo\n")
    rc = run(
        f"git push origin feature/test\nCO_LEAD_GATE_OVERRIDE: {FULL_OVERRIDE_REASON}",
        None,
        repo_path,
    )
    assert rc == 2


def test_cd_semicolon_blocks_even_with_matching_cwd_pass() -> None:
    repo_path = setup_git_repo_with_staged("repo\n")
    digest = staged_digest(repo_path)
    other = setup_git_repo_with_staged("other\n")
    rc = run(
        f'cd "{other}"; git commit -m test',
        fresh_chain(digest),
        repo_path,
    )
    assert rc == 2


def test_cd_or_exit_blocks_even_with_matching_cwd_pass() -> None:
    repo_path = setup_git_repo_with_staged("repo\n")
    digest = staged_digest(repo_path)
    other = setup_git_repo_with_staged("other\n")
    rc = run(
        f'cd "{other}" || exit 1; git commit -m test',
        fresh_chain(digest),
        repo_path,
    )
    assert rc == 2


def test_git_dir_work_tree_globals_block() -> None:
    repo_path = setup_git_repo_with_staged("repo\n")
    digest = staged_digest(repo_path)
    other = setup_git_repo_with_staged("other\n")
    rc = run(
        f'git --git-dir="{other}/.git" --work-tree="{other}" commit -m test',
        fresh_chain(digest),
        repo_path,
    )
    assert rc == 2


def test_git_bare_global_blocks() -> None:
    repo_path = setup_git_repo_with_staged("repo\n")
    digest = staged_digest(repo_path)
    rc = run("git --bare commit -m test", fresh_chain(digest), repo_path)
    assert rc == 2


def test_git_dir_env_override_blocks() -> None:
    repo_path = setup_git_repo_with_staged("repo\n")
    digest = staged_digest(repo_path)
    other = setup_git_repo_with_staged("other\n")
    rc = run(
        f'GIT_DIR="{other}/.git" git commit -m test',
        fresh_chain(digest),
        repo_path,
    )
    assert rc == 2


def test_disallowed_global_c_blocks() -> None:
    repo_path = setup_git_repo_with_staged("repo\n")
    digest = staged_digest(repo_path)
    rc = run("git -c user.name=test commit -m test", fresh_chain(digest), repo_path)
    assert rc == 2


def test_git_ci_alias_blocks() -> None:
    repo_path = setup_git_repo_with_staged("repo\n")
    digest = staged_digest(repo_path)
    rc = run("git ci -m test", fresh_chain(digest), repo_path)
    assert rc == 2


def test_commit_and_push_same_command_blocks() -> None:
    repo_path = setup_git_repo_with_staged("repo\n")
    digest = staged_digest(repo_path)
    rc = run(
        "git commit -m test && git push origin feature/test",
        fresh_chain(digest),
        repo_path,
    )
    assert rc == 2


def test_prior_git_action_blocks() -> None:
    repo_path = setup_git_repo_with_staged("repo\n")
    digest = staged_digest(repo_path)
    other = setup_git_repo_with_staged("other\n")
    rc = run(
        f'git -C "{other}" add . && git commit -m test',
        fresh_chain(digest),
        repo_path,
    )
    assert rc == 2


def test_subshell_cd_blocks() -> None:
    repo_path = setup_git_repo_with_staged("repo\n")
    digest = staged_digest(repo_path)
    other = setup_git_repo_with_staged("other\n")
    rc = run(
        f'( cd "{other}" && git commit -m test )',
        fresh_chain(digest),
        repo_path,
    )
    assert rc == 2


def test_dynamic_path_expansion_blocks() -> None:
    repo_path = setup_git_repo_with_staged("repo\n")
    digest = staged_digest(repo_path)
    rc = run(
        'git -C "$HOME" commit -m test',
        fresh_chain(digest),
        repo_path,
    )
    assert rc == 2


def test_commit_heredoc_message_pass() -> None:
    repo_path = setup_git_repo_with_staged("repo\n")
    digest = staged_digest(repo_path)
    command = """git commit -F - <<'EOF'
commit message via heredoc
EOF"""
    rc = run(command, fresh_chain(digest), repo_path)
    assert rc == 0


def test_grep_git_commit_substring_pass_through() -> None:
    repo_path = setup_git_repo_with_staged("repo\n")
    rc = run('grep "git commit" README.md', None, repo_path)
    assert rc == 0, "grep containing git commit substring must not trigger gate"


def test_echo_git_commit_substring_pass_through() -> None:
    repo_path = setup_git_repo_with_staged("repo\n")
    rc = run('echo "run git commit later"', None, repo_path)
    assert rc == 0, "echo containing git commit substring must not trigger gate"


def test_core_worktree_global_blocks() -> None:
    repo_path = setup_git_repo_with_staged("repo\n")
    digest = staged_digest(repo_path)
    other = setup_git_repo_with_staged("other\n")
    rc = run(
        f'git -c core.worktree="{other}" commit -m test',
        fresh_chain(digest),
        repo_path,
    )
    assert rc == 2


def test_bash_c_wrapper_blocks() -> None:
    repo_path = setup_git_repo_with_staged("repo\n")
    digest = staged_digest(repo_path)
    rc = run(
        'bash -c "git commit -m test"',
        fresh_chain(digest),
        repo_path,
    )
    assert rc == 2


def test_git_index_file_env_blocks() -> None:
    repo_path = setup_git_repo_with_staged("repo\n")
    digest = staged_digest(repo_path)
    rc = run(
        "GIT_INDEX_FILE=/tmp/other-index git commit -m test",
        fresh_chain(digest),
        repo_path,
    )
    assert rc == 2


def test_quoted_git_command_word_blocks() -> None:
    repo_path = setup_git_repo_with_staged("repo\n")
    digest = staged_digest(repo_path)
    rc = run("'git' commit -m test", fresh_chain(digest), repo_path)
    assert rc == 2, "quoted git command word must gate and fail-closed block"


def test_double_quoted_git_command_word_blocks() -> None:
    repo_path = setup_git_repo_with_staged("repo\n")
    digest = staged_digest(repo_path)
    rc = run('"git" commit -m test', fresh_chain(digest), repo_path)
    assert rc == 2


def test_path_qualified_git_blocks() -> None:
    repo_path = setup_git_repo_with_staged("repo\n")
    digest = staged_digest(repo_path)
    rc = run("/usr/bin/git commit -m test", fresh_chain(digest), repo_path)
    assert rc == 2


def test_relative_path_git_blocks() -> None:
    repo_path = setup_git_repo_with_staged("repo\n")
    digest = staged_digest(repo_path)
    rc = run("./git commit -m test", fresh_chain(digest), repo_path)
    assert rc == 2


def test_dotgit_path_not_misclassified_as_commit() -> None:
    repo_path = setup_git_repo_with_staged("repo\n")
    rc = run("ls repo/.git", None, repo_path)
    assert rc == 0, ".git directory path must not trigger commit gate"


def test_wrapper_command_git_blocks() -> None:
    repo_path = setup_git_repo_with_staged("repo\n")
    digest = staged_digest(repo_path)
    for cmd in (
        "command git commit -m test",
        "exec git commit -m test",
        "env git commit -m test",
    ):
        rc = run(cmd, fresh_chain(digest), repo_path)
        assert rc == 2, cmd


def test_post_commit_operators_block() -> None:
    repo_path = setup_git_repo_with_staged("repo\n")
    digest = staged_digest(repo_path)
    for cmd in (
        "git commit -m test && echo ok",
        "git commit -m test; echo ok",
        "git commit -m test | cat",
        "git commit -m test & echo ok",
    ):
        rc = run(cmd, fresh_chain(digest), repo_path)
        assert rc == 2, cmd


def test_escaped_git_blocks() -> None:
    repo_path = setup_git_repo_with_staged("repo\n")
    digest = staged_digest(repo_path)
    rc = run(r"\git commit -m test", fresh_chain(digest), repo_path)
    assert rc == 2


def test_newline_separated_commands_block() -> None:
    repo_path = setup_git_repo_with_staged("repo\n")
    digest = staged_digest(repo_path)
    rc = run("git commit -m test\necho ok", fresh_chain(digest), repo_path)
    assert rc == 2


def test_comment_then_commit_blocks() -> None:
    repo_path = setup_git_repo_with_staged("repo\n")
    digest = staged_digest(repo_path)
    rc = run("# comment\ngit commit -m test", fresh_chain(digest), repo_path)
    assert rc == 2


def test_dynamic_ifs_expansion_blocks() -> None:
    repo_path = setup_git_repo_with_staged("repo\n")
    digest = staged_digest(repo_path)
    for cmd in (
        "git${IFS}commit -m test",
        "git$IFS commit -m test",
        "${GIT:-git} commit -m test",
    ):
        rc = run(cmd, fresh_chain(digest), repo_path)
        assert rc == 2, cmd


def test_wrapper_with_options_blocks() -> None:
    repo_path = setup_git_repo_with_staged("repo\n")
    digest = staged_digest(repo_path)
    for cmd in (
        "env -i git commit -m test",
        "env -C /tmp git commit -m test",
        "sudo -u gabe git commit -m test",
        "nice -n 5 git commit -m test",
    ):
        rc = run(cmd, fresh_chain(digest), repo_path)
        assert rc == 2, cmd


def test_plain_dollar_var_command_word_blocks() -> None:
    repo_path = setup_git_repo_with_staged("repo\n")
    digest = staged_digest(repo_path)
    rc = run("$GIT commit -m test", fresh_chain(digest), repo_path)
    assert rc == 2


def test_plain_dollar_var_mid_token_blocks() -> None:
    repo_path = setup_git_repo_with_staged("repo\n")
    digest = staged_digest(repo_path)
    rc = run("git$EMPTY commit -m test", fresh_chain(digest), repo_path)
    assert rc == 2


def test_env_string_exec_option_blocks() -> None:
    repo_path = setup_git_repo_with_staged("repo\n")
    digest = staged_digest(repo_path)
    rc = run('env -S "git commit -m test"', fresh_chain(digest), repo_path)
    assert rc == 2


def test_eval_git_commit_blocks() -> None:
    repo_path = setup_git_repo_with_staged("repo\n")
    digest = staged_digest(repo_path)
    for cmd in (
        "eval git commit -m test",
        'eval "git commit -m test"',
    ):
        rc = run(cmd, fresh_chain(digest), repo_path)
        assert rc == 2, cmd


def test_timeout_git_commit_blocks() -> None:
    repo_path = setup_git_repo_with_staged("repo\n")
    digest = staged_digest(repo_path)
    rc = run("timeout 5 git commit -m x", fresh_chain(digest), repo_path)
    assert rc == 2


def main() -> int:
    failed = 0
    try:
        test_resolve_target_repo_unit()
        print("PASS resolve_target_repo_unit")
    except AssertionError as exc:
        print(f"FAIL resolve_target_repo_unit: {exc}")
        failed += 1

    integration_tests = (
        test_cross_repo_git_c_pass,
        test_cross_repo_cd_prefix_pass,
        test_cross_repo_invalid_target_block,
        test_same_repo_unchanged,
        test_cd_then_git_c_resolves_to_c_target_not_cd_repo,
        test_git_c_only_global_allowed,
        test_git_c_with_disallowed_global_blocks,
        test_force_push_with_override_still_blocks,
        test_push_override_without_commit_blocks,
        test_cd_semicolon_blocks_even_with_matching_cwd_pass,
        test_cd_or_exit_blocks_even_with_matching_cwd_pass,
        test_git_dir_work_tree_globals_block,
        test_git_bare_global_blocks,
        test_git_dir_env_override_blocks,
        test_disallowed_global_c_blocks,
        test_git_ci_alias_blocks,
        test_commit_and_push_same_command_blocks,
        test_prior_git_action_blocks,
        test_subshell_cd_blocks,
        test_dynamic_path_expansion_blocks,
        test_commit_heredoc_message_pass,
        test_grep_git_commit_substring_pass_through,
        test_echo_git_commit_substring_pass_through,
        test_core_worktree_global_blocks,
        test_bash_c_wrapper_blocks,
        test_git_index_file_env_blocks,
        test_quoted_git_command_word_blocks,
        test_double_quoted_git_command_word_blocks,
        test_path_qualified_git_blocks,
        test_relative_path_git_blocks,
        test_dotgit_path_not_misclassified_as_commit,
        test_wrapper_command_git_blocks,
        test_post_commit_operators_block,
        test_escaped_git_blocks,
        test_newline_separated_commands_block,
        test_comment_then_commit_blocks,
        test_dynamic_ifs_expansion_blocks,
        test_wrapper_with_options_blocks,
        test_plain_dollar_var_command_word_blocks,
        test_plain_dollar_var_mid_token_blocks,
        test_env_string_exec_option_blocks,
        test_eval_git_commit_blocks,
        test_timeout_git_commit_blocks,
    )
    for test_fn in integration_tests:
        try:
            test_fn()
            print(f"PASS {test_fn.__name__}")
        except AssertionError as exc:
            print(f"FAIL {test_fn.__name__}: {exc}")
            failed += 1

    for name, expected, command, log_lines, repo_path in CASES:
        rc = run(command, log_lines, repo_path)
        if rc != expected:
            print(f"FAIL {name}: expected exit {expected}, got {rc}")
            failed += 1
        else:
            print(f"PASS {name}")
    if failed:
        print(f"{failed} failed")
        return 1
    total = len(CASES) + 1 + len(integration_tests)
    print(f"ALL {total} PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
