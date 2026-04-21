"""R22b lift — measure PT+Delta MQAR card's lift on the fail corpus.

Pre-req: scripts/r22b_gate.py must have produced .cache/r22b/fail_corpus.jsonl.

Protocol:
1. Load fail corpus (the prompts where stock Gemma missed).
2. Install MQAR card via scripts/r22_install_mqar_card.install().
3. For each fail, run Gemma-with-card forward pass, measure top token.
4. Report: {still_fails, now_solved, regressed_to_different_wrong}.

This is the R22b measurement round. Iteration cadence N=5 per
eval_defaults.ITERATION_N: first round measures on 5 fails only,
subsequent rounds tweak adapter/install if needed, final round scales
to FINAL_N for commit baseline.

Usage: bin/gemma-run scripts/r22b_measure_lift.py [--n N]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Daemon binds m, tok
assert "m" in globals() and "tok" in globals(), (  # type: ignore[name-defined]
    "run via bin/gemma-run scripts/r22b_measure_lift.py"
)

from calm.llm_computer.gemma_substrate import KVCache
from calm.llm_computer.eval_defaults import ITERATION_N, FINAL_N

# Import adapter + install functions directly from the sibling script by
# file path — scripts/ isn't a Python package, so `from scripts.x import y`
# fails. Use importlib to load the sibling module.
import importlib.util as _ilu
_r22_spec = _ilu.spec_from_file_location(
    "r22_install_mqar_card",
    ROOT / "scripts" / "r22_install_mqar_card.py",
)
_r22_mod = _ilu.module_from_spec(_r22_spec)
# Bypass `main()` auto-exec by stripping the trailing two lines before load.
# Simpler approach: catch SystemExit / just let the helper functions be
# defined via exec of the file with __name__ set to block main(). But since
# r22's main() IS the top-level call, loading the module runs it. Easiest
# workaround: read the source, slice off the trailing main() + print, exec.
_src = (ROOT / "scripts" / "r22_install_mqar_card.py").read_text()
# Drop the final "main()\nprint("R22_DONE")" block
_src = _src.split("main()\nprint(\"R22_DONE\")")[0]
_ns = {"__name__": "_r22_mod", "__file__": str(ROOT / "scripts" / "r22_install_mqar_card.py")}
# The loaded module does an `if "m" not in globals()` fallback that tries to
# import GemmaSubstrate from scratch. In daemon mode m/tok are already in our
# globals; forward them.
_ns["m"] = m  # type: ignore[name-defined]
_ns["tok"] = tok  # type: ignore[name-defined]
exec(compile(_src, str(ROOT / "scripts" / "r22_install_mqar_card.py"), "exec"),
     _ns)
load_mqar_card = _ns["load_mqar_card"]
install = _ns["install"]
parse_mqar_prompt = _ns["parse_mqar_prompt"]
mqar_to_ids = _ns["mqar_to_ids"]


CACHE = ROOT / ".cache" / "r22b"
FAIL_CORPUS = CACHE / "fail_corpus.jsonl"
CKPT = ROOT / "calm/hrm/checkpoints/copy_augmented_delta_mqar_best.pt"


def build_gemma_digit_ids():
    return {d: tok.encode(f" {d}")[-1] for d in range(10)}  # type: ignore[name-defined]


def main():
    # Arg parsing: read sys.argv in daemon mode (daemon passes script args).
    n_arg = ITERATION_N
    for i, arg in enumerate(sys.argv):
        if arg == "--n" and i + 1 < len(sys.argv):
            n_arg = int(sys.argv[i + 1])
        elif arg == "--final":
            n_arg = FINAL_N

    print(f"[r22b.lift] loading fail corpus: {FAIL_CORPUS}")
    if not FAIL_CORPUS.exists():
        print(f"[r22b.lift] ERROR: fail corpus missing. Run r22b_gate.py first.")
        return
    fails = [json.loads(line) for line in FAIL_CORPUS.open()]
    print(f"[r22b.lift] {len(fails)} prompts in fail corpus")
    if len(fails) == 0:
        print("[r22b.lift] empty fail corpus — Gemma solved everything. No "
              "failure surface to measure lift on.")
        return

    # Cap to iteration size
    fails = fails[:n_arg]
    print(f"[r22b.lift] iterating on {len(fails)} prompts (N={n_arg})")

    # Load + install card
    print(f"[r22b.lift] loading MQAR card: {CKPT.name}")
    card = load_mqar_card(CKPT)
    slot, state, hook = install(m, card, layer_idx=30, ch_off=2480)  # type: ignore[name-defined]
    digit_ids = build_gemma_digit_ids()

    print("\n=== RESULTS ===")
    n_still_wrong = 0
    n_solved = 0
    n_other_wrong = 0  # different wrong answer
    n_parse_miss = 0   # adapter couldn't parse the prompt
    per_cell = {}

    t0 = time.time()
    for i, fc in enumerate(fails):
        prompt = fc["prompt"]
        expected = fc["expected"]
        baseline_top = fc["top_token"]

        # Parse + stash
        mqar_str = parse_mqar_prompt(prompt)
        if mqar_str is None:
            state["mqar_ids"] = None
            state["active"] = False
            n_parse_miss += 1
            parsed_marker = "(parse miss)"
        else:
            state["mqar_ids"] = mqar_to_ids(mqar_str)
            state["active"] = True
            parsed_marker = mqar_str[:40] + ("..." if len(mqar_str) > 40 else "")

        ids = tok.encode(prompt)  # type: ignore[name-defined]
        cache = KVCache(m.config.n_layers, device="cuda")  # type: ignore[name-defined]
        with torch.no_grad():
            logits = m.forward(  # type: ignore[name-defined]
                torch.tensor([ids]), device="cuda",
                kv_cache=cache, start_pos=0,
            )
        top = int(logits[0, -1].argmax())
        top_tok = tok.id_to_token.get(top, "?")  # type: ignore[name-defined]
        got = top_tok.lstrip(" ▁")

        if top == digit_ids[int(expected)]:
            n_solved += 1
            verdict = "SOLVED"
        elif top in digit_ids.values() and top != digit_ids.get(int(fc["got_digit"])
                                                                  if fc["got_digit"].isdigit()
                                                                  else -1, -99):
            # Different wrong digit than baseline
            n_other_wrong += 1
            verdict = "OTHER_WRONG"
        else:
            n_still_wrong += 1
            verdict = "STILL_WRONG"

        cell_key = (fc["n_pairs"], fc["distractor_tokens"])
        per_cell.setdefault(cell_key, []).append(verdict)

        mark = "✓" if verdict == "SOLVED" else ("~" if verdict == "OTHER_WRONG" else "✗")
        print(f"  [{i+1}/{len(fails)}] N={fc['n_pairs']} "
              f"dist={fc['distractor_tokens']} "
              f"base={baseline_top!r} → card={top_tok!r} exp={expected!r} "
              f"{mark} {verdict}  parsed: {parsed_marker}")

    elapsed = time.time() - t0
    print(f"\n=== SUMMARY (elapsed {elapsed:.1f}s) ===")
    print(f"  N={len(fails)} prompts (from fail corpus)")
    print(f"    SOLVED       {n_solved:3d}  (card lifted these)")
    print(f"    OTHER_WRONG  {n_other_wrong:3d}  (wrong but different from baseline)")
    print(f"    STILL_WRONG  {n_still_wrong:3d}")
    print(f"    parse misses {n_parse_miss:3d}  (adapter couldn't build MQAR string)")

    if per_cell:
        print("\n=== PER-CELL LIFT ===")
        for (n_p, dist), verdicts in sorted(per_cell.items()):
            n_s = verdicts.count("SOLVED")
            print(f"  N_pairs={n_p} distractor={dist}: "
                  f"{n_s}/{len(verdicts)} solved")

main()
print("R22B_LIFT_DONE")
