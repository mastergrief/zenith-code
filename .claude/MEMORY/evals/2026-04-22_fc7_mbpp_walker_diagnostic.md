# FC7 — MBPP walker diagnostic (partial)

**Status**: Incomplete. 4 of N=5 problems finished (MBPP#5-#8); #9
was killed after 15+ min stuck in long generation.

## Configuration

- Daemon: fresh restart with `MBPP_MAX_TOKENS=16384` in env
- `EVAL_CTX_SIZE=32768`, `max_tokens=16384`
- `USE_TQ4_KV=True`
- Corpus: MBPP N=5, skip=5 (problems MBPP#5–#9)

## Results (4/5 problems, problem 5 killed)

| Problem | fn_name | Stock Gemma | Force-fence | Outcome |
|---|---|---|---|---|
| MBPP#5 | radian_degree | 25s NO CODE | +31s NO CODE | format_fail |
| MBPP#6 | find_literals | 63s NO CODE | +73s NO CODE | format_fail |
| MBPP#7 | bell_Number | 267s NO CODE | +105s NO CODE | format_fail |
| MBPP#8 | floor_Min | 114s NO CODE | +379s NO CODE | format_fail |
| MBPP#9 | remove_kth_element | killed @ ~15 min | — | — |

Prior N=2 run (MBPP#2, #3):

| Problem | fn_name | Outcome |
|---|---|---|
| MBPP#2 | get_ludic | format_fail (324s + 96s) |
| MBPP#3 | reverse_words | **CLEAN 3/3 (39s)** |

## Finding

**Gemma + 16K max_tokens produces unparseable rambling on most MBPP
problems.** Of 6 MBPP problems attempted across both runs, only 1
(reverse_words) produced extractable code; it passed cleanly without
needing walker intervention. The other 5 failed at the extraction
stage (format_fail) — Gemma emitted prose or unparseable text, not
a fenced code block.

This is consistent with the R53.35 reaudit finding: extractor strictness
+ Gemma's prose-heavy output format are the dominant failure modes on
MBPP, not walker coverage gaps. The walker can't fire if there's no
extractable code.

## Implication

The `ast_repair` + `repair_cascade` walker delivers lift on the
problem class where:
  1. Gemma DOES produce extractable code
  2. That code FAILS at test-time (syntax bug, wrong name, KeyError, etc)

Both conditions must hold. On MBPP-style problems with 16K max_tokens:
  - Condition (1) fails ~83% of the time (5/6 runs)
  - Condition (2) would apply to the remaining 17%, but we saw no
    test-time fails in the completed runs

**The diagnostic doesn't falsify the walker thesis** — R53.35 already
shipped two real lifts (token_bucket 0/0→5/5, csv_column_stats
0/0→8/8) on the R53.0 corpus where Gemma DOES produce extractable
code. It just confirms that raw MBPP isn't the right benchmark
surface — Gemma's extraction rate is the upstream bottleneck.

## Next-step recommendation

Better benchmark corpus for walker lift: **problems where Gemma
produces code but that code fails at test time.** Candidates:
  - HumanEvalPlus with explicit function signatures in the prompt
    (reduces prose rambling)
  - MBPP subset filtered by prior-run `extract_code` success
  - Hand-curated corpus of Gemma output snippets that fail at runtime
  - The existing `calm/llm_computer/tests/` synthetic fixture set

With 16K max_tokens, Gemma's format drift grows. A lower cap
(4K-8K) forces tighter output and might improve extraction rate —
**untested this cycle**. Worth measuring in a future round.

## Deferred work

- Re-run with `MBPP_MAX_TOKENS=4096` to test the extraction-rate
  hypothesis
- Build a curated `walker_target_corpus` of Gemma-produces-code-but-fails
  examples
- Separate "extraction failure rate" metric in r53_39 output
- Prompt-engineering for MBPP to reduce prose: force Gemma to emit
  code-fence BEFORE prose via more aggressive template
