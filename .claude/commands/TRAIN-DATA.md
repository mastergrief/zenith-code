# TRAIN-DATA — Training Data Generation & Expansion

**Input**: [$ARGUMENTS] - Domain to expand (e.g., "rust", "python", "reasoning"), count target, or specific topic areas

**Training data workflow**: Generate, validate, and expand high-quality training examples for fine-tuning Qwen 3.5 specialists. Uses the trainer agent for data writing and validation.

**When to use:**
- Expanding existing training data for a domain
- Writing new hand-crafted examples for reasoning base or specialists
- Validating and cleaning existing JSONL files
- Merging and filtering data after additions

---

## Step 0: Assess Current State (orchestrator)

Before spawning, check current data inventory:
```bash
# Count examples per file
wc -l agents/distill/data/*.jsonl
# Check for duplicates
PYTHONPATH=. python3 -c "
import json
from pathlib import Path
for f in sorted(Path('agents/distill/data').glob('*.jsonl')):
    lines = f.read_text().strip().split('\n')
    firsts = set()
    dupes = 0
    for l in lines:
        first60 = json.loads(l)['messages'][1]['content'][:60]
        if first60 in firsts: dupes += 1
        firsts.add(first60)
    print(f'{f.name}: {len(lines)} examples, {dupes} dupes')
"
```

Determine:
- Which domain needs expansion (from $ARGUMENTS or largest gap)
- Current count vs target (5,000 per domain per config.py)
- Quality issues in existing data

---

## Step 1: Generate Examples

Spawn trainer agent:

`subagent_type: trainer`, `model: opus`

> Generate $COUNT high-quality training examples for domain: $ARGUMENTS
>
> **Current data state**: [inject Step 0 inventory results]
>
> ### Categories to Cover
> [Based on domain — orchestrator selects from gaps in existing coverage]
>
> ### Writing Process
> 1. Write examples as a Python script that validates and appends
> 2. Each example: system prompt + user question + assistant response with `<think>` block
> 3. Validate: 3 messages, assistant starts with `<think>`, think block >200 chars, response >200 chars
> 4. Check for duplicates against existing data (first 60 chars of user message)
> 5. Write to appropriate JSONL file
>
> ### Quality Checklist (per example)
> - [ ] Realistic question someone would actually ask
> - [ ] `<think>` block shows genuine analytical reasoning (300-800 words)
> - [ ] Considers multiple approaches before choosing
> - [ ] Identifies edge cases and pitfalls
> - [ ] Final answer is concrete with working code where appropriate
> - [ ] No hallucinated facts, libraries, or API methods
> - [ ] Technically correct — verified against real documentation/behavior
>
> ### Output Format
> Write a Python script to `/tmp/gen_batch.py` that:
> ```python
> import json
> from pathlib import Path
>
> examples = [
>     {"messages": [
>         {"role": "system", "content": "You are a helpful assistant..."},
>         {"role": "user", "content": "..."},
>         {"role": "assistant", "content": "<think>\n...\n</think>\n\n..."}
>     ]},
>     # ... more examples
> ]
>
> path = Path("agents/distill/data/DOMAIN.jsonl")
> existing = set()
> if path.exists():
>     for line in path.read_text().strip().split('\n'):
>         msg = json.loads(line)['messages'][1]['content'][:60]
>         existing.add(msg)
>
> added = 0
> with open(path, "a") as f:
>     for ex in examples:
>         assert len(ex["messages"]) == 3
>         assert ex["messages"][2]["content"].startswith("<think>")
>         key = ex["messages"][1]["content"][:60]
>         if key not in existing:
>             f.write(json.dumps(ex) + "\n")
>             added += 1
>
> print(f"Added {added} examples, skipped {len(examples)-added} dupes")
> ```
> Run the script after writing.

---

## Step 2: Validate & Merge

After examples are written:

```bash
# Validate all examples in the file
PYTHONPATH=. python3 -c "
import json, sys
from pathlib import Path
path = Path('agents/distill/data/$DOMAIN.jsonl')
errors = []
for i, line in enumerate(path.read_text().strip().split('\n'), 1):
    try:
        ex = json.loads(line)
        msgs = ex['messages']
        if len(msgs) != 3: errors.append(f'Line {i}: {len(msgs)} messages, expected 3')
        if not msgs[2]['content'].startswith('<think>'): errors.append(f'Line {i}: no <think> block')
        if len(msgs[2]['content']) < 200: errors.append(f'Line {i}: response too short ({len(msgs[2][\"content\"])} chars)')
    except json.JSONDecodeError as e:
        errors.append(f'Line {i}: invalid JSON: {e}')
if errors:
    for e in errors: print(f'ERROR: {e}', file=sys.stderr)
    sys.exit(1)
print(f'All {i} examples valid')
"

# If reasoning base data was updated, re-merge
PYTHONPATH=. python3 -m agents.distill.filter_reasoning --merge
```

---

## Step 3: Report

Present to user:
- Examples added (count, domain)
- Categories covered
- Any quality issues found and fixed
- Updated data inventory
- Recommended next steps (more data? training run? different domain?)

---

## Rules

- **Data quality > data quantity** — 20 great examples beat 100 mediocre ones
- Hand-written examples are always appended, never overwritten
- Always validate before and after writing
- Always check for duplicates (first 60 chars)
- Re-run `filter_reasoning --merge` after updating reasoning base data
- Training data format: `{"messages": [system, user, assistant]}`, assistant starts with `<think>`
- Think blocks must show genuine reasoning, not just restate the question
- Answers must be technically correct — verify claims
- Never generate NLP benchmark patterns, non-technical content, or science trivia
