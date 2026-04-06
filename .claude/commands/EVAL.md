# EVAL — Model Evaluation & A/B Comparison

**Input**: [$ARGUMENTS] - Model to evaluate (e.g., "reasoning-base", "specialist-py"), or "compare <model-a> <model-b>"

**Model evaluation workflow**: Test a fine-tuned model against reference prompts, compare against base model or teacher, and produce a quality verdict. Uses trainer for prompt design and harness-tester for live model interaction.

**When to use:**
- After a training run completes — evaluate the new model
- Comparing two models (fine-tuned vs base, specialist vs generalist)
- Validating that a model meets quality bar before deployment
- Regression testing after retraining

---

## Step 0: Assess Environment (orchestrator)

```bash
# What models are available?
ollama list 2>/dev/null
curl -s localhost:8080/health 2>/dev/null && echo "llama.cpp running"

# What's the model under test?
# Ollama: ollama show <model> --modelfile
# llama.cpp: check which GGUF is loaded
```

Determine:
- Model under test (from $ARGUMENTS)
- Comparison model (base Qwen 3.5, or teacher 9B)
- Serving backend (Ollama or llama.cpp)
- Which eval prompts to use

---

## Step 1: Design Eval Prompts

### Standard Eval Set (always run these)

These are the prompts that the 0.8B model got wrong. They're the quality bar:

1. **Race Condition**: "Two users try to buy the last item in stock simultaneously. How should the backend handle this?"
   - Expected: atomic `UPDATE ... WHERE stock > 0`, optimistic locking, or database-level constraint
   - Fail signal: "priority-based synchronization", vague mutex talk

2. **OOMKilled**: "My Docker container keeps getting OOMKilled in Kubernetes. How do I debug this?"
   - Expected: `kubectl describe pod`, check resource limits vs actual usage, `docker stats`, memory profiling
   - Fail signal: hallucinated kubectl commands, wrong flags

3. **Architecture**: "When should I use a message broker vs direct API calls between services?"
   - Expected: broker for async/decoupled/fan-out, direct for sync/simple/low-latency
   - Fail signal: wrong latency claims, confused terminology

4. **Debugging**: "My React app re-renders constantly and the page is laggy. What's wrong?"
   - Expected: useMemo/useCallback, object reference stability, React.memo, profiler
   - Fail signal: generic "optimize your code" without specifics

5. **Security**: "How do I properly handle file uploads in a web API?"
   - Expected: validate MIME type, limit size, randomize filename, store outside webroot, virus scan
   - Fail signal: missing critical steps, trusting client-side validation

### Domain-Specific Eval (based on $ARGUMENTS)

If evaluating a specialist, add domain-specific prompts:
- **Python specialist**: async context managers, pydantic validation, pytest fixtures
- **Rust specialist**: lifetime elision rules, tokio select!, serde custom deserialize
- **TypeScript specialist**: conditional types, discriminated unions, Next.js ISR
- **Orchestrator**: task routing classification (10 test tasks → correct specialist)

---

## Step 2: Run Evaluation

Spawn harness-tester:

`subagent_type: harness-tester`, `model: opus`

> Evaluate model: $MODEL_NAME
>
> ### Test Prompts
> [Inject eval prompts from Step 1]
>
> ### Execution
> For each prompt:
> 1. Send to model under test, capture full response
> 2. If comparison mode: send same prompt to comparison model
> 3. Record: response time, response length, presence of `<think>` block
>
> ### Testing via Ollama
> ```bash
> curl -s localhost:11434/api/chat -d '{
>   "model": "$MODEL_NAME",
>   "messages": [{"role": "user", "content": "PROMPT_HERE"}],
>   "stream": false
> }' | python3 -c "import sys,json; print(json.load(sys.stdin)['message']['content'])"
> ```
>
> ### Testing via llama.cpp
> ```bash
> curl -s localhost:8080/v1/chat/completions -d '{
>   "messages": [{"role": "user", "content": "PROMPT_HERE"}],
>   "max_tokens": 2048
> }' | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])"
> ```
>
> ### Report Format
> For each prompt:
> ```
> ## Prompt N: [title]
> Question: [the prompt]
> Expected: [key points that should appear]
>
> Model Under Test ($MODEL_NAME):
> Response: [full response]
> Think block: yes/no, length
> Correct: yes/partial/no
> Issues: [specific technical errors]
>
> Comparison ($COMPARISON_MODEL) [if applicable]:
> Response: [full response]
> Correct: yes/partial/no
> ```

---

## Step 3: Judge & Verdict

Orchestrator evaluates results:

### Scoring
- **PASS**: correct answer with genuine reasoning in `<think>` block
- **PARTIAL**: mostly correct but missing key details or minor errors
- **FAIL**: wrong answer, hallucinated facts, or no reasoning

### Verdict Criteria
- **READY**: ≥4/5 standard prompts PASS, no FAILs on critical prompts (race condition, security)
- **NEEDS WORK**: 3/5 PASS, or any critical FAIL — identify what training data to add
- **NOT READY**: <3/5 PASS — model needs more training or larger base

### Report to User
```
## Eval Report: $MODEL_NAME

### Summary
Verdict: READY / NEEDS WORK / NOT READY
Score: N/5 standard prompts passed
Format: <think> blocks present in N/5 responses

### Results
| Prompt | Result | Key Issue |
|--------|--------|-----------|
| Race condition | PASS/FAIL | ... |
| OOMKilled | PASS/FAIL | ... |
| Architecture | PASS/FAIL | ... |
| Debugging | PASS/FAIL | ... |
| Security | PASS/FAIL | ... |

### Comparison (if applicable)
[Side-by-side quality assessment]

### Recommendations
[What training data to add, what to retrain, or ready to deploy]
```

---

## Rules

- Always run the 5 standard eval prompts — they're the quality bar from the 0.8B failure
- Record full responses, not just pass/fail — the reasoning quality matters
- `<think>` block presence is required but not sufficient — content must be correct
- Compare against base model when possible to measure fine-tuning impact
- If model isn't running, report as blocker — don't skip eval
- Domain-specific eval only when testing a specialist model
- Never judge on style — judge on technical correctness and reasoning quality
