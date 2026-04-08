#!/usr/bin/env python3
"""Append hand-written tool-use / investigative training examples.

Targets the 0% tool-use coverage gap identified in the dataset analysis. The
filtered HF examples are pure one-shot Q&A — they teach the model to answer
without investigating, which is the opposite of what a coding agent should do.
These examples model investigation patterns: read first, grep for the thing,
check the logs, interpret the output, then act.

Same format as the other generators: system + user + assistant(<think> + answer).

Run once:
    python3 scripts/generate_tool_use_examples.py
"""

from __future__ import annotations

import json
from pathlib import Path

TARGET = Path("agents/distill/data/coding_reasoning_claude.jsonl")
SYSTEM = "You are a helpful assistant"


def example(user: str, think: str, answer: str) -> dict:
    assistant = f"<think>\n{think.strip()}\n</think>\n\n{answer.strip()}"
    return {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user.strip()},
            {"role": "assistant", "content": assistant},
        ]
    }


EXAMPLES: list[dict] = []


# ── Investigation patterns ────────────────────────────────────────

EXAMPLES.append(example(
    user="""My production pod keeps OOMKilling. How do I figure out what's actually using the memory?""",
    think="""Don't guess at memory issues — measure. The investigation chain is:

1. Confirm it's actually OOMKill (not some other restart cause) → kubectl describe pod
2. Check memory history right before the kill → metrics if available
3. Get a heap snapshot from the running process → pprof for Go, jmap for Java, py-spy / tracemalloc for Python
4. If you can't reproduce, capture next time it happens → preStop hook to dump heap before SIGKILL

Don't bump the limit as a first response — that just delays the real issue and burns more memory.""",
    answer="""Don't guess — investigate in this order:

**1. Confirm it's actually OOMKilled (not some other crash):**
```bash
kubectl describe pod <pod-name> | grep -A 5 "Last State"
```
Look for `Reason: OOMKilled` and `Exit Code: 137`. If you see something else (segfault, exit 1), the problem is different.

**2. Get the memory history right before the kill:**
```bash
# If you have metrics-server
kubectl top pod <pod-name> --containers

# If you have Prometheus, query container_memory_working_set_bytes for the pod
# over the last hour with 30s granularity
```

**3. Get a heap snapshot from a live instance.** Pick the language:

**Go:** import `net/http/pprof` and curl the heap endpoint:
```bash
kubectl port-forward pod/<pod-name> 6060:6060
go tool pprof http://localhost:6060/debug/pprof/heap
# Then in pprof: top, list <funcname>, web
```

**Java:** jmap from inside the container:
```bash
kubectl exec <pod-name> -- jmap -dump:live,format=b,file=/tmp/heap.hprof 1
kubectl cp <pod-name>:/tmp/heap.hprof ./heap.hprof
# Open in Eclipse MAT or VisualVM
```

**Python:** use `tracemalloc` (built in) or `py-spy` (no code change needed):
```bash
kubectl exec <pod-name> -- pip install py-spy
kubectl exec <pod-name> -- py-spy dump --pid 1
```

**Node.js:** `--inspect` and Chrome DevTools Memory tab, or `heapdump` package.

**4. If it dies before you can grab a snapshot**, dump heap on SIGTERM:

```yaml
lifecycle:
  preStop:
    exec:
      command:
      - /bin/sh
      - -c
      - "jmap -dump:live,format=b,file=/dumps/heap.hprof 1 || true"
volumeMounts:
- name: dumps
  mountPath: /dumps
volumes:
- name: dumps
  emptyDir: {}
```

Then `kubectl cp` the dump out before the pod is replaced.

**Don't just bump the memory limit as a fix.** That delays the real problem and you'll hit it again at scale. Find the actual leak first."""
))

EXAMPLES.append(example(
    user="""Where is the function `process_payment` defined and called in this codebase? I'm new to it.""",
    think="""Two distinct questions: where is it defined (one location, the source of truth) and where is it called (potentially many — the impact set).

For "defined", grep for `def process_payment` or `function process_payment` or `process_payment =` depending on language. If unsure, just `grep -n "process_payment("` and then look for the line that's an assignment vs a call.

For "called", grep for `process_payment(` everywhere. The trailing paren disambiguates from comments and similar identifiers.

Use ripgrep (`rg`) — much faster than `grep -r`, respects `.gitignore`. If the codebase has language servers (LSP) configured, "find references" is more accurate than text grep but text grep is universal and fast.""",
    answer="""Two greps, two answers:

**Find the definition** (typically one location):

```bash
# Python
rg -n "def process_payment\b"

# JavaScript / TypeScript
rg -n "function process_payment\b|process_payment\s*=\s*(\(|function|async)"

# Rust / Go / Java / C
rg -n "fn process_payment\b|func process_payment\b"

# Language-agnostic fallback — will catch defs and assignments
rg -n "^\s*\w*\s*process_payment" --type-add 'src:*.{py,js,ts,go,rs,java}' --type src
```

**Find the call sites** (could be many):

```bash
# The trailing `(` filters out comments and similar identifiers
rg -n "process_payment\("

# Want only callers, not the definition itself? Pipe through grep -v
rg -n "process_payment\(" | grep -v "def process_payment"
```

**Then read the definition:**

```bash
# Once you have the file:line, read just the function (typically 30-50 lines)
sed -n '120,170p' src/payments/processor.py
```

**For complex investigations, an LSP-aware tool is more accurate** than text grep — it handles renames, qualified names, and import aliases. If your editor has "find references" (VS Code, JetBrains, vim+coc), use it. But text grep is universal, fast, and works without setup.

**One nuance:** if `process_payment` is exported and re-exported, callers may import it under a different name. Check `import` / `require` / `use` statements:

```bash
rg -n "import.*process_payment|from.*process_payment"
```

**Rule of thumb:** define-grep is one query and the answer is at most a handful of lines. Call-grep gives you the impact set — start with `wc -l` of the results to know if it's "5 places" or "200 places", which dictates how careful you need to be."""
))

EXAMPLES.append(example(
    user="""My test passes locally but fails in CI. Where do I start?""",
    think="""Local-passes-CI-fails almost always comes down to one of:

1. **Environment difference**: missing env var, different config file, different secret, different DB seed
2. **Filesystem difference**: paths, file permissions, line endings, missing fixture
3. **Time/timezone**: tests sensitive to TZ, DST boundaries, clock drift
4. **Concurrency**: tests that pass when run alone but fail when parallel; flaky tests with race conditions
5. **Dependency version drift**: lockfile vs installed; or worse, no lockfile and CI got newer versions
6. **Test ordering**: dirty state from a previous test; random seed differences
7. **Resource limits**: CI has less RAM/CPU; OOM or timeout
8. **External services**: localhost works, CI can't reach the network

Investigation: get the actual failure output from CI first, don't guess. Then narrow with the question — what is different between my machine and CI?""",
    answer="""Don't guess — pull the actual CI output and narrow down systematically.

**Step 1: Get the real failure**

```bash
# Pull the CI logs
gh run view --log-failed
# Or for a specific run:
gh run view <run-id> --log-failed

# Or if you have the URL
gh run view <run-id> --log | grep -A 20 "FAILED\|Error\|Traceback"
```

Read the actual stack trace. Don't theorize about what's failing — find out what IS failing.

**Step 2: Reproduce locally**

The most useful thing you can do is reproduce the failure on your machine. Try in this order (cheapest first):

```bash
# 1. Run the EXACT same test command CI runs
# Check .github/workflows/*.yml for the test command and copy it verbatim

# 2. Run with the same env vars as CI
CI=true NODE_ENV=test pytest tests/test_thing.py -v

# 3. Run in a Docker container that mirrors CI's environment
docker run --rm -v $(pwd):/app -w /app python:3.11-slim bash -c "
  pip install -r requirements.txt && pytest tests/test_thing.py
"

# 4. If you have nektos/act, run the actual workflow locally
act -j test
```

**Step 3: Narrow with the diff checklist**

If the test still passes locally, it's an environment delta. Check each:

| Check | Command |
|---|---|
| Python/Node version | `python --version` vs CI logs |
| Installed package versions | `pip freeze` vs CI lockfile install |
| Env vars | `env | sort` — compare to CI env |
| Timezone | `date` and `echo $TZ` |
| File permissions | `ls -la` on test fixtures |
| Random seed | does the test use one? CI may seed differently |
| Test isolation | run with `pytest -p no:randomly` to remove ordering randomness |

**Step 4: Common gotchas to check**

```bash
# Line endings (Windows → Unix)
file tests/fixtures/*.txt

# File permissions
ls -la tests/fixtures/

# Hardcoded paths
rg -n "/Users/|C:\\\\|/home/$USER" tests/

# Timezone-sensitive code
rg -n "datetime.now\(\)|new Date\(\)" tests/ src/

# Test ordering dependencies (pollution from previous tests)
pytest tests/test_thing.py  # runs alone — does it pass?
```

**Step 5: If you still can't reproduce**

Add diagnostics directly to the test that runs in CI:

```python
def test_thing():
    import os, sys, platform
    print(f"Python: {sys.version}", flush=True)
    print(f"Platform: {platform.platform()}", flush=True)
    print(f"Env: {sorted(os.environ.items())}", flush=True)
    print(f"CWD: {os.getcwd()}", flush=True)
    print(f"Files: {os.listdir('.')}", flush=True)
    # ... your test
```

Push to a branch, let CI run, read the diagnostic output. The delta will jump out.

**Rule of thumb:** local-passes-CI-fails is always a real difference, not a flake. The investigation is "what's different between my machine and CI?" — answer that and the bug is obvious."""
))

EXAMPLES.append(example(
    user="""Production is slow. Users complaining. I have no idea where to start. What do I do?""",
    think=""""Production is slow" needs to be turned into a measurable hypothesis. The investigation goes from broad to narrow:

1. **Confirm the symptom**: which endpoint? all of them or specific ones? slow responses or hung connections? user-facing or background jobs?
2. **Layer attribution**: client (browser, mobile), CDN/edge, load balancer, application, database, downstream services. Where does the latency live?
3. **Time correlation**: did it just start? gradual? specific time? matches a deploy?
4. **Change scope**: deploys, infra changes, traffic spikes, downstream incidents
5. **Drill in to the bottleneck**: once you know "DB slow" or "specific endpoint", investigate THAT layer

The mistake is firing off random fixes (restart pods, bump memory, scale up) before knowing what's actually slow. That sometimes works by accident but doesn't teach you anything and the problem comes back.""",
    answer="""**Stop, measure, then act.** Random fixes (restart, scale up, bump memory) sometimes work by accident but the problem comes back.

**Step 1: Define "slow" precisely (1-2 minutes)**

Open your monitoring dashboard and answer:
- **Which endpoints?** All of them, or specific ones? `/api/checkout` is different from `/api/static-asset`.
- **What's the latency?** p50, p95, p99. "Slow" usually means p99 spiked while p50 stayed flat — long tail.
- **When did it start?** Suddenly at 14:23, or gradual over days? Sudden = recent change. Gradual = capacity or accumulating leak.
- **How many users affected?** Everyone or a subset?

```
# Example characterization (write this down):
# /api/checkout p99 latency: 200ms → 8000ms starting at 14:23 today
# /api/products: unchanged (50ms p99)
# Affects all users
# No client-side errors, just slow responses
```

**Step 2: Check the recent changes (1-2 minutes)**

```bash
# Did we deploy?
git log --since="2 hours ago" --oneline

# Any infra/config changes?
# Check your deploy tool history (k8s, terraform, helm, etc)
kubectl rollout history deployment/api -n prod | head

# Any incidents at upstream providers?
# Statuspages: AWS, GCP, Stripe, Datadog, etc.
```

If a deploy went out at 14:20, you have your suspect. Roll it back if you can:
```bash
kubectl rollout undo deployment/api -n prod
```
If latency drops back to normal, you've confirmed the cause. NOW investigate WHAT in that deploy caused it (don't skip this — you need to fix forward).

**Step 3: Layer attribution (5-10 minutes)**

Find where the latency lives:

| Layer | Check |
|---|---|
| Client/CDN | Compare server-reported latency vs client-perceived (RUM, browser dev tools) |
| Load balancer | LB metrics: request count, target latency, 5xx rate |
| App server | APM tool — pick the slow endpoint, look at the trace breakdown |
| Database | DB metrics — slow query log, active connections, lock waits, IOPS |
| Downstream API | APM downstream span, or grep your logs for upstream call latency |
| Cache | Hit ratio, eviction rate |

The APM trace is the most useful tool here — it shows you "request entered at T0, hit DB at T+10ms, DB returned at T+7800ms, response sent at T+7810ms" → DB is the bottleneck.

**Step 4: Drill into the bottleneck**

If DB is slow:
```sql
-- Postgres: which queries are slow right now?
SELECT pid, now() - query_start AS duration, state, query
FROM pg_stat_activity
WHERE state != 'idle'
ORDER BY duration DESC LIMIT 10;

-- Slow query log
SELECT query, calls, mean_exec_time, max_exec_time
FROM pg_stat_statements
ORDER BY mean_exec_time DESC LIMIT 10;
```

If specific endpoint is slow:
```bash
# Look at the trace for one slow request
# Then trace what changed in that code path
git log --since="1 day ago" -- src/api/checkout/
```

If it's resource exhaustion:
```bash
kubectl top pods -n prod --sort-by=cpu
kubectl top pods -n prod --sort-by=memory

# Connections from app to DB
psql -c "SELECT count(*), state FROM pg_stat_activity GROUP BY state;"
```

**Step 5: Mitigate, then fix**

Sometimes you need to mitigate before you fully understand the root cause (users are angry, the call is escalating). Acceptable mitigations IF they're temporary:
- **Rollback** the suspect deploy
- **Scale up** if obviously resource-bound
- **Cache** the slow endpoint if it's read-only and tolerates staleness
- **Disable** a noncritical feature if you can isolate it

Then go back and find the root cause once the immediate fire is out. Write a postmortem.

**Don't:**
- Restart pods without knowing why ("turning it off and on again" hides the symptom)
- Bump memory limits without evidence of OOMs (delays the real issue)
- Add a cache without understanding what was slow (caches mask real problems)
- Blame "the network" without measuring (rarely actually the network)

**Rule of thumb:** every minute spent characterizing the problem saves ten minutes of guessing. Get a number, get a layer, get a timeframe, then act."""
))

EXAMPLES.append(example(
    user="""I want to know what changed in this file in the last week and who changed it. The file is `src/billing/invoice.py`.""",
    think="""Standard `git log` and `git blame` workflow. Three useful queries:

1. Commits touching the file in the last week → `git log --since="1 week ago" -- path`
2. Per-line authorship → `git blame path`
3. Diff of changes in the last week → `git log -p --since="1 week ago" -- path`

Also useful: `git log --follow` if the file was renamed.""",
    answer="""**Three git commands cover this:**

**1. Commits touching the file in the last week:**

```bash
git log --since="1 week ago" --oneline -- src/billing/invoice.py
```

Add `--stat` to see lines added/removed per commit:
```bash
git log --since="1 week ago" --stat -- src/billing/invoice.py
```

Add author info:
```bash
git log --since="1 week ago" --pretty=format:"%h %an %ar  %s" -- src/billing/invoice.py
```

**2. Who currently owns each line:**

```bash
git blame src/billing/invoice.py
```

That's noisy for big files. Filter to the function you care about:
```bash
git blame -L 120,180 src/billing/invoice.py    # lines 120-180
git blame -L '/def calculate_tax/,/^def /' src/billing/invoice.py    # the calculate_tax function
```

**3. Actual diffs from the last week:**

```bash
# All changes in the last week, with diffs
git log -p --since="1 week ago" -- src/billing/invoice.py

# Just the cumulative diff between "1 week ago" and now
git diff HEAD@{"1 week ago"} HEAD -- src/billing/invoice.py
```

**If the file was renamed**, you'll miss history. Use `--follow`:

```bash
git log --follow --since="1 week ago" -- src/billing/invoice.py
```

`--follow` tracks the file across renames. (It only works on a single path, though — no globs.)

**Find changes by a specific person:**

```bash
git log --since="1 week ago" --author="alice" -- src/billing/invoice.py
```

**Find changes that mention an issue or PR:**

```bash
git log --since="1 week ago" --grep="JIRA-1234" -- src/billing/invoice.py
```

**Compare two specific commits:**

```bash
git diff abc1234 def5678 -- src/billing/invoice.py
```

**Rule of thumb:** `git log` for "what happened", `git blame` for "who and why", `git diff` for "what's different between two points". Add `-- path` to scope to a file."""
))

EXAMPLES.append(example(
    user="""Service A in my k8s cluster can't reach service B. Both are in the same namespace. How do I debug this?""",
    think="""K8s service-to-service connectivity issues come down to a checklist:

1. DNS resolution — does service B's name resolve to a ClusterIP?
2. Endpoints — does the Service have any ready pods backing it?
3. Network policy — is there a NetworkPolicy blocking traffic?
4. Port match — is service B listening on the port the Service exposes?
5. Pod readiness — are B's pods passing readiness probes?
6. App-level — is service B's app actually accepting connections?

Walk through each in order. The diagnostic is mostly `kubectl exec` from inside service A's pod plus some `kubectl get` queries.""",
    answer="""Walk through this checklist in order — the issue is almost always one of these.

**Step 1: Get a shell inside service A's pod**

```bash
kubectl exec -it -n <namespace> <service-a-pod> -- sh
# (or 'bash' if the image has it; some minimal images don't)
```

If the image is distroless and has no shell, use a debug container:
```bash
kubectl debug -it <service-a-pod> --image=nicolaka/netshoot --target=<container>
```

`netshoot` has every networking tool you could want.

**Step 2: DNS resolution**

```bash
# Inside service A's pod
nslookup service-b
# Or: getent hosts service-b
# Or: dig service-b.<namespace>.svc.cluster.local
```

Expected: returns the ClusterIP of service-b.

**Failures and what they mean:**
- `NXDOMAIN` → service-b doesn't exist (typo? wrong namespace? not deployed?)
- `SERVFAIL` → CoreDNS issue (`kubectl get pods -n kube-system | grep coredns`)
- Wrong IP → maybe an old Service object or a typo in your service name

**Step 3: Verify the Service has endpoints**

```bash
kubectl get endpoints service-b -n <namespace>
```

Expected: lists IP:port pairs of the backing pods.

**`<none>` means the service has zero ready pods backing it.** Common causes:
- Pods aren't passing the readiness probe
- The Service's `selector` doesn't match any pod labels
- The pods are in a different namespace than the Service

```bash
# Check pod labels match Service selector
kubectl get pods -n <namespace> --show-labels
kubectl get service service-b -n <namespace> -o jsonpath='{.spec.selector}'

# Check pod readiness
kubectl get pods -n <namespace> -l app=service-b
# Look at the READY column. 0/1 means readiness probe failing.

kubectl describe pod <service-b-pod> -n <namespace> | grep -A 5 Readiness
```

**Step 4: Try the connection from inside service A**

```bash
# TCP test (no app-layer)
nc -zv service-b 80

# HTTP test
curl -v http://service-b:80/health
# Or just: wget -qO- http://service-b/

# Time it to see if it's a timeout vs immediate refusal
time curl http://service-b/
```

**Read the error:**
- `Connection refused` → DNS resolved, IP is right, but no process is listening on that port. Service B's container might have crashed or be listening on a different port.
- `Connection timed out` → packets are being dropped. Network policy, firewall, or wrong port.
- `No route to host` → routing/CNI issue. Rare; usually a cluster-level problem.

**Step 5: Check NetworkPolicy**

```bash
kubectl get networkpolicy -n <namespace>
kubectl describe networkpolicy <name> -n <namespace>
```

If there are NetworkPolicies, they default-deny ingress to selected pods. You may need to allow service A explicitly:

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-a-to-b
spec:
  podSelector:
    matchLabels:
      app: service-b
  policyTypes:
  - Ingress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: service-a
    ports:
    - protocol: TCP
      port: 80
```

**Step 6: Verify port mapping**

The Service `port` and pod `containerPort` must match what the app actually listens on:

```bash
kubectl get service service-b -n <namespace> -o yaml | grep -A 3 ports:
kubectl get pod <service-b-pod> -n <namespace> -o yaml | grep -A 3 containerPort:
```

If `Service.spec.ports[0].targetPort` is 8080 but the app listens on 8000, you get `Connection refused`. Fix the Service or the app.

**Step 7: From the pod itself, is the app accepting connections?**

```bash
# Exec into service B
kubectl exec -it <service-b-pod> -n <namespace> -- sh

# Inside the pod
netstat -tlnp 2>/dev/null || ss -tlnp
# Look for your port. If nothing is listening on it, the app isn't bound correctly.
# Common cause: app bound to 127.0.0.1 instead of 0.0.0.0

curl http://localhost:8080/health
# If this works inside the pod but not from service A, it's a network/port problem.
# If this fails inside the pod, the app itself is broken.
```

**Rule of thumb:** debug from inside-out — start at DNS (does the name resolve?), then endpoints (is anything backing the Service?), then connectivity (can I open a TCP socket?), then app (is it actually responding?). Each step rules out a layer."""
))


EXAMPLES.append(example(
    user="""Something broke between v1.4.0 and HEAD. There are 200 commits in between. How do I find which one introduced the bug?""",
    think="""git bisect is exactly the tool for this. Binary search through commits, with the user (or a script) marking each one as good or bad. log2(200) ≈ 8 steps to find the culprit.

Two flavors:
1. Manual: `git bisect start`, `git bisect bad`, `git bisect good v1.4.0`, then test each commit and mark good/bad
2. Automated: `git bisect run <test_command>` — script returns 0 (good) / non-0 (bad), git automates the search

The automation is much better when you have a reliable repro. Even a flaky 80% repro is faster automated than manual.""",
    answer="""**`git bisect`** — binary search across the commit range. log2(200) ≈ 8 steps to find it.

**Manual bisect:**

```bash
git bisect start
git bisect bad HEAD                    # current state is broken
git bisect good v1.4.0                  # last known good

# Git checks out the midpoint
# Test it (run your repro)
# Then mark it
git bisect good   # ...if it works
# OR
git bisect bad    # ...if it's broken

# Repeat until git tells you the first bad commit
git bisect reset    # cleanup
```

**Automated bisect (much better if you have a repro script):**

```bash
git bisect start
git bisect bad HEAD
git bisect good v1.4.0

# Provide a script that returns 0=good, non-zero=bad
git bisect run ./repro.sh
```

The script can be anything that exits 0 or non-0:

```bash
#!/bin/bash
# repro.sh
make build || exit 125    # 125 = "skip this commit, can't test"
./run_test.sh
# exit code propagates: 0 = test passed (good), 1+ = test failed (bad)
```

**Special exit codes:**
- `0` → good
- `1-124` → bad
- `125` → can't test this commit (broken build, etc.) — git skips it
- `>= 128` → abort the bisect entirely

**Tips:**

**Make the test fast.** A test that takes 5 minutes × 8 commits = 40 minutes total. Worth optimizing the repro.

**Confine the search to relevant files** if you suspect a subsystem:
```bash
git bisect start -- src/billing/ tests/billing/
```
Only commits that touched those paths get tested.

**Skip merge commits** if they make bisecting confusing:
```bash
git bisect start --first-parent
```

**Save the log so you can resume:**
```bash
git bisect log > bisect.log
# ... if you screw up ...
git bisect reset
git bisect replay bisect.log
```

**When the test is flaky:** repeat the test multiple times in your script and only mark good if all pass:
```bash
#!/bin/bash
for i in 1 2 3; do
  ./run_test.sh || exit 1
done
```

**Result:**

```
abc1234 is the first bad commit
commit abc1234
Author: alice <alice@example.com>
Date:   ...
    refactor: switch to new payment provider SDK
```

Now you have a SHA, an author, and a commit message. Read the diff (`git show abc1234`) to understand what broke.

**Rule of thumb:** if you have a deterministic repro and a commit range, bisect is always faster than guessing. The investment is writing the repro script — that one-time cost saves you hours of theory."""
))

EXAMPLES.append(example(
    user="""My long-running Node service slowly grows from 200MB to 2GB over 3 days. Then we restart it. How do I find the leak?""",
    think="""Slow memory growth over days = leak somewhere holding references. The investigation:

1. Confirm it's a real leak vs. expected growth (caches, buffers warming up)
2. Take heap snapshots at two points in time → diff them → see what's growing
3. The diff highlights "retainers" — objects that hold others alive
4. From the retainers, trace back to your code

Tools:
- Chrome DevTools heap snapshot (most usable)
- `--inspect` flag + DevTools, OR `heapdump` package programmatically
- `clinic.js heapprofiler` for production

Two snapshots is the key technique. One snapshot is hard to interpret (most stuff is "normal"). Two snapshots taken hours/days apart show what's accumulating.""",
    answer="""**Take two heap snapshots hours apart, diff them, look at what grew.** A single snapshot is hard to interpret because most of what you see is normal program state. The diff highlights what's leaking.

**Step 1: Enable heap dumps in your service**

```js
// At app startup
const v8 = require('v8');
const fs = require('fs');

// Trigger via signal: kill -USR2 <pid>
process.on('SIGUSR2', () => {
  const filename = `/tmp/heap-${Date.now()}.heapsnapshot`;
  const stream = v8.getHeapSnapshot();
  stream.pipe(fs.createWriteStream(filename));
  console.log(`Heap snapshot written to ${filename}`);
});
```

**Step 2: Take the first snapshot when the service is "warm"**

Wait until the service has been running an hour or so (initial warmup is done).

```bash
kubectl exec <pod> -- kill -USR2 1
kubectl cp <pod>:/tmp/heap-1234567.heapsnapshot ./heap-baseline.heapsnapshot
```

**Step 3: Wait for memory to grow noticeably (a few hours)**

Watch with `kubectl top pod` or your APM. When RSS has grown ~30%+, take the second snapshot:

```bash
kubectl exec <pod> -- kill -USR2 1
kubectl cp <pod>:/tmp/heap-7654321.heapsnapshot ./heap-leaked.heapsnapshot
```

**Step 4: Diff in Chrome DevTools**

1. Open Chrome → DevTools → Memory tab
2. Click "Load" and load `heap-baseline.heapsnapshot`
3. Click "Load" again and load `heap-leaked.heapsnapshot`
4. Switch the snapshot dropdown to the second one
5. Change "Summary" → **"Comparison"** dropdown
6. Set the comparison base to the first snapshot

You now see a list of object types sorted by "Delta" (count change since the baseline). The leak is at the top.

**Step 5: Identify the retainer**

Click an object type with a high positive delta. Expand a few instances. Look at the **"Retainers"** panel at the bottom — this shows the chain of references keeping the object alive.

The chain looks like:
```
Array @123456 (the leaked object)
  ↑ context  →  Closure @789012
  ↑ requestQueue  →  PaymentProcessor @345678
  ↑ globalProcessor  →  global
```

That tells you `requestQueue` on `PaymentProcessor` is appending and never clearing.

**Common Node.js leak patterns to look for:**

| Pattern | Symptom |
|---|---|
| Unbounded array push | `Array` instances growing, retained by your singletons |
| Event listener leak | `EventEmitter` with high listener count; warning in logs |
| Closure capturing large object | Function closures retaining `req`/`res` after handler |
| Timer never cleared | `Timeout` instances accumulating |
| Cache without eviction | `Map` or `Object` with growing entries |
| Global registries | Anything assigned to `global.X` that grows over time |

**Step 6: Validate the fix**

After fixing, run the same two-snapshot diff and confirm the type that was growing is now stable.

**For production analysis without code changes**, use `clinic.js`:

```bash
clinic heapprofiler -- node server.js
# Run for an hour, then stop with Ctrl+C
# Generates an HTML report with growth attribution
```

Or if you want continuous profiling, services like Datadog, Pyroscope, and Sentry have heap profilers that sample in production.

**One trap:** "Memory keeps growing" isn't always a leak. Some legitimate growth:
- Initial cache warming (stabilizes after a while)
- JIT compilation adds inline caches
- The garbage collector is lazy and only triggers on pressure

The two-snapshot diff over a long enough window cuts through this — anything that's growing across a 4+ hour gap is suspicious.

**Rule of thumb:** memory leak debugging is "two snapshots, one diff, one retainer chain." Don't try to interpret a single snapshot — too noisy."""
))

EXAMPLES.append(example(
    user="""I need to know where the value of `DATABASE_URL` is actually coming from in my running service. There are .env files, k8s ConfigMaps, terraform, secret managers... how do I trace it?""",
    think="""Configuration provenance is a real pain when there are layers. The investigation goes from "what does the process see right now" backward through the layers:

1. Inside the running container: what's the actual env var value? (`printenv`)
2. Where did the orchestrator inject it from? (Pod YAML — env vs envFrom)
3. If from envFrom, what's in the source ConfigMap/Secret?
4. What deployed that ConfigMap/Secret? (CI/CD logs, terraform state, ArgoCD)
5. What's in the source of truth (terraform / vault / SOPS file)?

Each layer can be checked independently. Start at the running process (truth at runtime) and walk backward.""",
    answer="""**Walk backward from the running process** — start with the truth (what the process actually sees), then trace through each injection layer.

**Step 1: What does the process actually see right now?**

```bash
# From inside the container
kubectl exec <pod-name> -- printenv DATABASE_URL

# Or: full env
kubectl exec <pod-name> -- printenv | sort
```

That gives you the current value. If it's wrong, the rest of the trace tells you where to fix it.

**Step 2: Where did k8s inject it?**

```bash
kubectl get pod <pod-name> -o yaml > pod.yaml
```

Look at `spec.containers[].env` and `spec.containers[].envFrom`:

```yaml
env:
- name: DATABASE_URL
  valueFrom:
    secretKeyRef:
      name: prod-db-secret    # ← SOURCE
      key: url

envFrom:
- configMapRef:
    name: app-config          # ← OR a ConfigMap source
- secretRef:
    name: app-secrets         # ← OR a Secret source (all keys imported)
```

Three possibilities:
1. **Hardcoded `value:`** → the value is in the pod spec itself; trace back to whatever generated the spec (helm, kustomize, plain manifest)
2. **`secretKeyRef` / `configMapKeyRef`** → look up that named resource for the value
3. **`envFrom` with a SecretRef/ConfigMapRef** → ALL keys from that resource were imported

**Step 3: Look up the source resource**

```bash
# If from a Secret
kubectl get secret prod-db-secret -o yaml
# Decode the base64 value
kubectl get secret prod-db-secret -o jsonpath='{.data.url}' | base64 -d

# If from a ConfigMap
kubectl get configmap app-config -o yaml
```

**Step 4: Where did THAT resource come from?**

ConfigMaps and Secrets in production are usually managed by something:

| Source | How to check |
|---|---|
| Helm | `helm get values <release-name>` |
| Kustomize | `kustomize build overlays/prod | grep -A 5 prod-db-secret` |
| ArgoCD | ArgoCD UI shows the source repo + path |
| Terraform | `terraform state show kubernetes_secret.prod_db` |
| External Secrets Operator | `kubectl get externalsecret -A` then check the SecretStore |
| Sealed Secrets | `kubectl get sealedsecret -A` (the encrypted YAML is in git) |
| `kubectl apply -f` directly | check git history for the YAML file |

**Step 5: What's in the source of truth?**

If the chain leads to a vault/secrets manager:

```bash
# AWS Secrets Manager
aws secretsmanager get-secret-value --secret-id prod/db_url --query SecretString --output text

# HashiCorp Vault
vault kv get secret/prod/db_url

# GCP Secret Manager
gcloud secrets versions access latest --secret=prod-db-url
```

**Step 6: Walk it backward in code**

If you want to reproduce the layering for documentation:

```
[Running Pod]
    ↑ env DATABASE_URL
    ↑ secretKeyRef: prod-db-secret/url
    ↑ Secret prod-db-secret
    ↑ Created by: ExternalSecret prod-db (External Secrets Operator)
    ↑ Source: AWS Secrets Manager arn:aws:secretsmanager:...:secret:prod/db_url
    ↑ Updated by: terraform module .infrastructure/secrets/main.tf line 47
    ↑ Variable source: TF_VAR_prod_db_url (from CI secrets)
```

That's the full trace from "what the process sees" to "where a human sets it."

**Useful commands for this:**

```bash
# Find all references to the env var in your manifests
rg "DATABASE_URL" --type yaml

# Find all references in code
rg "DATABASE_URL|os\.environ\[.DATABASE_URL.\]"

# Find the most recent change to whatever defines it
git log --since="1 month ago" -p -- path/to/values.yaml
```

**Common gotchas:**

- **Multiple sources** — env can be set in `env:` AND in `envFrom:`. The `env:` value wins.
- **Stale ConfigMap** — k8s injects env vars at pod creation. If the ConfigMap was updated after the pod started, the pod still has the OLD value. Confirm with `printenv`.
- **`envFrom` order matters** — later sources override earlier ones with the same key.
- **Helm template vs deployed value** — `helm template` shows what would be deployed; `helm get values` shows what was actually deployed last.

**Rule of thumb:** start at `kubectl exec ... printenv` for the truth, walk backward through `pod.yaml → secret/configmap → manager`. Document the chain so the next person doesn't have to do this again."""
))

EXAMPLES.append(example(
    user="""A file on the server is locked / can't be deleted. lsof says nothing. How do I find what's holding it?""",
    think="""`lsof` is the standard tool but has limits:
- Doesn't see kernel-held references
- May not see other namespaces (containers)
- Permissions: needs root for full visibility
- Doesn't show memory-mapped files unless you specifically look

If lsof shows nothing, alternatives:
- `fuser -m <file>` — checks /proc for any process with the file mapped
- `lsof | grep <file>` from the HOST, not inside a container
- `find /proc/*/fd -lname '*<filename>*' 2>/dev/null` — scan all file descriptors
- Check if the file is bind-mounted somewhere (`mount | grep <file>`)
- Check if it's held by a deleted-but-still-referenced inode (deleted while open)
- Check NFS / fuse — different rules

Container case is most common: a process inside a container has the file open, and lsof on the host doesn't see it because of mount namespaces.""",
    answer="""**lsof has blind spots.** Try this in order:

**Step 1: lsof with sudo** (catches permission issues):

```bash
sudo lsof /path/to/file
# Or by directory
sudo lsof +D /path/to/dir
# Or by file substring
sudo lsof | grep filename
```

**Step 2: `fuser` — checks /proc directly:**

```bash
sudo fuser -mv /path/to/file
# -m: any process with the file mapped (mmap counts)
# -v: verbose, shows pid, user, command

# Or for the whole filesystem mountpoint
sudo fuser -mv /var/lib/some-mount/
```

**Step 3: scan /proc/*/fd directly:**

```bash
sudo find /proc/*/fd -lname '*filename*' 2>/dev/null
# Each result is /proc/<pid>/fd/<fd> → ../../actual/path
# The pid is in the path
```

To get more context:
```bash
sudo find /proc/*/fd -lname '*filename*' 2>/dev/null | while read fd; do
  pid=$(echo $fd | cut -d/ -f3)
  echo "$pid $(readlink $fd) $(cat /proc/$pid/comm 2>/dev/null)"
done
```

**Step 4: check memory-mapped files** (mmap'd files don't show in standard lsof):

```bash
sudo grep -l filename /proc/*/maps 2>/dev/null
# Each result is /proc/<pid>/maps; the pid is in the path
```

**Step 5: container case** — the holder might be inside a container, invisible to host lsof unless you cross namespaces:

```bash
# Find processes inside containers
sudo ls -la /proc/*/ns/mnt | sort -k 11
# Group by mnt namespace inode

# Check inside a specific container
docker exec <container-id> lsof /path/in/container/file
# Or for k8s
kubectl exec <pod> -- lsof /path
```

**Step 6: deleted-but-still-open files**:

```bash
sudo lsof +L1
# +L1 = files with link count < 1 = deleted but still open
```

These are files where the directory entry was removed but a process still holds them open. They consume disk space and won't actually be freed until the process closes the fd or exits. Common cause of "df shows space used but du shows nothing."

**Step 7: bind mounts and overlays**:

```bash
mount | grep filename
findmnt -T /path/to/file
```

If the file is bind-mounted from elsewhere, the "real" path might be different. Anyone holding the OTHER path is holding this one.

**Step 8: NFS / FUSE / network filesystems**:

These don't always play by local-process rules. For NFS:
```bash
showmount -a <nfs-server>
```

shows clients with active mounts. The lock might be on a different machine.

**Step 9: kernel-held references** (rare but possible):

```bash
# Files held by kernel modules — usually not lsof-visible
sudo lsmod
sudo cat /proc/locks | grep <inode>

# Inode of the file
ls -i /path/to/file
```

**Common scenarios and what "nothing in lsof" usually means:**

- **File on a bind-mounted path** → holder is using the other path
- **Inside a container** → host lsof doesn't see container processes
- **Memory-mapped (mmap)** → standard lsof view misses it; use `fuser -m`
- **Deleted while open** → use `lsof +L1`
- **Wrong user** → run lsof with sudo
- **NFS lock** → check the NFS server, not this machine

**Rule of thumb:** when lsof says nothing, expand the search: try `fuser -m`, scan `/proc/*/fd`, check for containers, check for mmap, check for bind mounts. One of those usually reveals the holder."""
))


def main() -> None:
    if not TARGET.exists():
        raise SystemExit(f"Target file not found: {TARGET}")
    with TARGET.open("a", encoding="utf-8") as f:
        for ex in EXAMPLES:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"Appended {len(EXAMPLES)} examples to {TARGET}")


if __name__ == "__main__":
    main()
