Long-running evals, training runs, quantization jobs, and similar shell
work must stay observable through Monitor. Run the job as the foreground
process in a dedicated shell/session, write stdout/stderr to a log, then
arm Monitor on that log with `bin/watch-wrap`.

Monitor gives streaming progress, failure signatures, heartbeats, replay,
and terminal-condition visibility. Close Monitor when the job reaches its
terminal condition so stale watchers do not pile up.

Two PreToolUse hooks enforce the convention:

- `.claude/hooks/enforce-monitor-on-bg-shell.sh` blocks Bash launch patterns
  that hide long-running work from Monitor: `run_in_background: true`,
  `setsid`, `nohup`, and `until <cond>; do sleep ...; done` poll loops.
- `.claude/hooks/enforce-watch-wrap.sh` blocks raw `tail -f` / `tail -F`
  Monitor commands unless they use `bin/watch-wrap`.

Allowed patterns:

- Ordinary foreground Bash commands.
- Long jobs run in a dedicated foreground shell/session with output
  redirected to a log.
- Continuous `while true; do ...; sleep N; done` poll loops when they
  produce their own stream of events.
- Monitor commands that use `bin/watch-wrap --log <path>` with error,
  progress, success, heartbeat, replay, and stop filters.

Example Monitor command:

```text
Monitor(command="bin/watch-wrap --log /tmp/train.log --heartbeat 180 --error 'Traceback|Error|Killed|OOM|FAILED|assert' --progress 'epoch|eval:|DECISION:' --success 'training complete|DONE' --stop-on 'training complete|DONE' --replay 20")
```
