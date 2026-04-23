When running evals, training runs etc as shell commands put them in the background ctrl + b and use monitor. A hook will fire if you've done this incorrectly and when the shell/monitor finishes close it to avoid a messy pile-up or staleness. 

Using the monitor allows streaming of progress that allows us to change direction pivot rather than waiting for done on bash and wasting time.

For reference: `.claude\hooks\enforce-watch-wrap.sh`