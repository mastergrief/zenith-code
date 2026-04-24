"""N=5 HumanEvalPlus smoke wrapper.

Sets EVAL_BENCHMARK=humanevalplus + DT_EVAL_N=5, then execs
scripts/dt_install_eval.py in the daemon's globals (m, tok). Required
because bin/gemma-run passes a script path through a named pipe — env
vars don't cross the pipe, so we can't `EVAL_BENCHMARK=... bin/gemma-run`.

Writes to /tmp/he_install_eval_results.json.
"""
import os
os.environ["EVAL_BENCHMARK"] = "humanevalplus"
os.environ["DT_EVAL_N"] = "5"
exec(open("scripts/dt_install_eval.py").read(), globals())
