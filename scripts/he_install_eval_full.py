"""Full N=164 HumanEvalPlus wrapper.

Sets EVAL_BENCHMARK=humanevalplus and DT_EVAL_N=164, then execs
scripts/dt_install_eval.py in the daemon's globals (m, tok).

Writes /tmp/he_install_eval_results.json. Expected runtime ~6-8 hours
on RTX 4070 Laptop (MBPP N=50 took 6h43m; HE+ has 164 problems but
many with larger input corpora — overall runtime similar or slightly
longer due to per-input scoring dominated by sandbox subprocess startup).

Run detached via:
    setsid nohup bin/gemma-run scripts/he_install_eval_full.py \\
        /tmp/he_install_eval_log.txt < /dev/null > /dev/null 2>&1 &
    disown -a
"""
import os
os.environ["EVAL_BENCHMARK"] = "humanevalplus"
os.environ["DT_EVAL_N"] = "164"
exec(open("scripts/dt_install_eval.py").read(), globals())
