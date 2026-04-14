"""Family A IR synth eval — exact-match AND functional-correctness gates."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from calm.llm_computer.synth.data import SynthFamilyAGenerator
from calm.llm_computer.synth.infer import (
    SynthFamilyAReasoner, functional_correct,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="calm/hrm/checkpoints/synth_familyA_best.pt")
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--seed", type=int, default=9999)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    if not Path(args.ckpt).exists():
        print(f"ERROR: checkpoint missing: {args.ckpt}", file=sys.stderr)
        sys.exit(1)

    reasoner = SynthFamilyAReasoner(args.ckpt)
    gen = SynthFamilyAGenerator(seed=args.seed)
    samples = gen.generate(args.n)

    exact = 0
    functional = 0
    for s in samples:
        emit = reasoner.predict(s)
        e_ok = emit.replace(" ", "") == s.template.replace(" ", "")
        f_ok = functional_correct(emit, s)
        exact += int(e_ok)
        functional += int(f_ok)
        if args.verbose and not f_ok:
            print(f"  [FAIL] template={s.template!r:14} emit={emit!r:14} "
                  f"query=(a={s.query_a},b={s.query_b})→{s.query_out}")
    print(f"\n[synthA] exact-match: {exact}/{args.n} = {exact/args.n:.1%}")
    print(f"[synthA] functional : {functional}/{args.n} = {functional/args.n:.1%}")


if __name__ == "__main__":
    main()
