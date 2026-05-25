"""F.3d-a — 2-digit hard-row predicate utility + analysis (codex msg
1779701225492 fallback after F.3c acquire 0.8502 < 0.90; slice-split
1779701431738 keeps this to predicate/analysis + tests ONLY — no rung, no
sampler, no trainer CLI).

The predicate `_l0c_is_hard` targets the F.3c step-4000 hole composition, which
was 2-digit-dominated (sampled holes: result-mag 10-99 vs 0-9 = 165:13;
max-operand 10-99 vs 0-9 = 161:17). This test pins the predicate's behavior on
representative rows AND over the full 1255-row exhaustive-L0c support, and checks
the hard/easy partition is consistent with that hole composition. No model/GPU.
"""
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from calm.hrm_text_158.curriculum.generators import _l0c_is_hard  # noqa: E402
from calm.hrm_text_158.curriculum.language_supports import (  # noqa: E402
    build_exhaustive_l0c_supports,
)


# --------------------------------------------------------------------------- #
# Representative unit cases — both the result-2digit and operand-2digit branches
# --------------------------------------------------------------------------- #

def test_hard_result_2digit_single_digit_operands():
    # single-digit operands, 2-digit result -> HARD (result branch).
    assert _l0c_is_hard("5 plus 7 equals what?", 12) is True


def test_hard_operand_2digit_result_1digit():
    # 2-digit operand, 1-digit result -> HARD (operand branch). This is the row
    # that a result-only predicate would miss.
    assert _l0c_is_hard("11 minus 5 equals what?", 6) is True


def test_hard_identity_2digit():
    assert _l0c_is_hard("42 equals what?", 42) is True


def test_easy_all_single_digit_binary():
    assert _l0c_is_hard("3 plus 4 equals what?", 7) is False


def test_easy_identity_single_digit():
    assert _l0c_is_hard("7 equals what?", 7) is False


def test_negative_2digit_result_is_hard():
    # abs() guard: a 2-digit-magnitude negative result is still HARD.
    assert _l0c_is_hard("3 minus 15 equals what?", -12) is True


def test_passthrough_question_without_suffix():
    # Predicate must not choke if the suffix is absent (defensive parse).
    assert _l0c_is_hard("13 plus 2", 15) is True
    assert _l0c_is_hard("1 plus 2", 3) is False


# --------------------------------------------------------------------------- #
# Apply over the EXISTING 1255-row exhaustive-L0c support: hard/easy counts +
# partition correctness + consistency with the step-4000 hole composition.
# --------------------------------------------------------------------------- #

def _all_support_rows():
    rows = []
    for _rung, rs in build_exhaustive_l0c_supports().items():
        rows.extend(rs)
    return rows


def test_support_total_is_1255():
    assert len(_all_support_rows()) == 1255


def test_support_hard_easy_composition_pinned():
    rows = _all_support_rows()
    hard = [(q, e) for (q, e) in rows if _l0c_is_hard(q, e)]
    easy = [(q, e) for (q, e) in rows if not _l0c_is_hard(q, e)]
    # Delivered composition PINNED (the F.3d-a analysis deliverable): the support
    # is overwhelmingly 2-digit, mirroring the F.3c step-4000 hole composition.
    assert len(hard) == 1170
    assert len(easy) == 85
    # F.3d-b carry-forward (NOT decided here): BECAUSE hard is already 93.2% of
    # the support, the F.3d-b sampler must up-weight hard PER ROW (down-weight the
    # 85 easy rows), NOT do literal 70/30 hard/easy POOL sampling — pool-70/30
    # would over-expose the 85 easy rows (~20x/row) and de-emphasize the 1170
    # hard rows. The exact emphasis ratio is deferred to F.3d-b.
    print("L0c-exhaustive support: hard=%d easy=%d (%.1f%% hard)" % (
        len(hard), len(easy), 100.0 * len(hard) / len(rows)))


def test_easy_set_is_exactly_all_single_digit():
    # Predicate correctness over real rows: a row is EASY iff its result is
    # single-digit AND every operand is single-digit. (Cross-check the predicate
    # against an independent definition over the actual support.)
    rows = _all_support_rows()
    for (q, e) in rows:
        expr = q[:-len(" equals what?")] if q.endswith(" equals what?") else q
        operands = []
        for tok in expr.split():
            try:
                operands.append(int(tok))
            except ValueError:
                pass
        indep_easy = abs(int(e)) < 10 and all(abs(o) < 10 for o in operands)
        assert _l0c_is_hard(q, e) == (not indep_easy), "mismatch on %r=%r" % (q, e)


def test_matches_step4000_hole_composition_2digit_dominant():
    # F.3c step-4000 sampled holes were 92.7% 2-digit-result (165/178). The
    # predicate marks every 2-digit-result OR 2-digit-operand row hard, so it
    # captures that dominant stratum; the single-digit minority (~7% of holes)
    # classifies easy by design.
    step4000_hole_shaped = [
        ("47 plus 8 equals what?", 55),    # 2-digit operand + result
        ("9 plus 6 equals what?", 15),     # single-digit operands, 2-digit result
        ("23 minus 4 equals what?", 19),   # 2-digit operand, 2-digit result
        ("10 minus 1 equals what?", 9),    # 2-digit operand, 1-digit result (the watch row's L0c surface)
    ]
    assert all(_l0c_is_hard(q, e) for (q, e) in step4000_hole_shaped)
    # a representative single-digit hole (the minority stratum) is EASY:
    assert _l0c_is_hard("2 plus 3 equals what?", 5) is False


if __name__ == "__main__":
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
            print(f"  {_name}: PASS")
    print("L0c hard-predicate tests: PASS")
