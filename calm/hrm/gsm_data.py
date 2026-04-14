"""GSM-style word problems — narrative pressure test of the CRLM scaling law.

Longer templates than `word_data.py`: multi-sentence setups, subordinate
clauses ('after she gave', 'before he found'), pronoun chains spanning
2-3 sentences, per-unit arithmetic ('each'), 3- and 4-term expressions.

Max sentence length ~120 chars (vs 78 in word_data). Same tokenizer
(char-level) and same structure-only training target. CRLM claim: the
48K HRM should still handle this because the output language
(arithmetic expression) stays simple — only the input language gets
harder.

Operand range: 1-99 to keep expressions within max_dec. Output
expressions max ~20 chars.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset

from calm.expression import ExpressionError, safe_eval
from calm.hrm.data import _CHAR_TO_ID


@dataclass
class GSMProblem:
    """GSM-style problem + its math expression + expected answer."""
    problem: str
    expression: str
    answer: str


_ACTORS = ["jane", "bob", "alice", "tom", "lisa", "tim", "mary", "kate",
           "sam", "eve", "jack", "lily"]
_PRONOUNS = {  # actor → (subj, possessive)
    "jane": ("she", "her"), "bob": ("he", "his"), "alice": ("she", "her"),
    "tom": ("he", "his"), "lisa": ("she", "her"), "tim": ("he", "his"),
    "mary": ("she", "her"), "kate": ("she", "her"),
    "sam": ("he", "his"), "eve": ("she", "her"),
    "jack": ("he", "his"), "lily": ("she", "her"),
}
_ITEMS = ["apples", "cookies", "marbles", "books", "candies", "balls",
          "stickers", "pencils", "coins", "cards", "toys", "stamps"]


# Template signatures: (builder_fn, expression_template, (max_x, max_y, max_z, max_w))

def _g_spend_find(v, a, pro):
    return (f"{a} had {v['x']} dollars. {pro[0]} spent {v['y']} on lunch. "
            f"later {pro[0]} found {v['z']} more. how much does {pro[0]} have now")


def _g_give_buy(v, a, pro, item, a2):
    return (f"{a} had {v['x']} {item}. after {pro[0]} gave {v['y']} to {a2}, "
            f"{pro[0]} bought {v['z']} more. how many {item} does {pro[0]} have now")


def _g_earn_work(v, a, pro):
    return (f"{a} earns {v['x']} dollars per day. {pro[0]} worked {v['y']} days. "
            f"{pro[0]} already had {v['z']}. how much does {pro[0]} have total")


def _g_boxes_sell(v, a, pro, item):
    return (f"{a} has {v['x']} boxes of {item}. each box holds {v['y']} {item}. "
            f"{pro[0]} sold {v['z']}. how many {item} does {pro[0]} have left")


def _g_shop_sum(v, item):
    return (f"a shop has {v['x']} {item} at {v['y']} dollars each "
            f"and {v['z']} {item} at {v['w']} dollars each. total revenue")


def _g_split_combine(v, a1, a2, item):
    return (f"{a1} has {v['x']} {item}. {a2} has {v['y']} {item}. "
            f"they combined and split evenly. how many {item} does each have")


def _g_saved_spent(v, a, pro):
    return (f"{a} saved {v['x']} dollars. then {pro[0]} spent {v['y']} on books "
            f"and {v['z']} on food. how much does {pro[0]} have left")


def _g_bought_sold_remaining(v, a, pro, item):
    return (f"a store had {v['x']} {item}. {a} bought {v['y']}. "
            f"then {pro[0]} returned {v['z']}. how many {item} are at the store")


def _g_each_spent(v, a, pro):
    return (f"{a} has {v['x']} coins. each coin is worth {v['y']} dollars. "
            f"{pro[0]} spent {v['z']} dollars. how many dollars does {pro[0]} have")


def _g_gave_twice(v, a, pro, a2, a3, item):
    return (f"{a} has {v['x']} {item}. {pro[0]} gave {v['y']} to {a2} "
            f"and {v['z']} to {a3}. how many {item} does {pro[0]} have left")


_TEMPLATES = [
    # 3-operand
    (_g_spend_find,            "{x} - {y} + {z}",           (99, 99, 99, 0)),
    (_g_give_buy,              "{x} - {y} + {z}",           (99, 99, 99, 0)),
    (_g_earn_work,             "{x} * {y} + {z}",           (30, 30, 99, 0)),
    (_g_boxes_sell,            "{x} * {y} - {z}",           (20, 20, 99, 0)),
    (_g_saved_spent,           "{x} - {y} - {z}",           (99, 99, 99, 0)),
    (_g_bought_sold_remaining, "{x} - {y} + {z}",           (99, 99, 99, 0)),
    (_g_each_spent,            "{x} * {y} - {z}",           (30, 30, 99, 0)),
    (_g_gave_twice,            "{x} - {y} - {z}",           (99, 99, 99, 0)),
    (_g_split_combine,         "({x} + {y}) / 2",           (98, 98, 0, 0)),
    # 4-operand
    (_g_shop_sum,              "{x} * {y} + {z} * {w}",     (20, 20, 20, 20)),
]


class GSMDataGenerator:
    """Generate (GSM-style problem, expression, answer) triples."""

    def __init__(self, seed: int = 42):
        self._rng = random.Random(seed)

    def generate(self, n: int = 2000) -> List[GSMProblem]:
        problems: List[GSMProblem] = []
        attempts = 0
        while len(problems) < n and attempts < n * 5:
            attempts += 1
            tmpl, expr_tmpl, (xmax, ymax, zmax, wmax) = self._rng.choice(_TEMPLATES)
            vals = {
                "x": self._rng.randint(1, xmax) if xmax else 0,
                "y": self._rng.randint(1, ymax) if ymax else 0,
                "z": self._rng.randint(1, zmax) if zmax else 0,
                "w": self._rng.randint(1, wmax) if wmax else 0,
            }
            item = self._rng.choice(_ITEMS)
            name = tmpl.__name__

            try:
                if name == "_g_give_buy":
                    actor, a2 = self._rng.sample(_ACTORS, 2)
                    pro = _PRONOUNS[actor]
                    problem = tmpl(vals, actor, pro, item, a2)
                elif name == "_g_boxes_sell" or name == "_g_bought_sold_remaining":
                    actor = self._rng.choice(_ACTORS)
                    pro = _PRONOUNS[actor]
                    problem = tmpl(vals, actor, pro, item)
                elif name == "_g_shop_sum":
                    problem = tmpl(vals, item)
                elif name == "_g_split_combine":
                    a1, a2 = self._rng.sample(_ACTORS, 2)
                    # Only accept when (x+y) is even (integer division).
                    if (vals["x"] + vals["y"]) % 2 != 0:
                        continue
                    problem = tmpl(vals, a1, a2, item)
                elif name == "_g_gave_twice":
                    actor, a2, a3 = self._rng.sample(_ACTORS, 3)
                    pro = _PRONOUNS[actor]
                    problem = tmpl(vals, actor, pro, a2, a3, item)
                else:
                    actor = self._rng.choice(_ACTORS)
                    pro = _PRONOUNS[actor]
                    problem = tmpl(vals, actor, pro)
            except Exception:
                continue

            expression = expr_tmpl.format(**vals)
            try:
                ans = safe_eval(expression)
                if isinstance(ans, float) and ans == int(ans):
                    ans = int(ans)
            except ExpressionError:
                continue

            if len(problem) + 2 > 128:
                continue
            if len(expression) + 2 + 1 > 28:
                continue
            # Verify all chars are in vocab.
            if any(c not in _CHAR_TO_ID for c in problem):
                continue

            problems.append(GSMProblem(problem=problem, expression=expression,
                                        answer=str(ans)))
        return problems


class GSMDataset(Dataset):
    """Encoder: GSM problem; decoder target: expression + `=` + <eos>."""

    def __init__(self, problems: List[GSMProblem], max_enc_len: int = 128,
                 max_dec_len: int = 28):
        self.problems = problems
        self.max_enc_len = max_enc_len
        self.max_dec_len = max_dec_len

    def __len__(self) -> int:
        return len(self.problems)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        p = self.problems[idx]
        pad_id = _CHAR_TO_ID["<pad>"]
        bos_id = _CHAR_TO_ID["<bos>"]
        eos_id = _CHAR_TO_ID["<eos>"]

        enc_ids = [bos_id] + [_CHAR_TO_ID[c] for c in p.problem if c in _CHAR_TO_ID] + [eos_id]
        enc_ids = enc_ids[: self.max_enc_len]
        while len(enc_ids) < self.max_enc_len:
            enc_ids.append(pad_id)

        target_str = p.expression + "="
        tgt_ids = [_CHAR_TO_ID[c] for c in target_str if c in _CHAR_TO_ID] + [eos_id]

        dec_in = [bos_id] + tgt_ids[:-1]
        dec_target = tgt_ids
        dec_in = dec_in[: self.max_dec_len]
        dec_target = dec_target[: self.max_dec_len]
        while len(dec_in) < self.max_dec_len:
            dec_in.append(pad_id)
        while len(dec_target) < self.max_dec_len:
            dec_target.append(pad_id)

        mask = [1 if t != pad_id else 0 for t in dec_target]

        return {
            "encoder_ids": torch.tensor(enc_ids, dtype=torch.long),
            "decoder_input_ids": torch.tensor(dec_in, dtype=torch.long),
            "decoder_target_ids": torch.tensor(dec_target, dtype=torch.long),
            "loss_mask": torch.tensor(mask, dtype=torch.bool),
        }
