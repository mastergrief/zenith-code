"""Word-problem → math expression training data.

Tests the CRLM scaling law ("HRM size scales with problem-language
complexity, not problem-difficulty") on NL that requires real structure
extraction — names, anaphora, multi-step actions, comparison semantics
("more than", "fewer than"), pronouns.

Output target is the same as `nl_data.py`: a parseable math expression
like `5 + 3` or `8 - 2 + 4`. LLM-Computer's interpreter handles all
values.

Operand range: 1-99 to keep sentence lengths under max_enc=80. Actor
vocabulary (sally, bob, alice, ...) and item vocabulary (apples,
cookies, ...) are small but varied enough to test generalization.
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
class WordProblem:
    """NL word problem + its math-expression answer."""
    problem: str
    expression: str
    answer: str


# Actor + item pools. Short names keep sentences inside max_enc.
_ACTORS = ["sally", "bob", "alice", "tom", "lisa", "tim", "mary", "jane",
           "kate", "dave", "eve", "sam"]
_PRONOUNS = {  # actor → (subj, possessive)
    "sally": ("she", "her"), "bob": ("he", "his"), "alice": ("she", "her"),
    "tom": ("he", "his"), "lisa": ("she", "her"), "tim": ("he", "his"),
    "mary": ("she", "her"), "jane": ("she", "her"), "kate": ("she", "her"),
    "dave": ("he", "his"), "eve": ("she", "her"), "sam": ("he", "his"),
}
_ITEMS = ["apples", "cookies", "marbles", "books", "candies", "balls",
          "stickers", "pencils", "coins", "cards", "toys", "stamps"]


# (template fn, expression template, operand ranges for {x, y, z})
# Template functions take a dict of values and return a problem string.

def _t_buy(v, item, actor, pro):
    return f"{actor} has {v['x']} {item}. {pro[0]} buys {v['y']} more. how many {item} does {pro[0]} have"


def _t_eat(v, item, actor, pro):
    return f"{actor} has {v['x']} {item}. {pro[0]} eats {v['y']}. how many {item} are left"


def _t_give(v, item, actor, pro):
    return f"{actor} has {v['x']} {item}. {pro[0]} gives away {v['y']}. how many {item} does {pro[0]} have left"


def _t_sum_two_actors(v, item, a1, a2):
    return f"{a1} has {v['x']} {item}. {a2} has {v['y']} {item}. how many {item} in total"


def _t_more_than(v, item, a1, a2):
    return f"{a1} has {v['x']} {item}. {a2} has {v['y']} more than {a1}. how many {item} does {a2} have"


def _t_fewer_than(v, item, a1, a2):
    return f"{a1} has {v['x']} {item}. {a2} has {v['y']} fewer than {a1}. how many {item} does {a2} have"


def _t_groups(v, item):
    return f"there are {v['x']} boxes with {v['y']} {item} each. how many {item} in total"


def _t_cost(v):
    return f"a pack contains {v['x']} items costing {v['y']} dollars each. total cost"


def _t_buy_eat(v, item, actor, pro):
    return f"{actor} has {v['x']} {item}. {pro[0]} buys {v['y']} more then eats {v['z']}. how many {item} are left"


def _t_buy_buy(v, item, actor, pro):
    return f"{actor} has {v['x']} {item}. {pro[0]} buys {v['y']} more and {v['z']} more. how many {item} now"


def _t_gain_lose(v, actor, pro):
    return f"{actor} has {v['x']} dollars. {pro[0]} earns {v['y']} and spends {v['z']}. how many dollars does {pro[0]} have"


def _t_three_actors(v, item, a1, a2, a3):
    return (f"{a1} has {v['x']} {item}. {a2} has {v['y']} {item}. "
            f"{a3} has {v['z']} {item}. how many {item} in total")


def _t_double(v, item, actor, pro):
    return f"{actor} has {v['x']} {item}. {pro[0]} doubles {pro[1]} collection. how many {item} does {pro[0]} have"


def _t_triple(v, item, actor, pro):
    return f"{actor} has {v['x']} {item}. {pro[0]} triples {pro[1]} collection. how many {item} does {pro[0]} have"


# Each entry: (builder, expression template, (x_max, y_max, z_max))
# Builder gets (vals, item, actor_or_tuple, pronouns).
_TEMPLATES = [
    # 1-operand
    (_t_double,          "{x} * 2",              (99, 0, 0)),
    (_t_triple,          "{x} * 3",              (99, 0, 0)),
    # 2-operand
    (_t_buy,             "{x} + {y}",            (99, 99, 0)),
    (_t_eat,             "{x} - {y}",            (99, 99, 0)),
    (_t_give,            "{x} - {y}",            (99, 99, 0)),
    (_t_sum_two_actors,  "{x} + {y}",            (99, 99, 0)),
    (_t_more_than,       "{x} + {y}",            (99, 99, 0)),
    (_t_fewer_than,      "{x} - {y}",            (99, 99, 0)),
    (_t_groups,          "{x} * {y}",            (30, 30, 0)),
    (_t_cost,            "{x} * {y}",            (30, 30, 0)),
    # 3-operand
    (_t_buy_eat,         "{x} + {y} - {z}",      (99, 99, 99)),
    (_t_buy_buy,         "{x} + {y} + {z}",      (99, 99, 99)),
    (_t_gain_lose,       "{x} + {y} - {z}",      (99, 99, 99)),
    (_t_three_actors,    "{x} + {y} + {z}",      (99, 99, 99)),
]


class WordProblemGenerator:
    """Generate word problem / expression pairs from templates."""

    def __init__(self, seed: int = 42):
        self._rng = random.Random(seed)

    def _sample_operand(self, max_val: int) -> int:
        """Balanced digit-length sampling (same as NLMathDataGenerator)."""
        if max_val <= 9:
            return self._rng.randint(1, max_val)
        buckets = [(1, 9)]
        if max_val >= 10:
            buckets.append((10, min(99, max_val)))
        if max_val >= 100:
            buckets.append((100, max_val))
        lo, hi = self._rng.choice(buckets)
        return self._rng.randint(lo, hi)

    def generate(self, n: int = 2000) -> List[WordProblem]:
        problems: List[WordProblem] = []
        attempts = 0
        while len(problems) < n and attempts < n * 5:
            attempts += 1
            tmpl, expr_tmpl, (xmax, ymax, zmax) = self._rng.choice(_TEMPLATES)
            vals = {
                "x": self._sample_operand(xmax) if xmax else 0,
                "y": self._sample_operand(ymax) if ymax else 0,
                "z": self._sample_operand(zmax) if zmax else 0,
            }
            # For subtraction templates, ensure non-absurd answers — but
            # allow negatives (HRM just echos structure; interpreter
            # handles the arithmetic either way).
            item = self._rng.choice(_ITEMS)

            # Build the sentence — templates have different actor arity.
            name = tmpl.__name__
            try:
                if name in ("_t_sum_two_actors", "_t_more_than", "_t_fewer_than"):
                    a1, a2 = self._rng.sample(_ACTORS, 2)
                    problem = tmpl(vals, item, a1, a2)
                elif name == "_t_three_actors":
                    a1, a2, a3 = self._rng.sample(_ACTORS, 3)
                    problem = tmpl(vals, item, a1, a2, a3)
                elif name == "_t_groups":
                    problem = tmpl(vals, item)
                elif name == "_t_cost":
                    problem = tmpl(vals)
                else:
                    # single actor + pronouns
                    actor = self._rng.choice(_ACTORS)
                    pro = _PRONOUNS[actor]
                    problem = tmpl(vals, item, actor, pro)
            except Exception:
                continue

            expression = expr_tmpl.format(**vals)
            try:
                ans = safe_eval(expression)
                if isinstance(ans, float) and ans == int(ans):
                    ans = int(ans)
            except ExpressionError:
                continue

            # Enforce length bound — bail if sentence exceeds 80 chars.
            if len(problem) + 2 > 80:  # +2 for bos/eos
                continue
            if len(expression) + 2 + 1 > 24:  # +2 bos/eos, +1 for '='
                continue

            problems.append(WordProblem(problem=problem, expression=expression,
                                         answer=str(ans)))
        return problems


class WordProblemDataset(Dataset):
    """Encoder input = word problem; decoder target = expression + `=` + <eos>."""

    def __init__(self, problems: List[WordProblem], max_enc_len: int = 80,
                 max_dec_len: int = 24):
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
