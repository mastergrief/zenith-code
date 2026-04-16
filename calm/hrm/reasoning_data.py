"""Multi-step reasoning → expression training data.

Covers 8 reasoning types: chained arithmetic, comparison, conditional,
sequence costing, syllogism/transitivity, max/min, percentage, ratio.
~35 templates with varied NL phrasings per operation type.

Output target: parseable expression string for safe_eval. The PT learns
to transduce NL reasoning descriptions into formal expressions.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Tuple

from calm.expression import ExpressionError, safe_eval

_ACTORS = ["alice", "bob", "tom", "lisa", "sam", "eve", "jack", "kate",
           "mary", "tim", "jane", "dave"]
_ITEMS = ["apples", "books", "coins", "marbles", "stickers", "pencils",
          "cookies", "toys", "cards", "stamps", "balls", "candies"]


@dataclass
class ReasoningProblem:
    """Multi-step reasoning problem + expression + answer."""
    problem: str
    expression: str
    answer: str


# Template: (builder_fn, expression_template, operand_config)
# operand_config: dict of name → (min, max) range

# --- CHAINED ARITHMETIC (2-3 steps, left-to-right) ---

def _chain_spend_earn(v, a):
    return f"{a} had {v['x']} dollars then spent {v['y']} and earned {v['z']}. how much now"

def _chain_buy_sell(v, a, item):
    return f"{a} started with {v['x']} {item} then bought {v['y']} more and sold {v['z']}. how many left"

def _chain_temperature(v):
    return f"temperature was {v['x']} degrees then rose {v['y']} and dropped {v['z']}. what is it now"

def _chain_score(v, a):
    return f"{a} scored {v['x']} points then lost {v['y']} and gained {v['z']}. total score"

def _chain_steps(v):
    return f"start with {v['x']} add {v['y']} then multiply by {v['z']}. what is the result"

# --- COMPARISON ---

def _compare_who_more(v, a1, a2, item):
    return f"{a1} has {v['x']} {item}. {a2} has {v['y']} {item}. who has more"

def _compare_is_greater(v):
    return f"is {v['x']} greater than {v['y']}"

def _compare_which_bigger(v):
    return f"which is bigger {v['x']} or {v['y']}"

def _compare_enough(v, a, item):
    return f"{a} needs {v['x']} {item} but only has {v['y']}. does {a} have enough"

# --- CONDITIONAL ---

def _cond_if_enough(v, a, item):
    return f"if {a} has more than {v['y']} {item} then {a} buys {v['z']} more otherwise {a} buys {v['w']}. {a} has {v['x']} {item}. how many bought"

def _cond_weather(v):
    return f"if temperature is above {v['y']} pack {v['z']} items otherwise pack {v['w']}. temperature is {v['x']}. how many items"

def _cond_discount(v):
    return f"if order is above {v['y']} dollars shipping is {v['z']} otherwise shipping is {v['w']}. order total is {v['x']}. shipping cost"

# --- SEQUENCE COST ---

def _cost_two_items(v, item):
    return f"buy {v['x']} {item} at {v['y']} dollars each and {v['z']} {item} at {v['w']} dollars each. total cost"

def _cost_shopping(v, a):
    return f"{a} bought {v['x']} shirts at {v['y']} dollars each and {v['z']} pants at {v['w']} dollars each. how much spent"

def _cost_tickets(v):
    return f"{v['x']} adult tickets at {v['y']} dollars and {v['z']} child tickets at {v['w']} dollars. total price"

def _cost_supplies(v, a):
    return f"{a} needs {v['x']} notebooks at {v['y']} each and {v['z']} pens at {v['w']} each. total expense"

# --- PERCENTAGE ---

def _pct_of(v):
    return f"what is {v['x']} percent of {v['y']}"

def _pct_discount(v, item):
    return f"a {item} costs {v['y']} dollars. there is a {v['x']} percent discount. how much is saved"

def _pct_tip(v, a):
    return f"{a} meal costs {v['y']} dollars. {a} tips {v['x']} percent. how much is the tip"

def _pct_tax(v):
    return f"price is {v['y']} dollars with {v['x']} percent tax. how much tax"

# --- RATIO ---

def _ratio_of(v):
    return f"ratio of {v['x']} to {v['y']}"

def _ratio_class(v):
    return f"a class has {v['x']} boys and {v['y']} girls. what is the boy to girl ratio"

def _ratio_recipe(v):
    return f"recipe uses {v['x']} cups flour and {v['y']} cups sugar. flour to sugar ratio"

# --- MAX/MIN ---

def _max_three(v, a1, a2, a3, item):
    return f"{a1} has {v['x']} {item}. {a2} has {v['y']} {item}. {a3} has {v['z']} {item}. who has the most"

def _min_three(v, a1, a2, a3, item):
    return f"{a1} has {v['x']} {item}. {a2} has {v['y']} {item}. {a3} has {v['z']} {item}. who has the fewest"

def _max_scores(v):
    return f"scores are {v['x']} and {v['y']} and {v['z']}. what is the highest score"

def _min_prices(v):
    return f"prices are {v['x']} and {v['y']} and {v['z']} dollars. cheapest price"

# --- SYLLOGISM / TRANSITIVITY ---

def _trans_taller(v, a1, a2, a3):
    return f"{a1} is {v['x']} cm tall. {a2} is {v['y']} cm tall. {a3} is {v['z']} cm tall. who is tallest"

def _trans_age(v, a1, a2, a3):
    return f"{a1} is {v['x']} years old. {a2} is {v['y']} years old. {a3} is {v['z']} years old. who is youngest"

def _trans_faster(v, a1, a2, a3):
    return f"{a1} ran {v['x']} miles. {a2} ran {v['y']} miles. {a3} ran {v['z']} miles. who ran farthest"

# --- EXPLICIT SYLLOGISM ---

def _syl_transitive_gt(v, a1, a2, a3):
    return (f"{a1} has {v['x']} points. {a2} has {v['y']} points. {a3} has {v['z']} points. "
            f"if {a1} has more than {a2} and {a2} has more than {a3} is {a1} more than {a3}")

def _syl_if_then_chain(v, a1, a2):
    return (f"{a1} scored {v['x']}. {a2} scored {v['y']}. "
            f"if {v['x']} is greater than {v['y']} then {a1} wins. does {a1} win")

def _syl_both_true(v, a1, a2, a3):
    return (f"{a1} has {v['x']} coins. {a2} has {v['y']} coins. {a3} has {v['z']} coins. "
            f"does {a1} have more than both {a2} and {a3}")

def _syl_neither(v, a1, a2):
    return (f"{a1} scored {v['x']}. {a2} scored {v['y']}. "
            f"is {a1} score less than {a2} score")


# --- TEMPLATE REGISTRY ---
# (builder, expression_template, operand_config, builder_needs)
# builder_needs: "actor", "two_actors", "three_actors", "item", "actor_item", "three_actors_item", "none"

_TEMPLATES = [
    # Chained arithmetic (5)
    (_chain_spend_earn, "{x} - {y} + {z}",       {"x": (1, 99), "y": (1, 99), "z": (1, 99)}, "actor"),
    (_chain_buy_sell,   "{x} + {y} - {z}",        {"x": (1, 99), "y": (1, 99), "z": (1, 99)}, "actor_item"),
    (_chain_temperature,"{x} + {y} - {z}",        {"x": (1, 99), "y": (1, 50), "z": (1, 50)}, "none"),
    (_chain_score,      "{x} - {y} + {z}",        {"x": (1, 99), "y": (1, 50), "z": (1, 50)}, "actor"),
    (_chain_steps,      "({x} + {y}) * {z}",      {"x": (1, 20), "y": (1, 20), "z": (1, 9)},  "none"),
    # Comparison (4) — use Python-native > / < that safe_eval handles
    (_compare_who_more, "{x} > {y}",               {"x": (1, 99), "y": (1, 99)}, "two_actors_item"),
    (_compare_is_greater,"{x} > {y}",              {"x": (1, 99), "y": (1, 99)}, "none"),
    (_compare_which_bigger,"{x} > {y}",            {"x": (1, 99), "y": (1, 99)}, "none"),
    (_compare_enough,   "{y} >= {x}",              {"x": (1, 99), "y": (1, 99)}, "actor_item"),
    # Conditional (3) — Python ternary: val_true if cond else val_false
    (_cond_if_enough,   "{z} if {x} > {y} else {w}", {"x": (1, 99), "y": (1, 99), "z": (1, 30), "w": (1, 30)}, "actor_item"),
    (_cond_weather,     "{z} if {x} > {y} else {w}", {"x": (1, 99), "y": (1, 99), "z": (1, 30), "w": (1, 30)}, "none"),
    (_cond_discount,    "{z} if {x} > {y} else {w}", {"x": (1, 99), "y": (1, 99), "z": (1, 20), "w": (1, 20)}, "none"),
    # Sequence cost (4)
    (_cost_two_items,   "sequence_cost({x}, {y}, {z}, {w})", {"x": (1, 20), "y": (1, 20), "z": (1, 20), "w": (1, 20)}, "item"),
    (_cost_shopping,    "sequence_cost({x}, {y}, {z}, {w})", {"x": (1, 10), "y": (5, 50), "z": (1, 10), "w": (10, 80)}, "actor"),
    (_cost_tickets,     "sequence_cost({x}, {y}, {z}, {w})", {"x": (1, 10), "y": (5, 30), "z": (1, 10), "w": (3, 15)}, "none"),
    (_cost_supplies,    "sequence_cost({x}, {y}, {z}, {w})", {"x": (1, 20), "y": (1, 10), "z": (1, 20), "w": (1, 5)}, "actor"),
    # Percentage (4)
    (_pct_of,           "percentage({x}, {y})",    {"x": (1, 50), "y": (10, 99)}, "none"),
    (_pct_discount,     "percentage({x}, {y})",    {"x": (5, 50), "y": (10, 99)}, "item"),
    (_pct_tip,          "percentage({x}, {y})",    {"x": (10, 25), "y": (10, 99)}, "actor"),
    (_pct_tax,          "percentage({x}, {y})",    {"x": (5, 20), "y": (10, 99)}, "none"),
    # Ratio (3)
    (_ratio_of,         "ratio_simplify({x}, {y})", {"x": (1, 99), "y": (1, 99)}, "none"),
    (_ratio_class,      "ratio_simplify({x}, {y})", {"x": (1, 40), "y": (1, 40)}, "none"),
    (_ratio_recipe,     "ratio_simplify({x}, {y})", {"x": (1, 10), "y": (1, 10)}, "none"),
    # Max/Min (4)
    (_max_three,        "multi_max({x}, {y}, {z})", {"x": (1, 99), "y": (1, 99), "z": (1, 99)}, "three_actors_item"),
    (_min_three,        "multi_min({x}, {y}, {z})", {"x": (1, 99), "y": (1, 99), "z": (1, 99)}, "three_actors_item"),
    (_max_scores,       "multi_max({x}, {y}, {z})", {"x": (1, 99), "y": (1, 99), "z": (1, 99)}, "none"),
    (_min_prices,       "multi_min({x}, {y}, {z})", {"x": (1, 99), "y": (1, 99), "z": (1, 99)}, "none"),
    # Transitivity (3)
    (_trans_taller,     "multi_max({x}, {y}, {z})", {"x": (100, 200), "y": (100, 200), "z": (100, 200)}, "three_actors"),
    (_trans_age,        "multi_min({x}, {y}, {z})", {"x": (5, 80), "y": (5, 80), "z": (5, 80)}, "three_actors"),
    (_trans_faster,     "multi_max({x}, {y}, {z})", {"x": (1, 30), "y": (1, 30), "z": (1, 30)}, "three_actors"),
    # Explicit syllogism (4) — boolean outputs
    (_syl_transitive_gt,"{x} > {y} and {y} > {z}", {"x": (1, 99), "y": (1, 99), "z": (1, 99)}, "three_actors"),
    (_syl_if_then_chain,"{x} > {y}",               {"x": (1, 99), "y": (1, 99)}, "two_actors"),
    (_syl_both_true,    "{x} > {y} and {x} > {z}", {"x": (1, 99), "y": (1, 99), "z": (1, 99)}, "three_actors"),
    (_syl_neither,      "{x} < {y}",               {"x": (1, 99), "y": (1, 99)}, "two_actors"),
]


class ReasoningDataGenerator:
    """Generate multi-step reasoning problems from templates."""

    def __init__(self, seed: int = 42):
        self._rng = random.Random(seed)

    def _sample_operand(self, lo: int, hi: int) -> int:
        """Balanced digit-length sampling."""
        if hi <= 9 or hi - lo < 9:
            return self._rng.randint(lo, hi)
        buckets = []
        if lo <= 9 and hi >= 1:
            buckets.append((max(lo, 1), min(9, hi)))
        if lo <= 99 and hi >= 10:
            buckets.append((max(lo, 10), min(99, hi)))
        if hi >= 100:
            buckets.append((max(lo, 100), hi))
        buckets = [(a, b) for a, b in buckets if a <= b]
        if not buckets:
            return self._rng.randint(lo, hi)
        blo, bhi = self._rng.choice(buckets)
        return self._rng.randint(blo, bhi)

    def generate(self, n: int = 5000) -> List[ReasoningProblem]:
        problems: List[ReasoningProblem] = []
        attempts = 0
        while len(problems) < n and attempts < n * 5:
            attempts += 1
            builder, expr_tmpl, op_cfg, needs = self._rng.choice(_TEMPLATES)

            vals = {}
            for name, (lo, hi) in op_cfg.items():
                vals[name] = self._sample_operand(lo, hi)

            try:
                # Build actors/items as needed
                if needs == "actor":
                    actor = self._rng.choice(_ACTORS)
                    problem = builder(vals, actor)
                elif needs == "actor_item":
                    actor = self._rng.choice(_ACTORS)
                    item = self._rng.choice(_ITEMS)
                    problem = builder(vals, actor, item)
                elif needs == "item":
                    item = self._rng.choice(_ITEMS)
                    problem = builder(vals, item)
                elif needs == "two_actors":
                    a1, a2 = self._rng.sample(_ACTORS, 2)
                    problem = builder(vals, a1, a2)
                elif needs == "two_actors_item":
                    a1, a2 = self._rng.sample(_ACTORS, 2)
                    item = self._rng.choice(_ITEMS)
                    problem = builder(vals, a1, a2, item)
                elif needs == "three_actors":
                    a1, a2, a3 = self._rng.sample(_ACTORS, 3)
                    problem = builder(vals, a1, a2, a3)
                elif needs == "three_actors_item":
                    a1, a2, a3 = self._rng.sample(_ACTORS, 3)
                    item = self._rng.choice(_ITEMS)
                    problem = builder(vals, a1, a2, a3, item)
                else:
                    problem = builder(vals)
            except Exception:
                continue

            expression = expr_tmpl.format(**vals)

            try:
                ans = safe_eval(expression)
                if isinstance(ans, float) and ans == int(ans):
                    ans = int(ans)
            except (ExpressionError, OverflowError, ZeroDivisionError):
                continue

            # Length bounds
            if len(problem) > 140:
                continue
            if len(expression) > 50:
                continue

            problems.append(ReasoningProblem(
                problem=problem, expression=expression, answer=str(ans)))
        return problems


def _is_funcall_expr(expr_tmpl: str) -> bool:
    """True if expression template uses function-call syntax."""
    return any(fn in expr_tmpl for fn in
               ("sequence_cost", "percentage", "ratio_simplify",
                "multi_max", "multi_min"))


# Split templates by output language family
_FUNCALL_TEMPLATES = [t for t in _TEMPLATES if _is_funcall_expr(t[1])]
_LOGIC_TEMPLATES = [t for t in _TEMPLATES if not _is_funcall_expr(t[1])]


class FuncallReasoningGenerator(ReasoningDataGenerator):
    """Function-call reasoning only: percentage, ratio, seq_cost, max/min."""

    def generate(self, n: int = 5000) -> List[ReasoningProblem]:
        # Temporarily swap template list
        import calm.hrm.reasoning_data as mod
        orig = mod._TEMPLATES
        mod._TEMPLATES = _FUNCALL_TEMPLATES
        try:
            result = super().generate(n)
        finally:
            mod._TEMPLATES = orig
        return result


class LogicReasoningGenerator(ReasoningDataGenerator):
    """Logic/operator reasoning: arithmetic, compare, conditional, syllogism."""

    def generate(self, n: int = 5000) -> List[ReasoningProblem]:
        import calm.hrm.reasoning_data as mod
        orig = mod._TEMPLATES
        mod._TEMPLATES = _LOGIC_TEMPLATES
        try:
            result = super().generate(n)
        finally:
            mod._TEMPLATES = orig
        return result
