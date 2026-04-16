"""Creative writing → function routing training data.

The PT maps writing-related NL queries to the correct verification or
analysis function. This is a NEW output-language family: function routing
(classify intent → emit function name + args). Unlike math domains where
the PT copies operands from input, here the PT mostly generates function
names and copies text arguments.

Output targets: function calls like `syllable_count(word)`,
`haiku_valid`, `form_valid(sonnet)`, `readability_score`,
`rhymes(word1, word2)`, `meter_type`, etc.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List


@dataclass
class WritingProblem:
    """Creative writing query + function routing target."""
    problem: str
    expression: str
    answer: str


_WORDS_SHORT = ["cat", "dog", "sun", "moon", "rain", "love", "tree", "sea",
                "wind", "fire", "bird", "star", "rose", "snow", "song"]
_WORDS_LONG = ["beautiful", "wonderful", "extraordinary", "magnificent",
               "melancholy", "serendipity", "ephemeral", "luminous",
               "tranquility", "catastrophe", "magnificent", "eloquent"]
_RHYME_PAIRS = [("cat", "hat"), ("moon", "soon"), ("day", "way"),
                ("light", "night"), ("love", "dove"), ("rain", "pain"),
                ("tree", "free"), ("song", "long"), ("fire", "higher"),
                ("rose", "grows"), ("snow", "flow"), ("star", "far")]
_NON_RHYMES = [("cat", "dog"), ("moon", "sun"), ("day", "night"),
               ("love", "hate"), ("rain", "snow"), ("tree", "rock")]
_FORMS = ["haiku", "sonnet", "limerick", "tanka", "villanelle", "ghazal",
          "free verse", "blank verse", "couplet"]
_TOPICS = ["autumn", "rain", "love", "the ocean", "a sunset", "winter",
           "friendship", "solitude", "a garden", "the stars", "memory",
           "childhood", "hope", "a storm", "silence"]
_TEXTS = [
    "The old pond. A frog jumps in. Splash silence again.",
    "She sells seashells by the seashore.",
    "It was the best of times it was the worst of times.",
    "The quick brown fox jumps over the lazy dog.",
    "To be or not to be that is the question.",
    "All that glitters is not gold.",
    "A rose by any other name would smell as sweet.",
    "The world is too much with us late and soon.",
]


class WritingDataGenerator:
    """Generate creative writing routing problems."""

    def __init__(self, seed: int = 42):
        self._rng = random.Random(seed)

    def generate(self, n: int = 5000) -> List[WritingProblem]:
        problems: List[WritingProblem] = []
        builders = [
            self._syllable_query, self._rhyme_query, self._meter_query,
            self._form_write, self._form_check, self._readability_query,
            self._alliteration_query, self._passive_query, self._vocab_query,
            self._word_count_query, self._stress_query, self._form_lookup,
        ]
        while len(problems) < n:
            builder = self._rng.choice(builders)
            p = builder()
            if p:
                problems.append(p)
        return problems

    def _syllable_query(self) -> WritingProblem:
        word = self._rng.choice(_WORDS_SHORT + _WORDS_LONG)
        templates = [
            f"how many syllables in {word}",
            f"count the syllables in {word}",
            f"syllable count for {word}",
            f"how many syllables does {word} have",
        ]
        q = self._rng.choice(templates)
        return WritingProblem(problem=q, expression=f"syllable_count({word})",
                              answer="")

    def _rhyme_query(self) -> WritingProblem:
        if self._rng.random() < 0.6:
            w1, w2 = self._rng.choice(_RHYME_PAIRS)
        else:
            w1, w2 = self._rng.choice(_NON_RHYMES)
        templates = [
            f"does {w1} rhyme with {w2}",
            f"do {w1} and {w2} rhyme",
            f"check if {w1} rhymes with {w2}",
            f"is {w1} a rhyme for {w2}",
        ]
        q = self._rng.choice(templates)
        return WritingProblem(problem=q, expression=f"rhymes({w1}, {w2})",
                              answer="")

    def _meter_query(self) -> WritingProblem:
        lines = [
            "shall i compare thee to a summers day",
            "once upon a midnight dreary",
            "the fog comes on little cat feet",
            "i wandered lonely as a cloud",
            "twas the night before christmas",
            "do not go gentle into that good night",
            "whose woods these are i think i know",
        ]
        line = self._rng.choice(lines)
        templates = [
            f"what meter is {line}",
            f"detect the meter of {line}",
            f"what is the meter in {line}",
        ]
        q = self._rng.choice(templates)
        return WritingProblem(problem=q, expression="meter_type",
                              answer="")

    def _form_write(self) -> WritingProblem:
        form = self._rng.choice(_FORMS)
        topic = self._rng.choice(_TOPICS)
        templates = [
            f"write a {form} about {topic}",
            f"compose a {form} on the theme of {topic}",
            f"create a {form} inspired by {topic}",
            f"write me a {form} about {topic}",
        ]
        q = self._rng.choice(templates)
        form_key = form.replace(" ", "_")
        return WritingProblem(problem=q, expression=f"form_valid({form_key})",
                              answer="")

    def _form_check(self) -> WritingProblem:
        form = self._rng.choice(_FORMS)
        templates = [
            f"is this poem a valid {form}",
            f"check if this is a proper {form}",
            f"validate this {form}",
            f"does this follow {form} form",
        ]
        q = self._rng.choice(templates)
        form_key = form.replace(" ", "_")
        return WritingProblem(problem=q, expression=f"form_valid({form_key})",
                              answer="")

    def _readability_query(self) -> WritingProblem:
        templates = [
            "what is the readability score",
            "check the reading level",
            "how readable is this text",
            "readability grade for this passage",
            "flesch kincaid score",
        ]
        q = self._rng.choice(templates)
        return WritingProblem(problem=q, expression="readability_score",
                              answer="")

    def _alliteration_query(self) -> WritingProblem:
        templates = [
            "how much alliteration is there",
            "check for alliteration",
            "alliteration density in this text",
            "measure the alliteration",
        ]
        q = self._rng.choice(templates)
        return WritingProblem(problem=q, expression="alliteration_density",
                              answer="")

    def _passive_query(self) -> WritingProblem:
        templates = [
            "how much passive voice is used",
            "check for passive voice",
            "passive voice percentage",
            "detect passive voice",
        ]
        q = self._rng.choice(templates)
        return WritingProblem(problem=q, expression="passive_voice_pct",
                              answer="")

    def _vocab_query(self) -> WritingProblem:
        templates = [
            "how diverse is the vocabulary",
            "vocabulary richness score",
            "lexical diversity of this text",
            "unique word ratio",
        ]
        q = self._rng.choice(templates)
        return WritingProblem(problem=q, expression="unique_word_ratio",
                              answer="")

    def _word_count_query(self) -> WritingProblem:
        templates = [
            "how many words",
            "word count",
            "count the words",
            "how long is this text in words",
        ]
        q = self._rng.choice(templates)
        return WritingProblem(problem=q, expression="word_count",
                              answer="")

    def _stress_query(self) -> WritingProblem:
        lines = [
            "shall i compare thee to a summers day",
            "once upon a midnight dreary",
            "the fog comes on little cat feet",
        ]
        line = self._rng.choice(lines)
        templates = [
            f"stress pattern of {line}",
            f"show the stress pattern for {line}",
        ]
        q = self._rng.choice(templates)
        return WritingProblem(problem=q, expression="stress_pattern",
                              answer="")

    def _form_lookup(self) -> WritingProblem:
        form = self._rng.choice(_FORMS)
        templates = [
            f"what is a {form}",
            f"explain the {form} form",
            f"describe the rules of a {form}",
            f"what are the rules for a {form}",
        ]
        q = self._rng.choice(templates)
        form_key = form.replace(" ", "_")
        return WritingProblem(problem=q, expression=f"get_poetry_form({form_key})",
                              answer="")
