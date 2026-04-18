"""R51.1 broad prompt bank for tier-3 L24 activation capture.

Hypothesis: a student trained only on arithmetic activations will
overfit to arithmetic and degrade Gemma's general capabilities. This
module produces 6 deterministic prompt generators so R51.1 can capture
L24 contributions across a representative spread of domains:

    multi     — multi-step arithmetic (R50.2 format, the R50.7 target)
    single    — single-step arithmetic (R50.4 format)
    trans     — short English -> {de,fr,es,it} translation
    code      — short Python code Q&A
    creative  — short creative writing (poems, descriptions, stories)
    factual   — short factual recall (capitals, elements, constants, dates)

Each sampler returns list[tuple[prompt, domain_label]] of length n.
Prompts end at a natural continuation point (":", "Answer: ", etc.)
so Gemma is primed to emit a next token. Generators dedup within their
own domain via hash + top-up, and are deterministic given the same
random.Random seed.

build_broad_corpus(rng, per_domain=500) calls all 6, concatenates,
shuffles (via rng), returns (prompts, counts_per_domain).
"""

from __future__ import annotations

import random
from typing import Callable

Prompt = tuple[str, str]


# ---------------------------------------------------------------------------
# multi-step arithmetic — match scripts/test_l24_sae_v2.py sample_multi_step
# ---------------------------------------------------------------------------

def sample_multi_step_arith(rng: random.Random, n: int = 500) -> list[Prompt]:
    out: list[Prompt] = []
    seen: set[str] = set()
    attempts = 0
    while len(out) < n and attempts < n * 50:
        attempts += 1
        a = rng.randint(2, 29)
        b = rng.randint(2, 29)
        if a * b >= 1000:
            continue
        c = rng.randint(1, 99)
        prompt = f"What is ({a} * {b}) + {c}? Answer: "
        if prompt in seen:
            continue
        seen.add(prompt)
        out.append((prompt, "multi"))
    return out


# ---------------------------------------------------------------------------
# single-step arithmetic — mixed templates
# ---------------------------------------------------------------------------

_SINGLE_TEMPLATES = (
    "What is {a} * {b}? Answer: ",
    "Compute {a} * {b}: ",
    "{a} times {b} equals ",
)


def sample_single_step_arith(rng: random.Random, n: int = 500) -> list[Prompt]:
    out: list[Prompt] = []
    seen: set[str] = set()
    attempts = 0
    while len(out) < n and attempts < n * 50:
        attempts += 1
        a = rng.randint(2, 29)
        b = rng.randint(2, 29)
        if a * b >= 1000:
            continue
        tmpl = rng.choice(_SINGLE_TEMPLATES)
        prompt = tmpl.format(a=a, b=b)
        if prompt in seen:
            continue
        seen.add(prompt)
        out.append((prompt, "single"))
    return out


# ---------------------------------------------------------------------------
# translation — English source x 4 target languages x framing variants
# ---------------------------------------------------------------------------

_TRANSLATION_SENTENCES = (
    "The cat is on the table.",
    "She went to the store yesterday.",
    "Three books and a pencil are on the desk.",
    "I would like a cup of coffee, please.",
    "The children are playing in the garden.",
    "He will travel to Paris next week.",
    "My brother lives in a small village.",
    "The weather is beautiful this morning.",
    "We have been friends for many years.",
    "She is writing a letter to her mother.",
    "The train leaves at nine o'clock.",
    "I cannot find my keys anywhere.",
    "They built a wooden house near the lake.",
    "The old man was reading a newspaper.",
    "Our team won the match last Saturday.",
    "She has already finished her homework.",
    "The dog ran quickly across the field.",
    "I am going to buy some bread.",
    "He spoke slowly and carefully.",
    "The garden is full of yellow flowers.",
    "We watched a good movie last night.",
    "My grandfather taught me how to fish.",
    "The children were laughing at the joke.",
    "She opened the window to let in fresh air.",
    "I forgot to bring my umbrella today.",
    "The museum opens at ten in the morning.",
    "He has never been to the mountains.",
    "They are planning a trip for the summer.",
    "The bread was still warm from the oven.",
    "She walked home through the empty streets.",
    "My sister wants to become a doctor.",
    "The river flows past the old mill.",
    "The baby is sleeping in the crib.",
    "I bought two loaves of bread at the bakery.",
    "They are building a new school near my house.",
    "He plays the piano every evening.",
    "She saw an interesting bird in the park.",
    "We will meet at the station tomorrow.",
    "The professor gave us a difficult exam.",
    "My parents traveled to Italy last year.",
    "The little boy was crying in the corner.",
    "I have already eaten my breakfast.",
    "The library closes at six in the evening.",
    "Can you pass me the salt, please?",
    "The sun rises early in the summer.",
    "He writes a poem every Sunday morning.",
    "The children made a snowman in the yard.",
    "She teaches mathematics at the university.",
    "The cake was sweeter than I expected.",
)

_TRANSLATION_LANGS = ("German", "French", "Spanish", "Italian")

_TRANSLATION_FRAMINGS = (
    "Translate to {lang}: {sentence}\nAnswer: ",
    "Translate this English sentence into {lang}: {sentence}\nTranslation: ",
    "Render the following in {lang}: {sentence}\n{lang}: ",
)


def sample_translation(rng: random.Random, n: int = 500) -> list[Prompt]:
    out: list[Prompt] = []
    seen: set[str] = set()
    attempts = 0
    while len(out) < n and attempts < n * 50:
        attempts += 1
        sentence = rng.choice(_TRANSLATION_SENTENCES)
        lang = rng.choice(_TRANSLATION_LANGS)
        framing = rng.choice(_TRANSLATION_FRAMINGS)
        prompt = framing.format(lang=lang, sentence=sentence)
        if prompt in seen:
            continue
        seen.add(prompt)
        out.append((prompt, "trans"))
    return out


# ---------------------------------------------------------------------------
# code — short Python code Q&A
# ---------------------------------------------------------------------------

_CODE_TASKS = (
    "reverses a list",
    "checks if a string is a palindrome",
    "returns the max of two numbers",
    "counts vowels in a string",
    "returns the factorial of a number",
    "computes the sum of a list of integers",
    "returns the length of a string",
    "removes duplicates from a list",
    "sorts a list of integers in ascending order",
    "flattens a nested list one level deep",
    "checks whether a number is even",
    "returns the average of a list of numbers",
    "finds the minimum element in a list",
    "converts a string to uppercase",
    "counts the number of words in a sentence",
    "returns the Fibonacci number at index n",
    "checks whether a number is prime",
    "returns the greatest common divisor of two integers",
    "swaps two variables without a temporary",
    "reverses the order of words in a sentence",
    "capitalizes the first letter of every word",
    "strips punctuation from a string",
    "returns the intersection of two lists",
    "returns the union of two lists",
    "checks whether a list is sorted",
    "returns every other element of a list",
    "rotates a list by k positions",
    "computes the dot product of two vectors",
    "returns the transpose of a 2D matrix",
    "converts Celsius to Fahrenheit",
    "converts Fahrenheit to Celsius",
    "checks if a year is a leap year",
    "returns the days in a given month",
    "computes compound interest",
    "returns the nth prime number",
    "checks whether a string contains only digits",
    "removes whitespace from both ends of a string",
    "splits a string on commas",
    "joins a list of strings with a separator",
    "reads a file and returns its contents",
    "writes a string to a file",
    "counts occurrences of an item in a list",
    "finds the index of an item in a list",
    "returns a dict mapping letters to their counts",
    "reverses a dictionary's keys and values",
    "merges two dictionaries",
    "returns the keys of a dict sorted by value",
    "checks whether a list contains duplicates",
    "returns the Cartesian product of two lists",
    "returns all permutations of a list",
    "returns all combinations of length k from a list",
    "computes the standard deviation of a list",
    "computes the median of a list",
    "generates the first n primes",
    "returns the digits of an integer as a list",
    "sums the digits of an integer",
    "checks if a number is a perfect square",
    "computes integer square root",
    "returns the binary representation of an integer",
    "converts binary string to an integer",
    "checks if two strings are anagrams",
    "returns the longest word in a sentence",
    "returns the shortest word in a sentence",
    "counts the number of lines in a file",
    "reads a CSV file into a list of dicts",
    "writes a list of dicts to a CSV file",
    "converts a list to a set",
    "returns the symmetric difference of two sets",
    "computes the power of a number iteratively",
    "returns the first n elements of a list",
    "returns the last n elements of a list",
    "checks if a list is empty",
    "pads a string to a fixed length",
    "returns the current date as a string",
    "parses an ISO 8601 date string",
    "checks whether two lists are equal",
    "returns a random integer in a given range",
    "shuffles a list in place",
    "returns the mode of a list of values",
    "computes the Hamming distance between two strings",
    "computes Levenshtein distance between two strings",
    "returns the number of uppercase letters in a string",
    "returns the number of lowercase letters in a string",
    "checks whether a string is a valid integer",
    "checks whether a string is a valid float",
    "returns the reverse of a dictionary's key order",
    "returns the largest three elements of a list",
    "returns the smallest three elements of a list",
    "computes the cumulative sum of a list",
    "computes the cumulative product of a list",
    "returns pairs of adjacent elements from a list",
    "chunks a list into sublists of size k",
    "returns a generator of Fibonacci numbers",
    "implements binary search over a sorted list",
    "implements linear search over a list",
    "returns the last element of a list safely",
    "returns True if all elements of a list are positive",
    "returns True if any element of a list is negative",
    "filters odd numbers from a list",
    "filters even numbers from a list",
    "squares every element of a list",
    "cubes every element of a list",
    "applies a function to every element of a list",
    "zips two lists of equal length",
    "unzips a list of pairs into two lists",
    "returns the number of unique elements in a list",
    "checks whether a dictionary contains a key",
    "returns the value for a key with a default",
    "counts the frequency of items in a list",
    "returns the most frequent item in a list",
    "returns the least frequent item in a list",
    "sorts a list of tuples by the second element",
    "sorts a dict by value descending",
    "returns the longest common prefix of two strings",
    "returns the longest common suffix of two strings",
    "capitalizes only the first letter of a string",
    "returns the title case of a string",
    "detects whether a string ends with a substring",
    "detects whether a string starts with a substring",
    "replaces one substring with another",
    "returns the absolute value of an integer",
    "returns the sign of a number as -1, 0, or 1",
    "clamps a number between a minimum and maximum",
    "linearly interpolates between two numbers",
    "converts a number of seconds to a formatted time string",
    "parses a time string into seconds",
    "returns the unique sorted elements of a list",
    "returns the indices of the maximum and minimum elements",
    "computes the variance of a list",
    "computes the range of a list",
    "computes the geometric mean of a list",
    "computes the harmonic mean of a list",
    "returns True if a list is a palindrome",
    "rounds every number in a list to the nearest integer",
    "filters a list using a predicate function",
    "reduces a list to a single value using a function",
    "zips three lists element-wise",
    "returns the first n Fibonacci numbers as a list",
    "returns the factorial of n using recursion",
    "returns the factorial of n iteratively",
    "returns a dict of character counts in a string",
    "checks whether one string is a rotation of another",
    "returns the Caesar cipher of a string",
    "deciphers a Caesar cipher",
    "counts vowels and consonants in a string",
    "checks if a list contains all distinct values",
    "returns True if two dicts are equal",
    "serializes a dict to a JSON string",
    "deserializes a JSON string to a dict",
    "reads JSON from a file",
    "writes JSON to a file",
    "computes the running average of a stream of numbers",
    "returns a function that composes two functions",
    "returns a memoized version of a function",
    "returns a function that applies a given function n times",
    "computes a running total while yielding each step",
    "computes modular exponentiation efficiently",
    "generates all subsets of a given list",
    "returns True if three points are collinear",
    "computes Euclidean distance between two points",
    "returns the area of a triangle given three sides",
    "returns True if a year is a century leap year",
    "implements a stack using a list",
    "implements a queue using two stacks",
    "implements a simple LRU cache",
    "returns a deep copy of a nested list",
    "checks if a tree is balanced",
    "traverses a binary tree in order",
    "traverses a binary tree pre-order",
    "traverses a binary tree post-order",
    "counts leaves in a binary tree",
    "returns the height of a binary tree",
    "finds the lowest common ancestor in a binary tree",
    "merges two sorted lists into one sorted list",
)

_CODE_FRAMINGS = (
    "Write a Python function that {task}.\nAnswer: ",
    "In Python, write a function that {task}.\nCode: ",
    "Provide a short Python implementation that {task}.\n",
)


def sample_code(rng: random.Random, n: int = 500) -> list[Prompt]:
    out: list[Prompt] = []
    seen: set[str] = set()
    attempts = 0
    while len(out) < n and attempts < n * 50:
        attempts += 1
        task = rng.choice(_CODE_TASKS)
        framing = rng.choice(_CODE_FRAMINGS)
        prompt = framing.format(task=task)
        if prompt in seen:
            continue
        seen.add(prompt)
        out.append((prompt, "code"))
    return out


# ---------------------------------------------------------------------------
# creative — poems, descriptions, story starters
# ---------------------------------------------------------------------------

_CREATIVE_NOUNS = (
    "the ocean", "a forest", "the moon", "a mountain", "the rain",
    "an old clock", "a wooden door", "a lighthouse", "a thunderstorm",
    "the desert", "a garden", "a small village", "a broken mirror",
    "a silent room", "a winter morning", "a summer night", "a busy harbor",
    "a quiet library", "an abandoned house", "a snow-covered field",
    "a candle", "a hidden valley", "an autumn leaf", "a river at dusk",
    "a distant star", "a stone bridge", "a field of wheat", "a foggy morning",
    "a cathedral", "a rocky shore", "a kitchen at dawn", "a train station",
    "an empty beach", "a country road", "a marketplace", "an old photograph",
    "a flock of birds", "a forgotten song", "a letter never sent",
    "a painter's studio", "a violin", "a pile of books", "a child's laughter",
    "a wedding day", "a farewell", "a reunion", "a promise kept",
    "a first snowfall", "a spring morning", "a warm fire", "a secret garden",
    "a cup of tea", "a walk in the park", "a summer breeze",
    "an orchard", "a thunderclap", "a sunset",
)

_CREATIVE_ADJECTIVES = (
    "mysterious", "ancient", "silent", "radiant", "forgotten",
    "bustling", "serene", "crumbling", "hidden", "glittering",
    "lonely", "enchanted", "weathered", "fragrant", "tranquil",
    "windswept", "moonlit", "sun-drenched", "ghostly", "tender",
    "restless", "melancholy", "joyful", "curious", "wild",
    "humble", "majestic", "frozen", "golden", "rusted",
    "delicate", "rugged", "quiet", "luminous", "stormy",
    "overgrown", "polished", "weary", "timeless", "fleeting",
)

_CREATIVE_BARE_NOUNS = (
    "cathedral", "violin", "lighthouse", "clock", "forest", "mountain",
    "garden", "village", "harbor", "library", "bridge", "staircase",
    "orchard", "kitchen", "doorway", "painting", "lantern", "meadow",
    "river", "shoreline", "temple", "fountain", "attic", "balcony",
    "cottage", "stream", "tavern", "courtyard", "canyon", "observatory",
    "greenhouse", "workshop", "warehouse", "chapel", "hillside",
    "corridor", "marketplace", "bakery", "studio", "monastery",
)

_CREATIVE_CHARACTERS = (
    "the old sailor", "a young traveler", "the village baker",
    "a curious child", "the last shepherd", "a tired merchant",
    "the librarian", "a forgotten poet", "the blacksmith's daughter",
    "an old soldier", "a wandering musician", "the gardener",
    "a retired teacher", "a lonely painter", "a young doctor",
    "the innkeeper", "a stranger in the night", "the ferryman",
    "a country priest", "an orphan boy", "an old watchmaker",
    "a quiet scholar", "a shy dancer", "the fisherman's wife",
    "a young apprentice", "a weary pilgrim", "the postmaster",
    "a determined explorer", "a cautious spy", "a kind neighbor",
)

_CREATIVE_TEMPLATES = (
    "Write a two-line poem about {noun}. ",
    "Describe a {adj} {noun} in one sentence. ",
    "Complete the story: Once upon a time, {character} ",
    "Write a short haiku about {noun}. ",
    "Describe {noun} at dawn in one sentence. ",
    "Begin a story with: {character} opened the door and ",
)


def sample_creative(rng: random.Random, n: int = 500) -> list[Prompt]:
    out: list[Prompt] = []
    seen: set[str] = set()
    attempts = 0
    while len(out) < n and attempts < n * 50:
        attempts += 1
        tmpl = rng.choice(_CREATIVE_TEMPLATES)
        if "{noun}" in tmpl and "{adj}" in tmpl:
            prompt = tmpl.format(
                adj=rng.choice(_CREATIVE_ADJECTIVES),
                noun=rng.choice(_CREATIVE_BARE_NOUNS),
            )
        elif "{noun}" in tmpl:
            prompt = tmpl.format(noun=rng.choice(_CREATIVE_NOUNS))
        elif "{character}" in tmpl:
            prompt = tmpl.format(character=rng.choice(_CREATIVE_CHARACTERS))
        else:
            prompt = tmpl
        if prompt in seen:
            continue
        seen.add(prompt)
        out.append((prompt, "creative"))
    return out


# ---------------------------------------------------------------------------
# factual — capitals, elements, constants, historical dates
# ---------------------------------------------------------------------------

_FACTUAL_COUNTRIES = (
    "France", "Germany", "Italy", "Spain", "Portugal", "Greece",
    "Japan", "China", "India", "Brazil", "Argentina", "Chile",
    "Canada", "Mexico", "Australia", "Egypt", "Kenya", "Nigeria",
    "Sweden", "Norway", "Finland", "Denmark", "Poland", "Ukraine",
    "Turkey", "Iran", "Thailand", "Vietnam", "Indonesia", "Malaysia",
    "South Korea", "Philippines", "Pakistan", "Bangladesh", "Iraq",
    "Saudi Arabia", "Morocco", "Algeria", "Tunisia", "Ethiopia",
    "South Africa", "Colombia", "Peru", "Venezuela", "Cuba",
    "Netherlands", "Belgium", "Switzerland", "Austria", "Hungary",
    "Czechia", "Romania", "Bulgaria", "Ireland", "Iceland",
    "New Zealand", "Ghana", "Tanzania", "Russia", "Mongolia",
)

_FACTUAL_CONSTANTS = (
    "The boiling point of water in Celsius is ",
    "The freezing point of water in Celsius is ",
    "The speed of light in a vacuum in meters per second is approximately ",
    "The acceleration due to gravity on Earth in meters per second squared is approximately ",
    "The number of seconds in one hour is ",
    "The number of minutes in one day is ",
    "The number of days in a common year is ",
    "The number of days in a leap year is ",
    "The smallest prime number is ",
    "The number of continents on Earth is ",
    "The number of planets in the solar system is ",
    "The distance from the Earth to the Sun in millions of kilometers is approximately ",
    "The chemical symbol for gold is ",
    "The chemical symbol for silver is ",
    "The chemical symbol for iron is ",
    "The chemical symbol for sodium is ",
    "The chemical symbol for potassium is ",
    "The chemical formula for water is ",
    "The chemical formula for carbon dioxide is ",
    "The chemical formula for table salt is ",
    "The number of bones in the adult human body is approximately ",
    "The number of chromosomes in a human cell is ",
    "The largest ocean on Earth is the ",
    "The longest river in the world is the ",
    "The tallest mountain on Earth is ",
    "The largest desert on Earth is the ",
    "The currency of Japan is the ",
    "The currency of the United Kingdom is the ",
    "The currency of India is the ",
    "The currency of Brazil is the ",
    "The main language spoken in Brazil is ",
    "The main language spoken in Argentina is ",
    "The main language spoken in Egypt is ",
    "The number of sides on a hexagon is ",
    "The number of sides on an octagon is ",
    "The sum of the angles of a triangle in degrees is ",
    "The square root of 144 is ",
    "The value of pi to two decimal places is ",
    "The largest planet in the solar system is ",
    "The smallest planet in the solar system is ",
)

_FACTUAL_EVENTS = (
    "The French Revolution began in the year ",
    "The American Declaration of Independence was signed in the year ",
    "The Berlin Wall fell in the year ",
    "World War II ended in the year ",
    "World War I began in the year ",
    "The first moon landing occurred in the year ",
    "Christopher Columbus reached the Americas in the year ",
    "The Magna Carta was signed in the year ",
    "The Great Fire of London happened in the year ",
    "The Titanic sank in the year ",
    "The Roman Empire fell in the west in the year ",
    "The printing press was invented around the year ",
    "The Eiffel Tower was completed in the year ",
    "The Suez Canal was opened in the year ",
    "The Panama Canal was opened in the year ",
    "The Russian Revolution happened in the year ",
    "The Wright brothers first flew in the year ",
    "The Internet became widely public in the year ",
    "The European Union was founded in the year ",
    "The Apollo 11 mission launched in the year ",
    "The Battle of Hastings took place in the year ",
    "The Declaration of the Rights of Man was adopted in the year ",
    "The Treaty of Versailles was signed in the year ",
    "The Cuban Missile Crisis occurred in the year ",
    "The fall of Constantinople happened in the year ",
    "The discovery of penicillin was announced in the year ",
    "The first successful heart transplant occurred in the year ",
    "Shakespeare died in the year ",
    "Napoleon was defeated at Waterloo in the year ",
    "The California Gold Rush began in the year ",
    "The Eiffel Tower construction began in the year ",
    "The first successful airplane flight was in the year ",
    "The Chernobyl disaster occurred in the year ",
    "The Soviet Union dissolved in the year ",
    "The Great Depression began in the year ",
    "The American Civil War began in the year ",
    "The American Civil War ended in the year ",
    "The first modern Olympic Games were held in the year ",
    "The United Nations was founded in the year ",
    "NATO was founded in the year ",
)

_FACTUAL_CAPITAL_FRAMINGS = (
    "The capital of {country} is ",
    "The capital city of {country} is known as ",
    "{country}'s capital is ",
    "The main city of {country} is ",
)

_FACTUAL_ELEMENT_FRAMINGS = (
    "Element number {num} on the periodic table is ",
    "The element with atomic number {num} is ",
    "On the periodic table, element {num} is ",
)

_FACTUAL_CATEGORIES = ("capital", "element", "constant", "event")


def sample_factual(rng: random.Random, n: int = 500) -> list[Prompt]:
    out: list[Prompt] = []
    seen: set[str] = set()
    attempts = 0
    while len(out) < n and attempts < n * 50:
        attempts += 1
        cat = rng.choice(_FACTUAL_CATEGORIES)
        if cat == "capital":
            country = rng.choice(_FACTUAL_COUNTRIES)
            framing = rng.choice(_FACTUAL_CAPITAL_FRAMINGS)
            prompt = framing.format(country=country)
        elif cat == "element":
            num = rng.randint(1, 100)
            framing = rng.choice(_FACTUAL_ELEMENT_FRAMINGS)
            prompt = framing.format(num=num)
        elif cat == "constant":
            prompt = rng.choice(_FACTUAL_CONSTANTS)
        else:
            prompt = rng.choice(_FACTUAL_EVENTS)
        if prompt in seen:
            continue
        seen.add(prompt)
        out.append((prompt, "factual"))
    return out


# ---------------------------------------------------------------------------
# builder
# ---------------------------------------------------------------------------

_SAMPLERS: tuple[tuple[str, Callable[[random.Random, int], list[Prompt]]], ...] = (
    ("multi", sample_multi_step_arith),
    ("single", sample_single_step_arith),
    ("trans", sample_translation),
    ("code", sample_code),
    ("creative", sample_creative),
    ("factual", sample_factual),
)


def build_broad_corpus(
    rng: random.Random,
    per_domain: int = 500,
) -> tuple[list[Prompt], dict[str, int]]:
    prompts: list[Prompt] = []
    counts: dict[str, int] = {}
    for label, fn in _SAMPLERS:
        domain_prompts = fn(rng, per_domain)
        counts[label] = len(domain_prompts)
        prompts.extend(domain_prompts)
    rng.shuffle(prompts)
    return prompts, counts
