# R90a — HospitalDeck verticalized composition demo

Per `augmentation_thesis.md` §'Customer verticals = card decks'.
Bundles ICD-10 recall + DaysBetween + NumberTheory under one
deck entry point. Demonstrates the 'deck per vertical' pattern
that scales to 100-domain hospital / legal / financial customers
via Level-2 MetaFacade (hours per domain per recursion.md).

## Corpus

9 probes: 3 ICD-10 (incl. tier-3 stubborn code),
3 date-arithmetic, 2 number-theory, 1 multi-step.

## Results

| metric | value |
|---|---:|
| total probes | 9 |
| route correct | 9/9 |
| answer correct | 9/9 |
| wall time | 68.0s |

## Shipped cards in deck

['icd10', 'base_conv', 'number_theory', 'multi_step', 'numeric_encode', 'factorial', 'fibonacci', 'combinations', 'permutations', 'power', 'next_prime', 'days_between', 'is_prime']

