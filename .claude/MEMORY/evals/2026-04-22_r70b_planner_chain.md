# R70b — PlannerFacade 2-step chain (Option C step-1)

Extends r70a single-facade dispatch with cross-domain chains:
'X in hex/binary/octal' where X is a primary sub-query evaluated
by number_theory / multi_step, then encoded via
`NumericEncodeFacade` (new this round, `numeric_encode.py`).

## Corpus

12 probes: 6 NumberTheory → base, 5 MultiStep → base, 1 direct
NumericEncode. All expected answers are short (1-8 char)
hex/binary/octal strings.

## Result

| metric | value |
|---|---:|
| corpus size | 12 |
| route correct | 12/12 |
| answer correct | 12/12 |
| wall time | 33.9s |

## Notes

Chain detect strips 'in <base>' suffix from the prompt and
re-classifies the remainder. Primary facade runs with
use_bias=False (we only want its computed integer value);
numeric_encode then runs with use_bias=True to deliver the
encoded form through Gemma's decode.
