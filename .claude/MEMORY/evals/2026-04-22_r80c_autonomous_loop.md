# R80c — Autonomous loop demo (CALM-oracle → MetaFacade → install)

Closes the last missing link: given a Gemma-fail prompt, the system
infers the oracle signature, synthesizes a FacadeSpec, validates via
CALM, generates the facade, imports + installs, and answers correctly.

## Pipeline steps

1. baseline Gemma (wrong answer)
2. infer_oracle_signature → (fn_name, arity, operand_type, output_type)
3. MetaFacade.from_oracle → FacadeSpec
4. validate_facade (CALM safe_eval gate)
5. generate_facade (ast.parse-checked Python write)
6. import_facade_class + install + solve

## Results

| metric | value |
|---|---:|
| domains tested | 3 |
| baseline correct | 1/3 |
| loop correct | 2/3 |
| Δ | +1 |

## Per-domain

| domain | prompt | expected | baseline | loop |
|---|---|---|---:|---:|
| factorial | 'What is factorial of 13?' | 6227020800 | ✗ | ✓ |
| is_prime | 'Is 9973 prime?' | True | ✗ | ✗ |
| gcd | 'What is GCD of 420 and 150?' | 30 | ✓ | ✓ |

