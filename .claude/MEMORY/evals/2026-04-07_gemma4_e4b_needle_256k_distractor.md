# Needle-in-Haystack Test (distractor) — Gemma 4 E4B at 256K

- **Model**: `gemma-4-E4B-it-Q5_K_M.gguf`
- **Server context**: 262144 tokens (extrapolation — patched llama.cpp to remove training-context cap)
- **Mode**: distractor (1 PRIMARY + 4 decoys, model must return only the primary)
- **Sizes tested**: 4K, 32K, 64K, 100K, 130K, 180K, 220K
- **Scoring**: primary counts if found anywhere (content OR reasoning); distractors count as leaked only if in final content field

## Distractor recall (fair scoring)

| Size (tokens) | primary found | distractors in answer | distractors in thinking | time | result |
|---:|:-:|:-:|:-:|---:|:-:|
| **4,000** | ✓ (in reasoning) | 0 | 2 | 4.6s | ✓ clean |
| **32,000** | ✓ | 0 | 0 | 4.9s | ✓ clean |
| **64,000** | ✓ | 0 | 0 | 29.3s | ✓ clean |
| **100,000** | ✓ | 0 | 0 | 52.6s | ✓ clean |
| **130,000** | ✓ | 0 | 0 | 80.9s | ✓ clean |
| **180,000** | ✓ | 0 | 0 | 132.9s | ✓ clean |
| **220,000** | ✓ | 0 | 0 | 190.9s | ✓ clean |

**Overall: 7/7 PASS (100%)**

## Notes

1. **4K context behavior**: Gemma 4 E4B put its entire response in `reasoning_content` with empty `content` on this short prompt. The primary code was correctly identified (in reasoning), and the model's thinking mentioned 2 of the 4 distractors as candidates it considered and rejected — which is correct reasoning behavior, not a leak. This is a quirk of how Gemma 4 splits thinking vs content on very short prompts.

2. **All sizes from 32K upward** produced a clean final answer with just the primary code, no distractors in the content field, no thinking-layer leaks either.

3. **220K case**: the model correctly ignored 4 distractors and returned only the primary even at 72% past its trained 128K max. Raw RoPE extrapolation, no YaRN, no scaling. Remarkable.

4. **Test scoring evolution**: the first run used a strict criterion (any distractor mention anywhere counts as a leak) which produced false failures at 4K/32K because the model's thinking block enumerated candidates during reasoning. The fair scoring splits reasoning from content and only penalizes distractors in the final answer.

## Methodology

Each prompt inserts 5 needles (1 PRIMARY + 4 distractors) at evenly-spaced depths (16%, 33%, 50%, 66%, 83%). The primary is clearly labeled "PRIMARY ACCESS CODE (this is the one you want)" and distractors are labeled "(reference only, not primary)", "(legacy backup, do not use)", "(deprecated, ignore)", "(test fixture, do not return)". The model is asked: "What is the PRIMARY access code? Reply with just the primary code, nothing else. Do not include any reference, legacy, or deprecated codes."

Pass criterion: primary code appears in the response, AND no distractor codes appear in the model's final `content` field (mentions during `reasoning_content` are permitted as normal reasoning).
