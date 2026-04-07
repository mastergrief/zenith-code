# Needle-in-Haystack Test (distractor)

- **Model**: `/home/gabe/models/Qwen3.5-4B.Q5_K_M.gguf`
- **Server context**: 262144 tokens
- **Mode**: distractor
- **Sizes tested**: [4000, 32000, 64000, 100000, 130000, 180000, 220000]
- **Distractors per haystack**: 4


## Distractor recall (must return PRIMARY only, no decoys)

| Size (tokens) | primary found | distractors leaked | time | result |
|---:|:-:|---:|---:|:-:|
| **4,000** | ✓ | 0 | 13.9s | ✓ clean |
| **32,000** | ✓ | 0 | 17.4s | ✓ clean |
| **64,000** | ✗ | 0 | 36.5s | ✗ missed |
| **100,000** | ✗ | 0 | 62.4s | ✗ missed |
| **130,000** | ✓ | 0 | 88.7s | ✓ clean |
| **180,000** | ✓ | 0 | 146.9s | ✓ clean |
| **220,000** | ✓ | 0 | 200.5s | ✓ clean |

## Legend

- ✓ = pass (all expected facts found, no leaks)
- ✗ = fail
