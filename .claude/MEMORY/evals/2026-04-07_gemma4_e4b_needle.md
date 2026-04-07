# Needle-in-Haystack Test

- **Model**: `/home/gabe/models/gemma-4-E4B-it-Q5_K_M.gguf`
- **Server context**: 131072 tokens
- **Sizes tested**: [4000, 16000, 32000, 64000, 100000]
- **Depths tested**: [10, 30, 50, 70, 90]
- **Total prompts**: 25

## Recall grid (size × depth)

| Size (tokens) | 10% | 30% | 50% | 70% | 90% |
|---:|:-:|:-:|:-:|:-:|:-:|
| **4,000** | ✓ | ✓ | ✓ | ✓ | ✓ |
| **16,000** | ✓ | ✓ | ✓ | ✓ | ✓ |
| **32,000** | ✓ | ✓ | ✓ | ✓ | ✓ |
| **64,000** | ✓ | ✓ | ✓ | ✓ | ✓ |
| **100,000** | ✓ | ✓ | ✓ | ✓ | ✓ |

## Timing grid (seconds)

| Size (tokens) | 10% | 30% | 50% | 70% | 90% |
|---:|---:|---:|---:|---:|---:|
| **4,000** | 5.0 | 5.4 | 5.7 | 5.2 | 4.9 |
| **16,000** | 9.0 | 9.3 | 9.6 | 3.4 | 6.4 |
| **32,000** | 14.2 | 11.8 | 11.6 | 10.9 | 4.6 |
| **64,000** | 27.7 | 24.9 | 24.8 | 19.7 | 13.4 |
| **100,000** | 54.2 | 48.0 | 36.1 | 35.0 | 24.6 |

## Legend

- ✓ = needle retrieved correctly (exact substring match)
- ✗ = retrieval failed (model returned wrong code or none)
