# Needle-in-Haystack Test (single)

- **Model**: `/home/gabe/models/gemma-4-E4B-it-Q5_K_M.gguf`
- **Server context**: 262144 tokens
- **Mode**: single
- **Sizes tested**: [4000, 32000, 64000, 100000, 130000, 180000, 220000]
- **Depths**: [10, 50, 90]


## Single-needle recall grid (size × depth)

| Size (tokens) | 10% | 50% | 90% |
|---:|:-:|:-:|:-:|
| **4,000** | ✓ | ✓ | ✓ |
| **32,000** | ✓ | ✓ | ✓ |
| **64,000** | ✓ | ✓ | ✓ |
| **100,000** | ✓ | ✓ | ✓ |
| **130,000** | ✓ | ✓ | ✓ |
| **180,000** | ✓ | ✓ | ✓ |
| **220,000** | ✓ | ✓ | ✓ |

## Single-needle timing grid (seconds)

| Size (tokens) | 10% | 50% | 90% |
|---:|---:|---:|---:|
| **4,000** | 5.5 | 5.4 | 5.5 |
| **32,000** | 14.2 | 15.2 | 10.9 |
| **64,000** | 32.9 | 26.3 | 26.4 |
| **100,000** | 59.8 | 44.5 | 34.2 |
| **130,000** | 80.3 | 58.3 | 56.6 |
| **180,000** | 135.7 | 104.0 | 99.6 |
| **220,000** | 192.6 | 143.4 | 153.6 |

## Legend

- ✓ = pass (all expected facts found, no leaks)
- ✗ = fail
