# Needle-in-Haystack Test (multi)

- **Model**: `/home/gabe/models/gemma-4-E4B-it-Q5_K_M.gguf`
- **Server context**: 262144 tokens
- **Mode**: multi
- **Sizes tested**: [4000, 32000, 64000, 100000, 130000, 180000, 220000]
- **Needles per haystack**: 5


## Multi-needle recall (must find ALL needles)

| Size (tokens) | found / expected | time | result |
|---:|---:|---:|:-:|
| **4,000** | 5/5 | 11.7s | ✓ |
| **32,000** | 5/5 | 20.9s | ✓ |
| **64,000** | 5/5 | 37.5s | ✓ |
| **100,000** | 5/5 | 67.6s | ✓ |
| **130,000** | 5/5 | 103.2s | ✓ |
| **180,000** | 5/5 | 157.2s | ✓ |
| **220,000** | 4/5 | 218.0s | ✗ |

## Legend

- ✓ = pass (all expected facts found, no leaks)
- ✗ = fail
