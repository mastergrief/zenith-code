# Needle-in-Haystack Test (multi)

- **Model**: `/home/gabe/models/Qwen3.5-4B.Q5_K_M.gguf`
- **Server context**: 262144 tokens
- **Mode**: multi
- **Sizes tested**: [4000, 32000, 64000, 100000, 130000, 180000, 220000]
- **Needles per haystack**: 5


## Multi-needle recall (must find ALL needles)

| Size (tokens) | found / expected | time | result |
|---:|---:|---:|:-:|
| **4,000** | 5/5 | 15.8s | ✓ |
| **32,000** | 5/5 | 21.9s | ✓ |
| **64,000** | 5/5 | 40.9s | ✓ |
| **100,000** | 5/5 | 67.8s | ✓ |
| **130,000** | 5/5 | 94.3s | ✓ |
| **180,000** | 4/5 | 153.4s | ✗ |
| **220,000** | 3/5 | 206.9s | ✗ |

## Legend

- ✓ = pass (all expected facts found, no leaks)
- ✗ = fail
