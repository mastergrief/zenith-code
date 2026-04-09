"""Generate the TurboQuant precomputed tables for the llama.cpp C port.

The TurboQuant algorithm needs two precomputed tables per (head_dim, bits):
  1. Rotation matrix Pi (head_dim × head_dim, fp32) — random orthogonal matrix
     derived from seed=42 via QR decomposition of a Gaussian.
  2. Lloyd-Max codebook (2^bits centroids + (2^bits)-1 boundaries, fp32) —
     optimal scalar quantizer for N(0, 1/d).

The C side could re-derive both tables at runtime, but:
  - Replicating PyTorch's torch.Generator(42) Mersenne Twister output exactly
    in C is fragile (the seeding + Box-Muller details aren't documented as a
    stable API).
  - Implementing QR decomposition in C for byte-exactness is more code than
    we want to maintain in the quant path.
  - The codebook needs scipy.integrate.quad in Python; the C side replaces
    that with the closed-form Gaussian E[x|a<x<b] which uses erff().

So: dump Pi from PyTorch (tiny — 256 KB for d=256) and embed as a C header.
The codebook is computed in C from scratch (closed-form, no RNG). We dump
the Python-computed codebook here too purely as a sanity reference for the
validation script, NOT for the C runtime.

Output: $LLAMA_CPP/ggml/src/turboquant_tables.h
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

# Allow running this script from anywhere — add the repo root to sys.path so
# we can import scripts.turboquant_patches and turboquant_gpu.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import scripts.turboquant_patches  # noqa: F401  installs compat patches
from turboquant_gpu import TurboQuantEngine


HEAD_DIM = 256
BITS = 3
SEED = 42

LLAMA_CPP_ROOT = Path(os.environ.get("LLAMA_CPP_ROOT", str(Path.home() / "llama.cpp")))
OUT_PATH = LLAMA_CPP_ROOT / "ggml" / "src" / "turboquant_tables.h"


def _format_float(f: float) -> str:
    """Format a float as a C99 hex literal so we get bit-exact reproducibility."""
    return f"{f:.9e}f"


def _emit_array(name: str, values, per_line: int = 8) -> str:
    lines = [f"static const float {name}[{len(values)}] = {{"]
    for i in range(0, len(values), per_line):
        chunk = values[i : i + per_line]
        lines.append("    " + ", ".join(_format_float(v) for v in chunk) + ",")
    lines.append("};")
    return "\n".join(lines)


def main() -> int:
    print(f"generating turboquant tables → {OUT_PATH}")
    print(f"  head_dim={HEAD_DIM}, bits={BITS}, seed={SEED}")

    if not OUT_PATH.parent.exists():
        print(f"ERROR: target directory does not exist: {OUT_PATH.parent}", file=sys.stderr)
        print("Set LLAMA_CPP_ROOT or pass the path explicitly.", file=sys.stderr)
        return 1

    engine = TurboQuantEngine(
        head_dim=HEAD_DIM, total_bits=BITS, seed=SEED, device="cpu"
    )

    # Pi is the (d × d) rotation matrix; PiT is its transpose. We dump Pi in
    # row-major order. The C side stores it the same way.
    Pi = engine.Pi.contiguous().cpu().float().numpy()
    assert Pi.shape == (HEAD_DIM, HEAD_DIM), f"unexpected Pi shape {Pi.shape}"

    centroids = engine.codebook.centroids.cpu().float().numpy()
    boundaries = engine.codebook.boundaries.cpu().float().numpy()
    assert centroids.shape == (1 << BITS,)
    assert boundaries.shape == ((1 << BITS) - 1,)

    n_levels = 1 << BITS

    pi_flat = Pi.reshape(-1).tolist()

    header = []
    header.append("// AUTO-GENERATED — do not edit by hand.")
    header.append("// Source: scripts/generate_turboquant_tables.py in the zenith-code repo.")
    header.append(f"// Generated for head_dim={HEAD_DIM}, bits={BITS}, seed={SEED}.")
    header.append("//")
    header.append("// Pi is the row-major (head_dim × head_dim) random orthogonal rotation")
    header.append("// matrix used by TurboQuant. The matching transpose PiT is recovered at")
    header.append("// runtime by indexing the columns of Pi.")
    header.append("//")
    header.append("// REFERENCE_centroids and REFERENCE_boundaries are the Python-computed")
    header.append("// Lloyd-Max codebook for N(0, 1/d). The C runtime computes its OWN")
    header.append("// codebook from scratch using a closed-form Gaussian E[x|a<x<b] (no RNG,")
    header.append("// no scipy needed). These reference values are exposed here so the C unit")
    header.append("// test can compare the runtime-computed codebook against the PyTorch one.")
    header.append("")
    header.append("#pragma once")
    header.append("")
    header.append(f"#define TQ3_K256_HEAD_DIM   {HEAD_DIM}")
    header.append(f"#define TQ3_K256_BITS       {BITS}")
    header.append(f"#define TQ3_K256_N_LEVELS   {n_levels}")
    header.append(f"#define TQ3_K256_SEED       {SEED}")
    header.append("")
    header.append("// row-major (head_dim × head_dim) rotation matrix Pi, fp32")
    header.append(_emit_array("TQ3_K256_PI", pi_flat))
    header.append("")
    header.append(f"// Reference Lloyd-Max codebook (n_levels = {n_levels})")
    header.append(_emit_array("TQ3_K256_REFERENCE_CENTROIDS", centroids.tolist()))
    header.append("")
    header.append(_emit_array("TQ3_K256_REFERENCE_BOUNDARIES", boundaries.tolist()))
    header.append("")

    OUT_PATH.write_text("\n".join(header))
    size_kb = OUT_PATH.stat().st_size / 1024.0
    print(f"  wrote {OUT_PATH} ({size_kb:.1f} KB)")
    print(f"  centroids: {centroids.tolist()}")
    print(f"  boundaries: {boundaries.tolist()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
