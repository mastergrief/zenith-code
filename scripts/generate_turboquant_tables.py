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


BITS = 3

# (head_dim, seed, name_prefix). Different seeds per head_dim so the rotation
# matrices are independent — reusing the same Pi at different sizes would weaken
# the ensemble of rotations the model effectively gets across heterogeneous
# attention layers.
TABLES = [
    (256, 42, "TQ3_K256"),
    (512, 43, "TQ3_K512"),
]

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


def _emit_table_block(head_dim: int, seed: int, prefix: str) -> tuple[list[str], list, list]:
    """Generate the header lines for one (head_dim, seed) table.

    Returns (lines, centroids, boundaries) — centroids/boundaries are returned
    so the caller can print them for visual inspection.
    """
    print(f"  building table {prefix}: head_dim={head_dim}, bits={BITS}, seed={seed}")
    engine = TurboQuantEngine(
        head_dim=head_dim, total_bits=BITS, seed=seed, device="cpu"
    )

    Pi = engine.Pi.contiguous().cpu().float().numpy()
    assert Pi.shape == (head_dim, head_dim), f"unexpected Pi shape {Pi.shape}"

    centroids = engine.codebook.centroids.cpu().float().numpy()
    boundaries = engine.codebook.boundaries.cpu().float().numpy()
    assert centroids.shape == (1 << BITS,)
    assert boundaries.shape == ((1 << BITS) - 1,)

    n_levels = 1 << BITS
    pi_flat = Pi.reshape(-1).tolist()

    lines: list[str] = []
    lines.append(f"// ── {prefix} (head_dim={head_dim}, bits={BITS}, seed={seed}) ──")
    lines.append("")
    lines.append(f"#define {prefix}_HEAD_DIM   {head_dim}")
    lines.append(f"#define {prefix}_BITS       {BITS}")
    lines.append(f"#define {prefix}_N_LEVELS   {n_levels}")
    lines.append(f"#define {prefix}_SEED       {seed}")
    lines.append("")
    lines.append("// row-major (head_dim × head_dim) rotation matrix Pi, fp32")
    lines.append(_emit_array(f"{prefix}_PI", pi_flat))
    lines.append("")
    lines.append(f"// Reference Lloyd-Max codebook (n_levels = {n_levels})")
    lines.append(_emit_array(f"{prefix}_REFERENCE_CENTROIDS", centroids.tolist()))
    lines.append("")
    lines.append(_emit_array(f"{prefix}_REFERENCE_BOUNDARIES", boundaries.tolist()))
    lines.append("")

    return lines, centroids.tolist(), boundaries.tolist()


def main() -> int:
    print(f"generating turboquant tables → {OUT_PATH}")
    for head_dim, seed, prefix in TABLES:
        print(f"  - {prefix}: head_dim={head_dim}, seed={seed}")

    if not OUT_PATH.parent.exists():
        print(f"ERROR: target directory does not exist: {OUT_PATH.parent}", file=sys.stderr)
        print("Set LLAMA_CPP_ROOT or pass the path explicitly.", file=sys.stderr)
        return 1

    header: list[str] = []
    header.append("// AUTO-GENERATED — do not edit by hand.")
    header.append("// Source: scripts/generate_turboquant_tables.py in the zenith-code repo.")
    header.append("//")
    header.append("// One block per (head_dim, seed) pair. Each block defines the rotation")
    header.append("// matrix Pi (row-major, head_dim × head_dim) plus the Python-computed")
    header.append("// Lloyd-Max codebook for N(0, 1/d) as a reference for the C runtime tests.")
    header.append("//")
    header.append("// The C runtime computes its OWN codebook from scratch using a closed-form")
    header.append("// Gaussian E[x|a<x<b] (no RNG, no scipy needed). The REFERENCE arrays here")
    header.append("// exist purely so the C unit test can compare the runtime-computed codebook")
    header.append("// against the PyTorch one for byte-exact validation.")
    header.append("//")
    header.append("// Different head_dims use different seeds so the rotation matrices are")
    header.append("// independent across the heterogeneous-attention layer types in models")
    header.append("// like Gemma 4 26B-A4B (head_dim=256 SWA layers + head_dim=512 FA layers).")
    header.append("")
    header.append("#pragma once")
    header.append("")

    summaries: list[tuple[str, list, list]] = []
    for head_dim, seed, prefix in TABLES:
        lines, centroids, boundaries = _emit_table_block(head_dim, seed, prefix)
        header.extend(lines)
        summaries.append((prefix, centroids, boundaries))

    OUT_PATH.write_text("\n".join(header))
    size_kb = OUT_PATH.stat().st_size / 1024.0
    print(f"  wrote {OUT_PATH} ({size_kb:.1f} KB)")
    for prefix, centroids, boundaries in summaries:
        print(f"  {prefix} centroids:  {centroids}")
        print(f"  {prefix} boundaries: {boundaries}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
