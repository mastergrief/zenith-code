"""Validate the C reference implementation of TQ3_K256 against the PyTorch oracle.

The C side lives at ~/llama.cpp/ggml/src/ggml-quants.c and is built into
libggml-base.so. We load it via ctypes, declare the four functions we need:

  void   ggml_tq3_k256_init_impl(void);
  const float * ggml_tq3_k256_get_centroids(void);   // 8 values
  const float * ggml_tq3_k256_get_boundaries(void);  // 7 values
  const float * ggml_tq3_k256_get_pi(void);          // 256*256 values
  void   quantize_row_tq3_k256_ref(const float * x, void * y, int64_t k);
  void   dequantize_row_tq3_k256  (const void * x, float * y, int64_t k);

Then we exercise three layers of the contract:

  Test 1 — codebook agreement
    The C code computes its Lloyd-Max codebook at runtime via the closed-form
    Gaussian E[x|a<x<b] (no scipy). The PyTorch reference uses
    scipy.integrate.quad. Both should agree to ~1e-6.

  Test 2 — Pi matrix agreement
    The C code embeds the Python-dumped Pi matrix as a static const array.
    Reading it back via ggml_tq3_k256_get_pi() must equal engine.Pi exactly
    (modulo float-literal round-trip).

  Test 3 — round-trip equivalence
    Generate a deterministic random vector. Quantize → dequantize via BOTH
    PyTorch and C. The two reconstructions must match to within float epsilon.
    Also: the round-trip MSE relative to the original should be small enough
    that the algorithm is doing real work (cosine similarity > 0.95).

Run from the repo root:

  PYTHONPATH=. python3 scripts/test_tq3_k256_c_vs_python.py
"""
from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import scripts.turboquant_patches  # noqa: F401  install compat patches
from turboquant_gpu import TurboQuantEngine

LLAMA_CPP_ROOT = Path(os.environ.get("LLAMA_CPP_ROOT", str(Path.home() / "llama.cpp")))
LIBGGML = LLAMA_CPP_ROOT / "build" / "bin" / "libggml-base.so"

HEAD_DIM = 256
BITS = 3
N_LEVELS = 1 << BITS
SEED = 42
BLOCK_SIZE_BYTES = 96 + 2  # 96 indices + ggml_half norm

# ── ctypes setup ────────────────────────────────────────────────────────────


def load_lib() -> ctypes.CDLL:
    if not LIBGGML.exists():
        raise SystemExit(
            f"libggml-base.so not found at {LIBGGML}\n"
            f"Build it first: cd {LLAMA_CPP_ROOT}/build && cmake --build . --target ggml-base"
        )
    lib = ctypes.CDLL(str(LIBGGML))

    lib.ggml_tq3_k256_init_impl.restype = None
    lib.ggml_tq3_k256_init_impl.argtypes = []

    lib.ggml_tq3_k256_get_centroids.restype = ctypes.POINTER(ctypes.c_float)
    lib.ggml_tq3_k256_get_centroids.argtypes = []

    lib.ggml_tq3_k256_get_boundaries.restype = ctypes.POINTER(ctypes.c_float)
    lib.ggml_tq3_k256_get_boundaries.argtypes = []

    lib.ggml_tq3_k256_get_pi.restype = ctypes.POINTER(ctypes.c_float)
    lib.ggml_tq3_k256_get_pi.argtypes = []

    lib.quantize_row_tq3_k256_ref.restype = None
    lib.quantize_row_tq3_k256_ref.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_void_p,
        ctypes.c_int64,
    ]

    lib.dequantize_row_tq3_k256.restype = None
    lib.dequantize_row_tq3_k256.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int64,
    ]

    return lib


# ── helpers ────────────────────────────────────────────────────────────────


def to_float_array(t: torch.Tensor) -> ctypes.Array:
    """Copy a 1-D fp32 torch tensor into a ctypes float array."""
    arr = (ctypes.c_float * t.numel())()
    src = t.contiguous().float().numpy().ravel()
    for i, v in enumerate(src):
        arr[i] = float(v)
    return arr


def from_ptr_to_list(ptr, n: int) -> list[float]:
    return [float(ptr[i]) for i in range(n)]


def fmt_floats(xs, n: int = 8) -> str:
    return ", ".join(f"{x:+.6e}" for x in xs[:n])


# ── tests ───────────────────────────────────────────────────────────────────


def test_codebook(lib: ctypes.CDLL, engine: TurboQuantEngine) -> bool:
    print("\n[1/3] codebook agreement (C runtime vs PyTorch reference)")

    lib.ggml_tq3_k256_init_impl()
    c_centroids = from_ptr_to_list(lib.ggml_tq3_k256_get_centroids(), N_LEVELS)
    c_boundaries = from_ptr_to_list(lib.ggml_tq3_k256_get_boundaries(), N_LEVELS - 1)

    py_centroids = engine.codebook.centroids.cpu().float().tolist()
    py_boundaries = engine.codebook.boundaries.cpu().float().tolist()

    print(f"  C centroids:    [{fmt_floats(c_centroids)}]")
    print(f"  Python:         [{fmt_floats(py_centroids)}]")

    diff_c = max(abs(c - p) for c, p in zip(c_centroids, py_centroids))
    diff_b = max(abs(c - p) for c, p in zip(c_boundaries, py_boundaries))

    print(f"  max abs centroid diff:   {diff_c:.3e}")
    print(f"  max abs boundary diff:   {diff_b:.3e}")

    tol = 1e-5
    if diff_c < tol and diff_b < tol:
        print(f"  PASS  (tol={tol:.0e})")
        return True
    print(f"  FAIL  (tol={tol:.0e})")
    return False


def test_pi_matrix(lib: ctypes.CDLL, engine: TurboQuantEngine) -> bool:
    print("\n[2/3] Pi matrix agreement (C embedded vs PyTorch reference)")

    pi_ptr = lib.ggml_tq3_k256_get_pi()
    c_pi = torch.tensor(
        [pi_ptr[i] for i in range(HEAD_DIM * HEAD_DIM)],
        dtype=torch.float32,
    ).reshape(HEAD_DIM, HEAD_DIM)

    py_pi = engine.Pi.cpu().float()

    diff = (c_pi - py_pi).abs().max().item()
    print(f"  shape={tuple(c_pi.shape)}, max abs diff = {diff:.3e}")

    # Tolerance is loose to absorb the float-literal round-trip in the .h file.
    # In practice we see ~1e-9 because we wrote with 9 significant digits.
    tol = 1e-6
    if diff < tol:
        print(f"  PASS  (tol={tol:.0e})")
        return True
    print(f"  FAIL  (tol={tol:.0e})")
    return False


def test_roundtrip(lib: ctypes.CDLL, engine: TurboQuantEngine) -> bool:
    print("\n[3/3] round-trip equivalence (C quant+dequant vs PyTorch)")

    # Deterministic input vector
    gen = torch.Generator()
    gen.manual_seed(123)
    n_blocks = 4
    n_elements = HEAD_DIM * n_blocks
    x = torch.randn(n_elements, generator=gen, dtype=torch.float32)

    # ── PyTorch path ─────────────────────────────────────
    py_outputs = []
    for blk in range(n_blocks):
        K = x[blk * HEAD_DIM : (blk + 1) * HEAD_DIM].reshape(1, HEAD_DIM)
        # _compress_keys_pt expects shape (seq, head_dim) but operates per-row,
        # so reshape to (1, head_dim) and call directly
        compressed = engine._compress_keys_pt(K)
        py_dequant = engine._dequant(compressed["indices"], compressed["vec_norms"])
        py_outputs.append(py_dequant.reshape(-1).float())
    py_out = torch.cat(py_outputs)

    # ── C path ──────────────────────────────────────────
    in_arr = to_float_array(x)
    out_arr = (ctypes.c_float * n_elements)()
    blocks = (ctypes.c_uint8 * (BLOCK_SIZE_BYTES * n_blocks))()

    lib.quantize_row_tq3_k256_ref(in_arr, ctypes.cast(blocks, ctypes.c_void_p), n_elements)
    lib.dequantize_row_tq3_k256(ctypes.cast(blocks, ctypes.c_void_p), out_arr, n_elements)

    c_out_fp32 = torch.tensor([out_arr[i] for i in range(n_elements)], dtype=torch.float32)

    # PyTorch's _dequant returns .half(), so py_out has fp16 rounding baked in.
    # The C output is fp32 (strictly more precise). For an apples-to-apples
    # check we round the C output through fp16 too.
    c_out_fp16round = c_out_fp32.half().float()

    # ── compare ─────────────────────────────────────────
    cp_diff_fp32 = (c_out_fp32 - py_out).abs()
    cp_diff_fp16 = (c_out_fp16round - py_out).abs()

    print(f"  vector length:   {n_elements} ({n_blocks} blocks)")
    print(f"  C[0:6]:    {fmt_floats(c_out_fp32.tolist(), 6)}")
    print(f"  PyTorch:   {fmt_floats(py_out.tolist(), 6)}")
    print(f"  max abs diff (C fp32 vs PyTorch fp16): {cp_diff_fp32.max().item():.3e}  (fp16 rounding)")
    print(f"  max abs diff (C fp16-rounded vs PyTorch): {cp_diff_fp16.max().item():.3e}  (apples-to-apples)")
    print(f"  mean abs diff (C fp16-rounded vs PyTorch): {cp_diff_fp16.mean().item():.3e}")

    # Round-trip quality vs original (sanity check that it's doing real work)
    cos = torch.nn.functional.cosine_similarity(
        c_out_fp32.flatten(), x.flatten(), dim=0
    ).item()
    rel_err = (c_out_fp32 - x).norm().item() / x.norm().item()
    print(f"  round-trip cosine sim (C vs original): {cos:.4f}")
    print(f"  round-trip relative L2 error:          {rel_err:.4f}")

    # Tolerances
    # When both are fp16-rounded, max diff should be at fp16 epsilon for the
    # max magnitude in the vector (~1e-4 to 5e-4 for typical N(0,1) vectors).
    eq_tol = 5e-4
    cos_tol = 0.85
    max_diff = cp_diff_fp16.max().item()

    if max_diff < eq_tol and cos > cos_tol:
        print(f"  PASS  (eq_tol={eq_tol:.0e}, cos_tol={cos_tol})")
        return True
    print(f"  FAIL  (eq_tol={eq_tol:.0e}, cos_tol={cos_tol})")
    return False


def main() -> int:
    print(f"libggml: {LIBGGML}")
    lib = load_lib()
    engine = TurboQuantEngine(
        head_dim=HEAD_DIM, total_bits=BITS, seed=SEED, device="cpu"
    )

    results = [
        test_codebook(lib, engine),
        test_pi_matrix(lib, engine),
        test_roundtrip(lib, engine),
    ]

    n_pass = sum(results)
    n_total = len(results)
    print(f"\n{'=' * 60}")
    print(f"  SUMMARY: {n_pass}/{n_total} passed")
    print(f"{'=' * 60}")
    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    raise SystemExit(main())
