"""Verify tq4_matvec_triton_v6 (int8 tl.dot tensor-core path) matches v2.

v6 uses Ada int8 tensor cores via tl.dot. Falls back to v2 when
BLOCK_M < 16 (tensor core constraint). Correctness gate same as v5:
cosine ≥ 0.999, max rel err ≤ 2%.
"""

import torch
from calm.llm_computer.tq4_torch import (
    build_pi, compute_lloyd_max_codebook, quantize_tq4,
)
from calm.llm_computer.tq4_triton import (
    tq4_matvec_triton_v2, tq4_matvec_triton_v6, _pick_block_m,
)


def compare_shape(in_f: int, out_f: int) -> bool:
    device = "cuda"
    torch.manual_seed(42)
    W = torch.randn(out_f, in_f, device=device, dtype=torch.float32) * 0.05
    pi = build_pi(device=device, source="torch")
    centroids, boundaries = compute_lloyd_max_codebook()
    centroids = centroids.to(device)
    boundaries = boundaries.to(device)

    qs_rows, d_rows = [], []
    for r in range(out_f):
        q = quantize_tq4(W[r], pi=pi, boundaries=boundaries)
        qs_rows.append(q.qs)
        d_rows.append(q.d)
    qs = torch.stack(qs_rows, dim=0).reshape(-1, 128).contiguous()
    d = torch.stack(d_rows, dim=0).reshape(-1).contiguous()

    x = torch.randn(in_f, device=device, dtype=torch.float32)
    bpr = in_f // 256
    x_rot = (x.reshape(bpr, 256) @ pi.T).reshape(in_f).contiguous()

    y_v2 = tq4_matvec_triton_v2(x_rot, qs, d, centroids, out_f, in_f)
    y_v6 = tq4_matvec_triton_v6(x_rot, qs, d, centroids, out_f, in_f)

    diff = (y_v2 - y_v6).abs().max().item()
    rel = diff / (y_v2.abs().max().item() + 1e-9)
    cos = torch.nn.functional.cosine_similarity(
        y_v2.unsqueeze(0), y_v6.unsqueeze(0)).item()

    block_m = _pick_block_m(out_f)
    path = "v6 tensor-core" if block_m >= 16 else "v2 fallback"
    cos_ok = cos >= 0.999
    rel_ok = rel <= 0.02
    ok = cos_ok and rel_ok
    print(f"  ({in_f}x{out_f}) BLOCK_M={block_m} [{path}]: "
          f"cos {cos:.5f} {'✓' if cos_ok else '✗'}, "
          f"rel {rel:.2%} {'✓' if rel_ok else '✗'} "
          f"→ {'OK' if ok else 'FAIL'}")
    return ok


def main():
    shapes = [
        (2560, 2048), (2560, 512), (2048, 2560),
        (2560, 10240), (10240, 2560),
    ]
    print("tq4_matvec_triton_v6 (tensor-core path) correctness vs v2:")
    print("  Gate: cosine ≥ 0.999 AND max rel err ≤ 2%")
    all_ok = True
    for in_f, out_f in shapes:
        all_ok = compare_shape(in_f, out_f) and all_ok
    print("\nALL OK" if all_ok else "\nFAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
