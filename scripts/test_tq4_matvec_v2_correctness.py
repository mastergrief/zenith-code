"""Verify tq4_matvec_triton_v2 produces same output as baseline."""

import torch
from calm.llm_computer.tq4_torch import (
    build_pi, compute_lloyd_max_codebook, quantize_tq4,
)
from calm.llm_computer.tq4_triton import (
    tq4_matvec_triton, tq4_matvec_triton_v2,
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

    y1 = tq4_matvec_triton(x_rot, qs, d, centroids, out_f, in_f)
    y2 = tq4_matvec_triton_v2(x_rot, qs, d, centroids, out_f, in_f)
    diff = (y1 - y2).abs().max().item()
    rel = diff / (y1.abs().max().item() + 1e-9)
    ok = torch.allclose(y1, y2, atol=1e-5)
    print(f"  ({in_f}x{out_f}): max abs diff {diff:.2e}, "
          f"rel {rel:.2e}, {'OK' if ok else 'FAIL'}")
    return ok


def main():
    shapes = [
        (2560, 2048), (2560, 512), (2048, 2560),
        (2560, 10240), (10240, 2560),
    ]
    print("tq4_matvec_triton_v2 correctness check:")
    all_ok = True
    for in_f, out_f in shapes:
        all_ok = compare_shape(in_f, out_f) and all_ok
    print("\nALL OK" if all_ok else "\nFAIL")


if __name__ == "__main__":
    main()
