"""Build-time TTIR manifest + integer-only static scan for credit-axis Triton kernels."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

KERNEL_MODULE = Path(__file__).resolve().parent / "integer_credit_axis_gpu_kernel.py"
DEFAULT_MANIFEST_DIR = Path(__file__).resolve().parent / "credit_axis_kernel_manifest"

# One manifest stage per @triton.jit kernel on the default native pipeline path.
STAGE_KERNELS: dict[str, str] = {
    "s1_attribution": "_credit_axis_attribution_row_tile_kernel",
    "s1_attribution_rescale": "_credit_axis_attribution_rescale_kernel",
    "s1_row_nz_count": "_credit_axis_row_nz_count_kernel",
    "s1_prefix_sum_exclusive": "_credit_axis_prefix_sum_exclusive_kernel",
    "s1_row_compact_scatter": "_credit_axis_row_compact_scatter_kernel",
    "s2_project_moves": "_credit_axis_project_moves_dense_kernel",
    "s2_dense_compact_scatter": "_credit_axis_dense_compact_scatter_kernel",
    "s3_gather_credit_q31": "_credit_axis_gather_attribution_kernel",
    "s3_credit_q31": "_credit_axis_credit_q31_kernel",
    "s4_rank_votes": "_credit_axis_grouped_bisect_right_rank_kernel",
    "s4_assign_bins_votes": "_credit_axis_assign_bins_votes_kernel",
}

FORBIDDEN_TTIR_PATTERNS = (
    "fp32",
    "bf16",
    "fp64",
    "sitofp",
    "fptosi",
    "fptoui",
    "uitofp",
)

FORBIDDEN_SOURCE_PATTERNS = (
    r"tl\.float32",
    r"tl\.bfloat16",
    r"\.to\(tl\.float32\)",
)


def _kernel_source_sha256() -> str:
    return hashlib.sha256(KERNEL_MODULE.read_bytes()).hexdigest()


def default_pipeline_triton_kernel_names() -> tuple[str, ...]:
    from calm.hrm_text_158.native_full_stack.integer_credit_axis_gpu_kernel import (
        DEFAULT_PIPELINE_TRITON_KERNEL_NAMES,
    )

    return DEFAULT_PIPELINE_TRITON_KERNEL_NAMES


def manifest_kernel_coverage_gap() -> tuple[str, ...]:
    """Kernel names launched on default path but missing from STAGE_KERNELS manifest."""
    covered = set(STAGE_KERNELS.values())
    return tuple(name for name in default_pipeline_triton_kernel_names() if name not in covered)


def list_orphan_manifest_artifacts(
    manifest_dir: Path,
    bundle: dict[str, Any],
) -> list[str]:
    """Committed .ttir / proof files not listed in manifest.json stages."""
    listed_stages = set(bundle.get("stages", {}))
    orphans: list[str] = []
    for path in sorted(manifest_dir.iterdir()):
        if path.name in ("manifest.json", "manifest_bundle.sha256"):
            continue
        if path.suffix == ".ttir":
            stage = path.stem
            if stage not in listed_stages:
                orphans.append(path.name)
        elif path.name.endswith(".integer_only_proof.json"):
            stage = path.name[: -len(".integer_only_proof.json")]
            if stage not in listed_stages:
                orphans.append(path.name)
    return orphans


def list_stale_manifest_proof_artifacts(
    manifest_dir: Path,
    bundle: dict[str, Any],
) -> list[str]:
    """Proof/ttir files whose embedded source_sha256 disagrees with manifest bundle."""
    expected = bundle.get("source_sha256")
    stale: list[str] = []
    for stage, proof in bundle.get("stages", {}).items():
        if proof.get("source_sha256") != expected:
            stale.append(f"{stage}.integer_only_proof.json")
        proof_path = manifest_dir / f"{stage}.integer_only_proof.json"
        if proof_path.is_file():
            on_disk = json.loads(proof_path.read_text(encoding="utf-8"))
            if on_disk.get("source_sha256") != expected:
                stale.append(proof_path.name)
    return sorted(set(stale))


def _scan_ttir_for_forbidden_float_ops(ttir_text: str) -> list[str]:
    violations: list[str] = []
    lowered = ttir_text.lower()
    for pattern in FORBIDDEN_TTIR_PATTERNS:
        if pattern in lowered:
            violations.append(f"ttir:{pattern}")
    return violations


def _scan_kernel_source_for_forbidden_float_ops(source_text: str) -> list[str]:
    violations: list[str] = []
    for pattern in FORBIDDEN_SOURCE_PATTERNS:
        if re.search(pattern, source_text):
            violations.append(f"source:{pattern}")
    return violations


def _extract_ttir_after_warmup(compiled_kernel: Any) -> str:
    asm = compiled_kernel.asm
    if isinstance(asm, dict) and "ttir" in asm:
        return str(asm["ttir"])
    raise RuntimeError("compiled kernel missing ttir artifact")


def _warmup_and_get_ttir(stage: str) -> str:
    import torch

    from calm.hrm_text_158.native_full_stack import integer_credit_axis_gpu_kernel as kernel_mod

    if stage == "s1_attribution":
        fn = kernel_mod._credit_axis_attribution_row_tile_kernel
        input_t = torch.zeros(2, 3, 8, dtype=torch.int32, device="cuda")
        grad_t = torch.zeros(2, 3, 4, dtype=torch.int32, device="cuda")
        acc_t = torch.zeros(8, dtype=torch.int64, device="cuda")
        fn[(1,)](
            input_t,
            grad_t,
            acc_t,
            0,
            2,
            3,
            8,
            0,
            8,
            input_t.stride(0),
            input_t.stride(1),
            input_t.stride(-1),
            grad_t.stride(0),
            grad_t.stride(1),
            grad_t.stride(-1),
            BLOCK=8,
        )
    elif stage == "s1_attribution_rescale":
        fn = kernel_mod._credit_axis_attribution_rescale_kernel
        acc_t = torch.zeros(8, dtype=torch.int64, device="cuda")
        out_t = torch.zeros(8, dtype=torch.int32, device="cuda")
        fn[(1,)](acc_t, out_t, 8, SHIFT=8, HALF=128)
    elif stage == "s1_row_nz_count":
        fn = kernel_mod._credit_axis_row_nz_count_kernel
        row_attrs = torch.zeros(8, dtype=torch.int32, device="cuda")
        row_nz = torch.zeros(1, dtype=torch.int32, device="cuda")
        fn[(1,)](row_attrs, row_nz, 8, 0)
    elif stage == "s1_prefix_sum_exclusive":
        fn = kernel_mod._credit_axis_prefix_sum_exclusive_kernel
        inp = torch.tensor([1, 2, 3], dtype=torch.int32, device="cuda")
        out = torch.zeros(3, dtype=torch.int32, device="cuda")
        fn[(1,)](inp, out, 3, MAX_N=4)
    elif stage == "s1_row_compact_scatter":
        fn = kernel_mod._credit_axis_row_compact_scatter_kernel
        row_attrs = torch.zeros(8, dtype=torch.int32, device="cuda")
        row_base = torch.zeros(1, dtype=torch.int32, device="cuda")
        out_flat = torch.zeros(4, dtype=torch.int64, device="cuda")
        out_attr = torch.zeros(4, dtype=torch.int32, device="cuda")
        fn[(1,)](row_attrs, row_base, out_flat, out_attr, 8, 0, IN_FEATURES=8)
    elif stage == "s2_project_moves":
        fn = kernel_mod._credit_axis_project_moves_dense_kernel
        flat_t = torch.zeros(4, dtype=torch.int64, device="cuda")
        attr_t = torch.zeros(4, dtype=torch.int32, device="cuda")
        q_t = torch.zeros(6, dtype=torch.int8, device="cuda")
        move_dense = torch.zeros(4, dtype=torch.int8, device="cuda")
        fn[(1,)](flat_t, attr_t, q_t, move_dense, 4, BLOCK=4)
    elif stage == "s2_dense_compact_scatter":
        fn = kernel_mod._credit_axis_dense_compact_scatter_kernel
        key = torch.zeros(4, dtype=torch.int64, device="cuda")
        val = torch.tensor([0, 1, 0, 2], dtype=torch.int8, device="cuda")
        out_key = torch.zeros(2, dtype=torch.int64, device="cuda")
        out_val = torch.zeros(2, dtype=torch.int8, device="cuda")
        fn[(1,)](key, val, out_key, out_val, 4, MAX_N=4)
    elif stage == "s3_gather_credit_q31":
        fn = kernel_mod._credit_axis_gather_attribution_kernel
        flat_t = torch.tensor([0, 1, 2, 3], dtype=torch.int64, device="cuda")
        attr_t = torch.tensor([10, 20, 30, 40], dtype=torch.int32, device="cuda")
        move_flat = torch.tensor([1, 3], dtype=torch.int64, device="cuda")
        out_attr = torch.zeros(2, dtype=torch.int32, device="cuda")
        fn[(1,)](flat_t, attr_t, 4, move_flat, out_attr, 2, MAX_EVENTS=4)
    elif stage == "s3_credit_q31":
        fn = kernel_mod._credit_axis_credit_q31_kernel
        attr = torch.tensor([10, 20], dtype=torch.int32, device="cuda")
        out = torch.zeros(2, dtype=torch.int32, device="cuda")
        fn[(1,)](attr, out, 2, BLOCK=2)
    elif stage == "s4_rank_votes":
        fn = kernel_mod._credit_axis_grouped_bisect_right_rank_kernel
        abs_t = torch.tensor([3, 1, 2], dtype=torch.int64, device="cuda")
        rank_t = torch.zeros(3, dtype=torch.int64, device="cuda")
        fn[(1,)](abs_t, rank_t, 3, MAX_N=4)
    elif stage == "s4_assign_bins_votes":
        fn = kernel_mod._credit_axis_assign_bins_votes_kernel
        rank = torch.tensor([1, 2, 3], dtype=torch.int64, device="cuda")
        moves = torch.tensor([1, -1, 1], dtype=torch.int8, device="cuda")
        votes = torch.zeros(3, dtype=torch.int16, device="cuda")
        fn[(1,)](rank, moves, votes, 3, LO_RANK=0, HI_LIMIT=4, VOTE_ABS=1, MAX_N=4)
    else:
        raise ValueError(f"unknown stage: {stage}")
    cache_entry = next(iter(fn.device_caches[0][0].values()))
    return _extract_ttir_after_warmup(cache_entry)


def _compile_stage_ttir(stage: str) -> str:
    return _warmup_and_get_ttir(stage)


def emit_manifest_bundle(
    *,
    emit_dir: Path,
    stages: tuple[str, ...],
) -> dict[str, Any]:
    emit_dir.mkdir(parents=True, exist_ok=True)
    source_sha256 = _kernel_source_sha256()
    source_text = KERNEL_MODULE.read_text(encoding="utf-8")
    source_violations = _scan_kernel_source_for_forbidden_float_ops(source_text)
    bundle: dict[str, Any] = {
        "schema_version": "hrm_text_158_credit_axis_kernel_manifest/v0",
        "source_sha256": source_sha256,
        "stages": {},
    }
    for stage in stages:
        kernel_name = STAGE_KERNELS[stage]
        ttir_text = _compile_stage_ttir(stage)
        ttir_path = emit_dir / f"{stage}.ttir"
        ttir_path.write_text(ttir_text, encoding="utf-8")
        ttir_violations = _scan_ttir_for_forbidden_float_ops(ttir_text)
        proof = {
            "stage": stage,
            "kernel_name": kernel_name,
            "pass": not ttir_violations and not source_violations,
            "ttir_violations": ttir_violations,
            "source_violations": source_violations,
            "allowed_dtypes": ["int8", "int16", "int32", "int64"],
            "forbidden_op_patterns": list(FORBIDDEN_TTIR_PATTERNS),
            "source_sha256": source_sha256,
        }
        proof_path = emit_dir / f"{stage}.integer_only_proof.json"
        proof_path.write_text(json.dumps(proof, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        bundle["stages"][stage] = proof
    manifest_path = emit_dir / "manifest.json"
    manifest_path.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    bundle_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    (emit_dir / "manifest_bundle.sha256").write_text(bundle_sha + "\n", encoding="utf-8")
    # Remove orphan artifacts from prior emits not listed in this bundle.
    listed = set(bundle["stages"])
    for path in list(emit_dir.iterdir()):
        if path.name in ("manifest.json", "manifest_bundle.sha256"):
            continue
        if path.suffix == ".ttir" and path.stem not in listed:
            path.unlink()
        elif path.name.endswith(".integer_only_proof.json"):
            stage = path.name[: -len(".integer_only_proof.json")]
            if stage not in listed:
                path.unlink()
    return bundle


def verify_kernel_manifest_bundle_at_launch(manifest_dir: Path | None = None) -> None:
    manifest_dir = manifest_dir or DEFAULT_MANIFEST_DIR
    manifest_path = manifest_dir / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("credit-axis kernel manifest missing")
    bundle_sha_path = manifest_dir / "manifest_bundle.sha256"
    if not bundle_sha_path.is_file():
        raise RuntimeError("credit-axis manifest_bundle.sha256 missing")
    expected_sha = bundle_sha_path.read_text(encoding="utf-8").strip()
    actual_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if expected_sha != actual_sha:
        raise RuntimeError("credit-axis kernel manifest stale")
    bundle = json.loads(manifest_path.read_text(encoding="utf-8"))
    if bundle.get("source_sha256") != _kernel_source_sha256():
        raise RuntimeError("credit-axis kernel manifest source_sha256 stale")
    coverage_gap = manifest_kernel_coverage_gap()
    if coverage_gap:
        raise RuntimeError(f"credit-axis kernel manifest coverage gap: {coverage_gap}")
    orphans = list_orphan_manifest_artifacts(manifest_dir, bundle)
    if orphans:
        raise RuntimeError(f"credit-axis kernel manifest orphan artifacts: {orphans}")
    stale = list_stale_manifest_proof_artifacts(manifest_dir, bundle)
    if stale:
        raise RuntimeError(f"credit-axis kernel manifest stale proof artifacts: {stale}")
    for stage, proof in bundle.get("stages", {}).items():
        if not proof.get("pass"):
            raise RuntimeError(f"credit-axis kernel manifest stage {stage} failed integer proof")


def main() -> None:
    parser = argparse.ArgumentParser(description="Emit credit-axis Triton kernel manifest")
    parser.add_argument("--emit-dir", type=Path, default=DEFAULT_MANIFEST_DIR)
    parser.add_argument(
        "--stages",
        default=",".join(STAGE_KERNELS),
        help="comma-separated stage ids",
    )
    args = parser.parse_args()
    stages = tuple(s.strip() for s in args.stages.split(",") if s.strip())
    coverage_gap = manifest_kernel_coverage_gap()
    if coverage_gap:
        raise SystemExit(f"STAGE_KERNELS missing default-path kernels: {coverage_gap}")
    bundle = emit_manifest_bundle(emit_dir=args.emit_dir, stages=stages)
    if not all(proof.get("pass") for proof in bundle["stages"].values()):
        raise SystemExit("integer-only proof failed for one or more stages")
    print(f"manifest emitted to {args.emit_dir}")


if __name__ == "__main__":
    main()
