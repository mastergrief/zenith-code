#!/usr/bin/env python3
"""Gate-1 dry-executable all-family characterization (proven 2026-07-29)."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import torch
from calm.hrm_text_158.bit_linear import BitLinear
from calm.hrm_text_158.native_full_stack.trainer_sub2_authority import (
    build_trainer_sub2_authority_local_update_receipt,
    build_trainer_sub2_authority_roundtrip_receipt,
    build_sparse_vote_authority_landing_receipt,
    save_trainer_sub2_live_checkpoint_envelope,
    SPARSE_VOTE_AUTHORITY_MODE_FUSED_ONLY,
    SPARSE_VOTE_AUTHORITY_MODE_ORACLE_ON,
)

class T(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.lin = BitLinear(4, 4, bias=False)
    def forward(self, x):
        return self.lin(x)

def loss_fn(m, b):
    return torch.nn.functional.mse_loss(m(b["x"]), b["target"])

def output_fn(m, b):
    return m(b["x"])

def mk():
    m = T(); m.lin.weight.data.zero_(); m.lin.weight.data.add_(0.1)
    return m, {"x": torch.randn(2, 4), "target": torch.randn(2, 4)}

def landing(mode: str):
    m, b = mk()
    p1 = save_trainer_sub2_live_checkpoint_envelope(
        m, use_ternary_bulk=True, step=0, config={"proof": "char"}, source_pin="char", epoch=0
    )
    return build_sparse_vote_authority_landing_receipt(
        plan_sha256="0" * 64,
        task_id="char_v4",
        p1_checkpoint=p1,
        p1_envelope_bytes=b"{}",
        fresh_model_fn=lambda: mk()[0],
        batch=b,
        forward_loss_fn=loss_fn,
        forward_output_fn=output_fn,
        parity_max_abs_diff_by_site={"cache_builder": 0.0, "main_kl": 0.0, "retained_fallback": 0.0},
        use_ternary_bulk=True,
        device=torch.device("cpu"),
        sparse_vote_authority_mode=mode,
    )

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    art: dict = {}
    m, b = mk()
    r1 = build_trainer_sub2_authority_local_update_receipt(
        model=m, batch=b, forward_loss_fn=loss_fn, use_ternary_bulk=True,
        device=torch.device("cpu"), sparse_vote_authority_mode=SPARSE_VOTE_AUTHORITY_MODE_FUSED_ONLY,
    )
    vp = r1.vote_projection_proof
    s1 = vp.get("sparse_event_map_binding_sha256_by_key")
    if not isinstance(s1, dict) or not s1:
        print("MISSING B1 S1", file=sys.stderr); return 2
    pk = next(iter(r1.candidate_step_summary["candidate_local_update_proof_by_key"].values()))
    s2 = pk.get("candidate_bounded_decode_sha256_after")
    if not isinstance(s2, str) or len(s2) != 64:
        print("MISSING B1 S2", file=sys.stderr); return 2
    art["B1_fused"] = {
        "sparse_event_map_binding_sha256_by_key": s1,
        "candidate_bounded_decode_sha256_after_sample": s2,
        "S1_n": len(s1),
    }
    m, b = mk()
    r2 = build_trainer_sub2_authority_roundtrip_receipt(
        model=m, fresh_model_fn=lambda: mk()[0], batch=b, forward_loss_fn=loss_fn,
        use_ternary_bulk=True, device=torch.device("cpu"),
        sparse_vote_authority_mode=SPARSE_VOTE_AUTHORITY_MODE_FUSED_ONLY,
    )
    pr = r2.post_resume_update_proof
    cps = r2.checkpoint_payload_summary
    pre = cps.get("authoritative_state_payload_sha256")
    post = cps.get("post_update_authoritative_state_payload_sha256")
    s1b = pr.get("sparse_event_map_binding_sha256_by_key")
    if not isinstance(s1b, dict) or not s1b:
        print("MISSING B2 S1", file=sys.stderr); return 2
    if not (isinstance(pre, str) and len(pre) == 64 and isinstance(post, str) and len(post) == 64 and pre != post):
        print("MISSING B2 payload VALUE change", file=sys.stderr); return 2
    art["B2_roundtrip"] = {
        "sparse_event_map_binding_sha256_by_key": s1b,
        "authoritative_state_payload_sha256": pre,
        "post_update_authoritative_state_payload_sha256": post,
        "payload_changed": True,
    }
    r3 = landing(SPARSE_VOTE_AUTHORITY_MODE_FUSED_ONLY)
    sub = dict(r3.sparse_vote_authority_subproof)
    core = dict(r3.core_execution_identity)
    s1c = sub.get("sparse_event_map_binding_sha256_by_key")
    post3 = core.get("post_update_payload_sha256")
    if not isinstance(s1c, dict) or not s1c:
        print("MISSING B3 fused S1", file=sys.stderr); return 2
    if sub.get("oracle_only"):
        print("UNEXPECTED oracle_only under fused", file=sys.stderr); return 2
    if not (isinstance(post3, str) and len(post3) == 64):
        print("MISSING B3 fused post payload", file=sys.stderr); return 2
    art["B3_landing_fused"] = {
        "sparse_event_map_binding_sha256_by_key": s1c,
        "post_update_payload_sha256": post3,
        "oracle_only_absent_under_fused": True,
    }
    r4 = landing(SPARSE_VOTE_AUTHORITY_MODE_ORACLE_ON)
    sub = dict(r4.sparse_vote_authority_subproof)
    oo = dict(sub.get("oracle_only") or {})
    eeq = oo.get("events_equal_by_key")
    if not isinstance(eeq, dict) or not eeq:
        print("MISSING B3 oracle events_equal_by_key", file=sys.stderr); return 2
    if oo.get("dense_reference_tagged") != "oracle_only":
        print("MISSING B3 oracle tag", file=sys.stderr); return 2
    art["B3_landing_oracle_on"] = {
        "sparse_event_map_binding_sha256_by_key": sub.get("sparse_event_map_binding_sha256_by_key"),
        "events_equal_by_key": eeq,
        "dense_reference_tagged": oo.get("dense_reference_tagged"),
        "events_equal_fused_vs_dense_derived": oo.get("events_equal_fused_vs_dense_derived"),
        "oracle_only_absent_under_fused": False,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(art, indent=2, sort_keys=True) + "\n")
    print("CHAR_ALL_FAMILIES_OK")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
