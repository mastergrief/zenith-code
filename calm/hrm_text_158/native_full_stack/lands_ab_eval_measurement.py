"""LANDS-AB measurement facade (IMPLEMENT_v3).

Thin re-export surface over seam modules. Prefer importing seams directly for new code.
Dependency: harness → this facade / seams → BDL/TSA (read-only). Never reverse.
"""
from __future__ import annotations

# seam a — twin apply
from calm.hrm_text_158.native_full_stack.lands_ab_eval_twin_apply import (  # noqa: F401
    assert_key_universe_complete,
    assert_twins_independent,
    build_twin_states_from_prior,
    clone_bounded_accumulator,
    clone_prior_states,
    events_maps_equal,
    key_universe_sha256,
    logical_int16_accumulator,
    prestate_digests,
    rank_spec_content_digest,
    require_canonical_rank_spec,
    required_keys_for_model,
    run_twin_apply_compare,
    scale_sha256,
    tensor_sha256,
    two_branch_dense_votes,
)

# seam b — fixture source
from calm.hrm_text_158.native_full_stack.lands_ab_eval_fixture_source import (  # noqa: F401
    DEFAULT_SOURCE_PINS,
    load_seed158_static_fixture,
    verify_recarry_receipt_ro,
    verify_source_pins,
)

# seam c — site measurement
from calm.hrm_text_158.native_full_stack.lands_ab_eval_site_measurement import (  # noqa: F401
    assert_phase_topology_complete,
    capture_weighted_grad_by_key,
    measure_from_production_capture,
    measure_g_cpu_static_ab,
    measure_oracle_events_equal,
    measure_site_apply_twin,
    phase_event_capture,
    run_s3_apply_equivalence_cpu,
    run_s3_apply_equivalence_cpu_tiny_diagnostic,
)
from calm.hrm_text_158.native_full_stack.lands_ab_eval_cuda_sites import (  # noqa: F401
    measure_b1_local_update_site,
    measure_b2_roundtrip_site,
    measure_b3_landing_site,
    measure_oracle_at_production_site,
)
from calm.hrm_text_158.native_full_stack.lands_ab_eval_oracle_sites import (  # noqa: F401
    measure_oracle_at_production_site as measure_oracle_at_production_site_direct,
)

# seam d — phase topology (pure)
from calm.hrm_text_158.native_full_stack.lands_ab_eval_phase_topology import (  # noqa: F401
    classify_phase_topology,
    synthesize_duplicate_start_events,
    synthesize_good_topology_events,
    synthesize_missing_coverage_events,
    synthesize_nested_start_events,
    topology_is_complete,
)
from calm.hrm_text_158.native_full_stack.lands_ab_eval_metric_reducer import (  # noqa: F401
    recompute_surface_cells_from_primitives,
    validate_metrics_schema,
    validate_required_key_universe,
)

# seam e — evidence contract
from calm.hrm_text_158.native_full_stack.lands_ab_eval_runtime_io import (  # noqa: F401
    harvest_exactly_one_raw_obs,
    o_excl_write_json,
    o_excl_write_text,
    resolve_run_scratch_dir,
    runtime_scratch_raw_path,
)
from calm.hrm_text_158.native_full_stack.lands_ab_eval_evidence_contract import (  # noqa: F401
    AUTHORIZED_RAW_FIELDS,
    EXPECTED_DEVICE_BY_ROW,
    build_eval_receipt_from_primitives,
    build_eval_receipt_from_raw_artifacts,
    build_eval_receipt_from_raw_observations,
    derive_matrix_from_raw_observations,
    load_and_validate_raw_artifact,
    make_raw_row_observation,
    o_excl_write_json,
    recompute_surface_cells_from_metrics,
    runtime_scratch_raw_path,
    validate_raw_row_observation,
)

# removed: complete_phase_cycle (manufactured missing flush — forbidden under v3)

from calm.hrm_text_158.native_full_stack.lands_ab_eval_metric_reducer import (  # noqa: F401
    recompute_surface_cells_from_primitives,
    validate_metrics_schema,
    validate_required_key_universe,
)
from calm.hrm_text_158.native_full_stack.lands_ab_eval_phase_topology import (  # noqa: F401
    synthesize_nested_start_events,
)

from calm.hrm_text_158.native_full_stack.lands_ab_eval_phase_jsonl import (  # noqa: F401
    ENV_JSONL,
    emit_enforcer_phase_pair,
    emit_one_enforcer_cycle_to_memory_and_jsonl,
    install_enforcer_jsonl_emitter,
    load_jsonl_events,
    make_enforcer_jsonl_emitter,
)

from calm.hrm_text_158.native_full_stack.lands_ab_eval_authoritative_payload import (  # noqa: F401
    authoritative_sidecar_payload_sha256,
)
from calm.hrm_text_158.native_full_stack.lands_ab_eval_production_post_state import (  # noqa: F401
    logical_int16_from_tensor_state,
    production_fused_apply_post_states,
    production_post_q_and_logical_acc_sha256_by_key,
)
from calm.hrm_text_158.native_full_stack.lands_ab_eval_production_binding import (  # noqa: F401
    bind_production_to_twin_landing,
    bind_production_to_twin_local_update,
    bind_production_to_twin_roundtrip,
    extract_landing_binding,
    extract_local_update_binding,
    extract_roundtrip_binding,
)
from calm.hrm_text_158.native_full_stack.lands_ab_eval_phase_jsonl import (  # noqa: F401
    emit_work_enclosing_cycle_with_sleep,
    phase_end,
    phase_start,
)
