# Slice B sparse cap GPU seam parity fixture

Multi-state event-coded sparse cap inputs for parity/residency tests.

- `state_a`: 4 elements, tensor offset 0
- `state_b`: 4 elements, tensor offset 4 (nonzero offset for index-class enforcement)
- Sparse backing via `event_coded_sparse_active_idx` / `event_coded_sparse_post_active_i32`
- Truly absent: full-numel `q_levels` CPU shim on the CUDA GPU-lane path
