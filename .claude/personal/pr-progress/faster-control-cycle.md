PR: cram2#1063 (faster-control-cycle) - QP filters copy values through a SubmatrixCache.

Plan
- SubmatrixCache remembers row/column selections per sparsity pattern; apply_filters and without_variables only copy values.

Done
- fd3235b576 (2026-11-25): cache in giskardpy; cse dropped (only -0.05 ms/tick for +0.38 s compile). Bench: tick 1.75 -> 1.49 ms.
- Pushed to origin, PR #1063 opened with new-feature (2026-11-25); giskardpy_test 1431 passed on fd3235b576.

Next
- Wait for CI and review (handle only when asked).
