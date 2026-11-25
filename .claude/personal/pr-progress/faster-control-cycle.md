PR: cram2#1063 (faster-control-cycle) - QP filters copy values through a SubmatrixCache.

Plan
- SubmatrixCache remembers row/column selections per sparsity pattern; apply_filters and without_variables only copy values.

Done
- fd3235b576 (2026-11-25): cache in giskardpy; cse dropped (only -0.05 ms/tick for +0.38 s compile). Bench: tick 1.75 -> 1.49 ms.
- Pushed to origin, PR #1063 opened with new-feature (2026-11-25); giskardpy_test 1431 passed on fd3235b576.
- c42a864739 pushed (2026-11-25): module docstrings; CI green at c42a864739 (20/21, deploy-all-docs skipped).
- 3c9eab823c (2026-11-25): first 6 review threads addressed and resolved; CI green at 3c9eab823c; approved by maltehue and LucaKro.
- 64b3139859 pushed (2026-11-26): LucaKro's review: dof_filter -> active_world_state_indices (NDArray[np.intp]), no None defaults, field docstrings; `or` -> `is None` cache fallback. test_qp + test_motion_statechart 1009 passed. Thread replied+resolved, PR comment on `or`, api-break label + migration section in body.

Next
- CI at 64b3139859; discuss the dof_filter api-break with giskardpy code owners before merge.
