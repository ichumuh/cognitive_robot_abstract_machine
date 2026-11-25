PR: cram2#1063 (faster-control-cycle) - QP filters copy values through a SubmatrixCache.

Plan
- SubmatrixCache remembers row/column selections per sparsity pattern; apply_filters and without_variables only copy values.

Done
- fd3235b576 (2026-11-25): cache in giskardpy; cse dropped (only -0.05 ms/tick for +0.38 s compile). Bench: tick 1.75 -> 1.49 ms.
- Pushed to origin, PR #1063 opened with new-feature (2026-11-25); giskardpy_test 1431 passed on fd3235b576.
- c42a864739 pushed (2026-11-25): module docstrings; CI green at c42a864739 (20/21, deploy-all-docs skipped).
- 3c9eab823c pushed (2026-11-25): all 6 review threads addressed (clip mode, warnings, take_submatrix/find_column_of_each_entry renames, staticmethods, docstring); #3 documented instead of indptr/nnz check, #5 helper not moved (YAGNI). test_qp 395 + test_motion_statechart 614 passed. Replied to and resolved all 6 threads; PR body updated.

Next
- Wait for CI at 3c9eab823c and further review (handle only when asked).
