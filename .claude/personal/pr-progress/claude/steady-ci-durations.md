<!-- BEGIN-PR-PROGRESS -->
### Plan
- Cut CI duration variance (cram2#981): deque ReferenceSearch instead of objgraph, own-library cache keys, per-script demo cache, small_packages job

### Done
- cram2#981 green and merged (a166e9d9fb, 2026-11-16)
- Measured on 4 runs after merge: slowest job median 23.6 -> 17.2 min, runner-minutes per run 124 -> 92, coraplex on AWS 6975P-C 60-min timeouts -> ~10 min pytest, leak tests 3-32 s each

### Next
- Later, separately: try `pytest -n 4` on the 4-vCPU runners; queueing (org concurrent-job limit) is now the largest remaining variance
<!-- END-PR-PROGRESS -->
