<!-- BEGIN-PR-PROGRESS -->
### Plan
- Cut CI duration variance: deque-based ReferenceSearch replaces objgraph in coraplex leak tests; restore keys limited to a job's own cache; per-script demo cache; small libraries share one `small_packages` job

### Done
- Analysis of 34 cram2 runs (queueing, objgraph list.pop(0), cross-lib cache restore, CPU lottery); leak tests 338 s -> 36 s locally
- Committed and pushed a31f37822d on origin/claude/steady-ci-durations; cram2#981 opened (labels bug-fix, maintenance); objgraph removed from pyprojects and Dockerfile

### Next
- Confirm cram2#981's CI passes: the workflow changes (libs JSON input, small_packages loop, demo cache save) are only validated by CI; actionlint was not run
- Compare coraplex/sdt job times on #981 against the earlier runs
- Later, separately: try `pytest -n 4` on the 4-vCPU runners
<!-- END-PR-PROGRESS -->
