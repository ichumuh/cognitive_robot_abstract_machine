Branch claude/statements-grounded-by-eql (stacked on claude/milk-scenario-on-main): let EQL ground the statements of a plan together.

Plan
- EQL: CombinationOrder.RANK_SUM; EvaluatesUnderBindings.evaluate_under -> Answer; nested `where` holds in construction
- cramph: StatementsGroundedTogether = one joint statement, under values fixed by other statements; CombinationRequirement
- coraplex: trial as SucceedsItsTrial predicate (TrialRequirement); grounding backend defaults to RANK_SUM
- ORM regeneration, krrood check / pre-commit, PR labels: api-break, new-feature

Done
- EQL steps + tests (krrood 1910 passed); cramph + coraplex grounding rewritten; designator/statement tests green before the latest fixes

Next
- rerun test_designator + cramph statechart after the exclusion/stale-value fixes and RANK_SUM default
- test_failure_diagnosis, ORM regeneration, pre-commit, commit (no push without asking)
