## PR progress: cram2#1014 (claude/shared-statement-variables)

Plan
- a(Sequence)(nodes=[a(...), the(...)]): a chooses (trial), the refers (no trial, no retry), the outer a is the scope a trial covers

Done
- Review PRR_kwDOQga1Oc8AAAABZ83TFg addressed in 1f83573409 (merged fork main 2f2f5aafc9 first); all 10 threads replied to and resolved; PR description updated
- the(...) counts what it describes (generate_without_quantifier): not exactly one fails the trial, or ends the plan with DefiniteStatementDescribesNoSingleAction
- Green locally at 1f83573409: krrood test_eql 1618, cramph 916, coraplex core 853 + rest 291 (bullet-world place setting fails locally on clean checkout too)

Next
- Wait for CI; discuss the api-break with the krrood/cramph/coraplex code owners
- Known gap: a statement opened before a shared variable was bound keeps its candidates after it is bound (needs the variables a match reads)
- Follow-up: let ReachabilityLocation range over an arm variable so TransportAction can share one
