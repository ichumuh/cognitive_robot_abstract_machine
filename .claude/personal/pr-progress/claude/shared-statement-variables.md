## PR progress: cram2#1014 (claude/shared-statement-variables)

Plan
- Address review PRR_kwDOQga1Oc8AAAABZ83TFg (10 threads) test-first
- Item 2 (decided with the user): a the(...) counts the actions it describes instead of quantifying; not exactly one fails the trial, and in the real run ends the plan with a named PlanFailure

Done
- Merged fork main (2f2f5aafc9) into the branch
- Before the review: green locally at 8a18fd6079 (krrood test_eql 1607, cramph 909, coraplex 352 + 531)

Next
- Fix 1 (!= compare), 3 (reground under changed bindings), 4 (the binds its variables), 5 (no StopIteration), 6 (bindings cleared at end of run), nits; item 2 as planned
- Run krrood test_eql, cramph, coraplex tests; push; reply to and resolve each thread
- Discuss the api-break with the krrood/cramph/coraplex code owners
