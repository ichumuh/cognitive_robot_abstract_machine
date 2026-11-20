## PR progress: cram2#1014 (claude/shared-statement-variables)

Plan
- a(Sequence)(nodes=[a(...), the(...)]): a chooses (trial), the refers (no trial, no retry), the outer a is the scope a trial covers

Done
- krrood generate under bound values; cramph StatementBindings, DefiniteStatementNode, ControlStructureNode.from_statement; coraplex trials carry bindings, chooser runs definite statements, root_around reaches the scope (71788d9f5d..8a18fd6079, merged cram2 main a720c71dad)
- Green locally at 8a18fd6079: krrood test_eql 1607, cramph 909, coraplex 352 + 531
- PR title and description updated for the rework

Next
- Wait for CI; discuss the api-break with the krrood/cramph/coraplex code owners
- Follow-up: let ReachabilityLocation range over an arm variable so TransportAction can share one
