## PR progress: cram2#1014 (claude/shared-statement-variables)

Plan
- Statements reusing an EQL variable share the choice: krrood generate under bound values, cramph StatementBindings, coraplex trials carry them

Done
- krrood 71788d9f5d, cramph a6af74e67b, coraplex + guide e01d19ac76
- Local tests green at e01d19ac76: krrood 1706, cramph 900, coraplex 346 + 364 + 4 passed
- Draft PR with new-feature, api-break; description lists the breaks and migration

Next
- Wait for CI; discuss the api-break with the krrood/cramph/coraplex code owners
- Follow-up: let ReachabilityLocation range over an arm variable so TransportAction can share one
