## PR progress: cram2#1071 (claude/milk-scenario-on-main, draft)

Plan
- Show on main that one statement shares a variable across its nested steps (milk scenario), and fix what stops a(Sequence)(...) statements

Done
- Milk test: one action and a(Sequence) variants; trials move to the next detected milk
- Fixed SubtreeRebuild._rebuild_statement (rebuilt, not rebound_to) and StatementNode.grounded_nodes (WeakSet of yielded candidates, found by identity)
- Merged cram2 main d862caa911 (0a9e75c237, conflict in underspecified.py with #1047 resolved); pushed; draft PR opened, labels bug-fix + api-break
- Green locally at 0a9e75c237: cramph 928 passed; coraplex designator/plan/orm/training 859 passed; PR Tests section updated

Next
- Wait for CI on #1071
- Discuss the api-break (unfinished_nodes keyword-only, grounded_nodes by identity) with the cramph/coraplex code owners before review
- Personal notes ask for a `bug` label; cram2 has only `bug-fix`
