Branch remove-cartesian-velocity-limits (off cram/main 2789f7c001), pushed to origin at b8b6f6cb4d: delete the Cartesian velocity limit tasks; cruise velocity covers their uses. Closes #51, #52. PR not opened yet (user's call).

Plan:
- User opens the PR (description drafted in the session scratchpad); label api-break.

Done:
- Removal, coraplex migration to cruise velocity, tests; formatted, krrood check green.
- giskardpy motion_statechart 462 passed; coraplex 987 passed, 1 failure that also fails on cram/main.

Next:
- Ask whether error_signals.py (now used only by tests) can go.
