Branch remove-cartesian-velocity-limits (off cram/main 2789f7c001), pushed to origin at 2f6bad8a54: delete the Cartesian velocity limit tasks and error_signals.py (only they used it); cruise velocity covers their uses. Closes #51, #52. PR not opened yet (user's call).

Plan:
- User opens the PR (description drafted in the session scratchpad); label api-break.

Done:
- Removal, coraplex migration to cruise velocity, tests; formatted, krrood check green.
- error_signals.py and its two tests removed (user agreed).
- giskardpy motion_statechart 460 passed; coraplex 987 passed, 1 failure that also fails on cram/main.

Next:
- Nothing until the user opens the PR.
