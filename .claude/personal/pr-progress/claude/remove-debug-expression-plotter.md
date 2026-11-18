## PR progress: cram2#1007 (claude/remove-debug-expression-plotter)

Plan
- Remove the unused debug expression trajectory/plotter/recording and collect_from; fix the debugging guide's RViz section

Done
- Removed everything listed and rewrote the guide (ef5fd2c4c3, 1bc950950c, feb1533b22)
- Local tests green at 1bc950950c: test_motion_statechart + publisher tests 503 passed
- Draft PR with api-break, maintenance; the code owner (the user) agreed to the break

Next
- Wait for CI, then mark ready for review when the user wants
