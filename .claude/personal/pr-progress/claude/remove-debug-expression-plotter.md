## PR progress: cram2#1007 (claude/remove-debug-expression-plotter)

Plan
- Remove the unused debug expression trajectory/plotter/recording; fix the debugging guide's RViz section

Done
- Removed the modules, exception, tests and guide section; rewrote the RViz section (ef5fd2c4c3)
- Local tests green: test_motion_statechart 478, publisher/motion_goal/orm 36
- Draft PR opened with labels api-break, maintenance

Next
- Ask whether `DebugExpression.collect_from` (now unused) can go too
- Discuss the api-break with giskardpy code owners before requesting review
