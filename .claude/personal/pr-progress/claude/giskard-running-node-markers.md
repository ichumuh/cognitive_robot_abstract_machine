### Plan
- DebugExpressionPublishing evaluates RUNNING nodes' debug expressions at publish_frequency; BackgroundMarkerPublisher thread renders/publishes
- CompiledSpatialTypeVisualizations shared with SpatialTypePublisher (sdt)
- Each goal decides: MotionGoal.publishes_debug_expressions, GiskardClientExecutor.execute(publishes_debug_expressions=...); Giskard always installs the extension switched off
- debug_mode and plot_* removed from GiskardServerConfig; PlotterMode removed from the benchmark; follow-up issue #49

### Done
- Pushed at bb756191 (running-node markers, tip anchoring); draft PR #48 description matches that commit
- Per-goal flag + debug_mode removal implemented and tested locally, *not committed* (user asked to hold the commit)

### Next
- User reviews the uncommitted change, then commit + push, update PR description (migration: debug_mode/plot_*/publish_debug_expressions gone) and the plan item notes
- is_in_github_workflow deleted (user approved)
- Before review: discuss the api-break with code owners / Cramgineers
