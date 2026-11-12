### Plan
- DebugExpressionPublishing evaluates RUNNING nodes' debug expressions at publish_frequency; BackgroundMarkerPublisher thread renders/publishes
- CompiledSpatialTypeVisualizations shared with SpatialTypePublisher (sdt)
- Each goal decides: MotionGoal.publishes_debug_expressions, GiskardClientExecutor.execute(publishes_debug_expressions=...); Giskard always installs the extension switched off
- debug_mode and plot_* removed; PlotterMode removed from the benchmark; follow-up issue #49

### Done
- cram/main (757b90532f) merged and pushed at ec1894bdb8; PR #48 description matches it
- Draft upstream PR cram2#931 opened at ec1894bdb8 (contains #47's work too); labels api-break, new-feature, maintenance; description condensed
- Fork PRs #47 and #48 closed as superseded by cram2#931
- Bowl and cup benchmark ~0.9 % tick time; markers per node, orientations/normals at the tip

### Next
- Keep cram2#931's description current with each push
- Address cram2#931 review: all but compile spike pushed at e383eab826 + 8 threads resolved; compile-once design committed locally (not pushed), measured; next push, update description bullet "evaluates the RUNNING nodes in one compiled function", reply on :276 with numbers; try-finally reply + post_goal_plotters decision still open
- Before review: discuss the api-break with code owners / Cramgineers
- Follow-up PR: GIL tail latency of the marker thread (p99/max tick time on/off), issue #50
