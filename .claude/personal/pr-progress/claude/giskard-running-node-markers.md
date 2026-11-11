### Plan
- DebugExpressionPublishing evaluates RUNNING nodes' debug expressions at publish_frequency; BackgroundMarkerPublisher thread renders/publishes
- CompiledSpatialTypeVisualizations shared with SpatialTypePublisher (sdt)
- Each goal decides: MotionGoal.publishes_debug_expressions, GiskardClientExecutor.execute(publishes_debug_expressions=...); Giskard always installs the extension switched off
- debug_mode and plot_* removed; PlotterMode removed from the benchmark; follow-up issue #49

### Done
- cram/main (757b90532f) merged and pushed at ec1894bdb8; PR #48 description matches it
- Bowl and cup benchmark ~0.9 % tick time; markers per node, orientations/normals at the tip

### Next
- #47 needs main merged (and the same Matrix.vstack keyword fix), else #48's diff shows main's changes
- Optional: migrate this branch's dataclasses/methods to the kw_only rules main brought in
- Before review: discuss the api-break with code owners / Cramgineers
