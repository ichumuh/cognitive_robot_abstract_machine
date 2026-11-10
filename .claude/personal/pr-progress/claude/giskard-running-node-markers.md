### Plan
- DebugExpressionPublishing evaluates RUNNING nodes' debug expressions at publish_frequency; BackgroundMarkerPublisher thread renders/publishes
- CompiledSpatialTypeVisualizations shared with SpatialTypePublisher (sdt)
- Each goal decides: MotionGoal.publishes_debug_expressions, GiskardClientExecutor.execute(publishes_debug_expressions=...); Giskard always installs the extension switched off
- debug_mode, plot_* and is_in_github_workflow removed; PlotterMode removed from the benchmark; follow-up issue #49

### Done
- Pushed at cedce783; draft PR #48 description and plan item notes match it
- Bowl and cup benchmark ~0.9 % tick time; markers per node, orientations/normals at the tip

### Next
- Wait for CI and review; rebase onto main once #47 lands
- Before review: discuss the api-break with code owners / Cramgineers
