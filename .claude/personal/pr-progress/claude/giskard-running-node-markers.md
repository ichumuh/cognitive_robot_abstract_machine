### Plan
- DebugExpressionPublishing evaluates RUNNING nodes' debug expressions at publish_frequency; BackgroundMarkerPublisher thread renders/publishes
- CompiledSpatialTypeVisualizations shared with SpatialTypePublisher (sdt)
- GiskardServerConfig.publish_debug_expressions + Giskard.create_debug_extensions

### Done
- Implemented and pushed at c8eeb729; draft PR #48 description current
- Local tests green: debug publisher 17, sdt publisher/renderer/node 48, launching + interface config 29

### Next
- Wait for CI on #48 and review; rebase onto main once #47 lands
- Integration check with a real Giskard goal in RViz (not done)
