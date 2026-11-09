### Plan
- DebugExpressionPublishing evaluates RUNNING nodes' debug expressions at publish_frequency; BackgroundMarkerPublisher thread renders/publishes
- CompiledSpatialTypeVisualizations shared with SpatialTypePublisher (sdt)
- GiskardServerConfig.publish_debug_expressions + Giskard.create_debug_extensions

### Done
- Implemented and pushed at 2e00c267; draft PR #48 description current
- Bowl and cup benchmark: ~0.9 % of tick time in the extension, real-time run 91.1 s vs 91.2 s
- Marker namespaces made unique per node (CartesianPose nodes collided)

### Next
- Decide on rotations drawn at the frame origin (CartesianPose orientation, AlignPlanes normals) - #47 / node side
- Wait for CI and review; rebase onto main once #47 lands
