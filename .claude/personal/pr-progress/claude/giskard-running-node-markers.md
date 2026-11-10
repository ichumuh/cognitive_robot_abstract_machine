### Plan
- DebugExpressionPublishing evaluates RUNNING nodes' debug expressions at publish_frequency; BackgroundMarkerPublisher thread renders/publishes
- CompiledSpatialTypeVisualizations shared with SpatialTypePublisher (sdt)
- GiskardServerConfig.publish_debug_expressions + Giskard.create_debug_extensions
- Orientations and normals drawn at the tip (CartesianOrientation, AlignPlanes, FeatureFunctionGoal)

### Done
- Implemented and pushed at bb756191; draft PR #48 description current
- Bowl and cup benchmark: ~0.9 % of tick time in the extension, real-time run 91.1 s vs 91.2 s
- Marker namespaces unique per node; orientation/normal markers start at the tip (test + RViz finding fixed)

### Next
- Wait for CI and review; rebase onto main once #47 lands
- Before review: discuss the api-break with code owners / Cramgineers
