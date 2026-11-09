## Plan
- Renderers declare `rendered_expression(spatial_type)` (point: position, vector: components, pose-like: position + quaternion); `render_markers` takes the evaluated ndarray
- `SpatialTypeMarkerPublisher` (HasROS2Node): publishes evaluated visualizations, stale-marker deletion, lock, no CasADi
- `SpatialTypeVisualizationEvaluator`: compiles requests once via `CompiledFunctionWithViews`, evaluates once per publish, returns copies
- `SpatialTypePublisher` (StateChangeCallback): evaluator + marker publisher; `publish_once` keeps its signature
- Update giskardpy `DebugExpressionPublisher` and the tests

## Done
- Bootstrap, draft PR

## Next
- Tests first, then the renderer, evaluator and publishers
