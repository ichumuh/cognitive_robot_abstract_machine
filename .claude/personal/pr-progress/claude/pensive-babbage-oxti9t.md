## Plan
- Renderers declare `rendered_expression`; `render_markers` takes the evaluated ndarray
- `EvaluatedSpatialTypePublisher`: evaluated values in, stale-marker deletion, lock, no CasADi
- `SpatialTypePublisher(node=…, _world=…)`: builds its evaluated publisher from its node, compiles once via `CompiledFunctionWithViews`, passes copies

## Done
- Implemented with tests (55 passed locally with ROS message stand-ins); `krrood check` clean
- Review: renamed `SpatialTypeMarkerPublisher` to `EvaluatedSpatialTypePublisher`; publisher built inside, not injected (shared publishers would delete each other's markers)
- PR description and labels (`api-break`, `maintenance`) current

## Next
- CI on the ROS image
- Bootstrap commit carries an assistant co-author trailer; squash on merge or rewrite with the user's OK
