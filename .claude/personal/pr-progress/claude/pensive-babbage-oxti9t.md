## Plan
- Renderers declare `rendered_expression`; `render_markers` takes the evaluated ndarray
- `SpatialTypeMarkerPublisher`: evaluated values in, stale-marker deletion, lock, no CasADi
- `SpatialTypePublisher`: injected marker publisher, compiles once via `CompiledFunctionWithViews`, passes copies

## Done
- Implemented, plus tests (53 passed locally with ROS message stand-ins); `krrood check` clean
- PR description and labels (`api-break`, `maintenance`) set
- Decided: composition over inheritance; variables read via `resolve()`; compiled state kept in `SpatialTypePublisher`

## Next
- Watch CI on the ROS image
- Bootstrap commit carries an assistant co-author trailer; squash on merge or rewrite with the user's OK
