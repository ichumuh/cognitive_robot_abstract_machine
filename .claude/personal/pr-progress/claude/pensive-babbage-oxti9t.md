## Plan
- Renderers declare `rendered_expression`; `render_markers` takes the evaluated ndarray; lookup is `find_spatial_type_renderer`
- `EvaluatedSpatialTypePublisher`: evaluated values in, stale-marker deletion, no CasADi, no lock (one publishing thread per instance)
- `SpatialTypePublisher(node=…, _world=…)`: builds its evaluated publisher from its node, compiles once via `CompiledFunctionWithViews`, passes copies

## Done
- Implemented with tests; `krrood check` clean
- Review round 1 (e4f9658a): lock removed, `renderer_for` renamed; both threads answered and resolved
- Real ROS 2 Jazzy in the cram CI image: 55 passed at e4f9658a; 233 passed (SDT `test_ros` without `test_world_synchronizer.py`) at d7a1ff06
- PR description and labels (`api-break`, `maintenance`) current

## Next
- `test_world_synchronizer.py` not run: ORM regeneration fails (`CouldNotResolveType: MetaData`, untouched `exceptions.py`) - ask the developer
- Bootstrap commit carries an assistant co-author trailer; squash on merge or rewrite with the user's OK
