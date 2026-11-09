## Plan
- Renderers declare `rendered_expression`; `render_markers` takes the evaluated ndarray
- `EvaluatedSpatialTypePublisher`: evaluated values in, stale-marker deletion, lock, no CasADi
- `SpatialTypePublisher(node=…, _world=…)`: builds its evaluated publisher from its node, compiles once via `CompiledFunctionWithViews`, passes copies

## Done
- Implemented with tests; `krrood check` clean
- Real ROS 2 Jazzy run in the cram CI image at d7a1ff06: 55 passed (affected tests), 233 passed (SDT `test_ros` without `test_world_synchronizer.py`)
- PR description and labels (`api-break`, `maintenance`) current

## Next
- `test_world_synchronizer.py` not run: ORM regeneration fails (`CouldNotResolveType: MetaData`, untouched `exceptions.py`) - ask the developer
- Bootstrap commit carries an assistant co-author trailer; squash on merge or rewrite with the user's OK
