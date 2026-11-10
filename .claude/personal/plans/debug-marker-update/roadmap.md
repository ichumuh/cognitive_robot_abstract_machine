# Debug expression markers — standing rules

- Giskard evaluates and publishes the markers; observers never build nodes, since
  goal bindings (Bind_on_start) and WiggleInsert targets live only in Giskard's float
  variables.
- Expressions are compiled once and evaluated once per publish; never call
  SpatialType.evaluate() in a loop, it recompiles.
- The publishing worker thread handles only numpy arrays and ROS messages, never
  CasADi expressions (CasADi is not thread-safe); copy CompiledFunction output first.
- Without a selection only RUNNING nodes are shown.
- DebugExpression stays in giskardpy and SpatialTypeVisualization stays in the ROS
  adapter; cramph's inspector only sends a node selection and imports no giskardpy.
- Each goal decides whether its debug expressions are published
  (MotionGoal.publishes_debug_expressions); there is no server-wide debug switch.
