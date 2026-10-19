PR: ichumuh/cognitive_robot_abstract_machine#14 (cramph -> main). Reviewers: LucaKro
(changes requested 2026-10-19, reviewed everything but cramph itself), Tigul.

Plan: answer Luca's unresolved review threads; do the simple ones (names, typing, docs)
right away, uncommitted until Simon says commit; propose the design ones in replies.

Done (2026-10-19, uncommitted in working tree):
- Renames: SimulatedPlanExecutor -> KinematicallySimulatedPlanExecutor,
  preceded_by -> copy_with_preceding_transformations, _on_the_copy ->
  _rebind_action_to_world, GraspCloseness.of/ranked -> from_grasp/sort_grasps_by_closeness,
  MovesHandle -> MovesContainerByItsHandle, _create_what_it_does -> _create_children,
  ReattachesBodies -> ReattachesBranches (+ MoveBranch.body -> branch_root),
  python_interface.py -> giskard_client_executor.py, _the_only_robot ->
  _single_robot_of_world, ForwardKinematicsBinding._set -> _set_root_T_tip,
  WorldVisualization._installed -> _create_installed_visualization
- DebuggingWithoutRosNode exception; executor no longer sets the coraplex log level
- TrainingEnvironment action_type ClassVar -> generic param; benchmark scenario names ->
  BenchmarkScenarioName StrEnum + classproperty
- NDArray typing in progress monitor, binding policy, qp_controller, qp_data
- collision avoidance: private field + forwarding property -> public init=False fields
- DifferentialDriveBaseGoal.sequence -> expanded_child_field
- tool path threshold uses resolved_position_threshold (test-first), demos ok
- doc wording; kin_sim/msc in giskardpy docs
- `expand = Action.expand` in ActionOfSteps is required (test_no_member_of_an_action_
  depends_on_the_order_of_its_bases), kept with a clearer comment
- demonstration debug tests assert executor.debug instead of the coraplex log level
- tests run 2026-10-19: cramph 831 ok, giskardpy motion_statechart+qp 776 ok, coraplex
  all touched suites ok (after fixing 6 rename fallouts); mixing/cutting/wiping demos ok
- replies posted on all of Luca's open threads + PR comment on findings 1/2

Next (proposed in replies, awaiting agreement):
- cramph: compile() ignores _hold_still() answer; settle() raises at exactly the pass
  budget (Luca's review-summary findings 1 and 2) - fix test-first
- fold DesignatorParameters into Action, drop krrood ParameterMetadata
- expose steps/drive/task kind as fields instead of boolean flags (SearchAction,
  DrivesBase, MovesToolCenterPoint straight line, ActionTrial diff-drive filter)
- tuples -> dataclasses in collision avoidance (+ sdt variable managers)
- Open/Close -> one generic grasped-mechanism goal; WorldWithRobot -> WorldSpecification
- merge ClosestGrasps/GraspsClosestToTheRobot; motion_designator.md rewrite
