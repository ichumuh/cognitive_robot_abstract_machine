## ichumuh#28 actions-are-language-nodes (2026-10-17) - draft, pushed cb421620c

Base: ichumuh:cramph. Goal: fewer nodes / less nesting in plan statecharts
(bowl-and-cup: 221 -> 171 nodes, deepest depth 13 -> 7). Also carries the
not-yet-merged d570a2865, 7a26cffb9 (vectorized progress monitor), 56fef21cb
(pose goals Parallel).
Plan (approved 2026-10-17), all done:
(1) b70164317 statement gives up on its candidate: ChildChooser
prepare_chosen_child / failure_monitors_for, chosen_children,
failure_reasons(_of); coraplex PlanChildChooser rewrites /steps and hands it
Stalled; AttemptRewritingItsTask gone [done]
(2) 382295c71 delete Attempt, AttemptWithoutFailureMonitorsError,
RepeatOnStall; template tests: user chose to drop the ~5 abandoning ones and
keep the rest with self-failing alternatives [done]
(3) cb421620c actions are language nodes: Action (shared), GoalAction
(create_goal), SequentialAction/ParallelAction/TryAllAction (create_steps);
MoveTorso is a 1-step SequentialAction, Mixing/Cutting ParallelAction,
Wiping TryAllAction; action_body/replace_action_body/ActionNotExpanded gone;
Stretch hold wraps from outside, root action via new Statechart.replace
(user's choice), PlanExecutor reads plan_nodes after rewriting; shape base
must follow the action-derived bases (else dataclass mixins make actions
unhashable / compare by value) [done]
Tests: cramph 829, giskardpy statechart+child choice 433, coraplex 728 passed;
only place_setting (PR2, local) fails. Not run: experiments_test, real robot.
CI on cb421620c (2026-10-17): red. bullet_world + garmi demos fail with
CollisionViolatedError (fingertips vs drawer) on the real (not trial) pick-up
of the spoon from its drawer, right after the grasp: no LossOfSupport, so the
spoon never lifts. Trials identical to green base 0cdcda2a5. Culprit commit
not pinned: both demos fail locally on base too (base_link vs cabinet2,
0.0489 < 0.05), so local bisect is impossible. Suspects: d570a2865 (collision
refactor; dropped the pause-while-no-data monitor) or Attempt removal changing
step-transition ticks around the reattach. experiments: stretch
process-boundary test hit ObjectIsNotHeld in the client-side PlaceAction trial
(escapes ActionTrial, which only catches ExecutionFailure); passes locally (2
passed), base green -> likely a race.
Root cause found (user suspected d570a2865): MoveBranch in pick-up (close ->
reattach -> lift) and place (open -> reattach -> retract) ran 1 tick without
the gripper's collision allowance; old paused per-contact tasks observed 1
tick late and hid it, the new single task cancels. Fixed in 744a986fd
(reattach(..., holding_end_effector) runs MoveBranch in a Parallel with
UpdateTemporaryCollisionRules), tests in test_reattaching_bodies.py, pushed,
PR body updated. coraplex_test 730 passed; place_setting fails locally only
(pure geometry check -> local apartment differs from CI, also why demos fail
locally at start).
Next: confirm CI demos green; then the stretch ObjectIsNotHeld race.
Side note for user: d570a2865 left executor.statechart.draw('/tmp/muh.pdf')
in test/plan_running.py.
(4) 0b2b5ca5f collision allowances as an action field (approved 2026-10-17,
user chose a coraplex-only optimization, giskardpy untouched
(UpdateTemporaryCollisionRules, MoveGripper.allow_gripper_collision,
Open.grasping_end_effector stay). Action.temporary_collision_rules (init=False,
JSON-serialized, not in ORM) filled from create_temporary_collision_rules() in
expand; put in force on_start, out on_end/on_reset. Overridden by Reach,
Grasping, PickUp, Place (holding arm), MovesHandle, Pouring, ToolMotionAction,
MoveManipulator (if allow_gripper_collision). allow_gripper_collision removed
from tool_center_point_goal/gripper_goal; 744a986fd's holding_end_effector
Parallel reverted (action rule covers MoveBranch; its tests pass unchanged).
Mixing/Cutting steps flattened (trajectory+alignments directly), Justin torso
anchor also looks at the action itself. bowl-and-cup 179 -> 142 nodes, depth 7.
Test changes approved in the plan: 4 MoveManipulator/tool-center-point tests
and the mixing allowance test; plus lifting_goal_of helper (lift is bare
unless something accompanies it). coraplex 738 passed (only place_setting,
local), cramph 829, giskardpy statechart+child choice 433. Pushed, PR body
updated (221 -> 142 nodes), still draft. Next: CI on 0b2b5ca5f.
