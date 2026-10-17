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
Next: nothing open; wait for review.
