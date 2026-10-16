## ichumuh#27 stretch-holds-its-base-heading (2026-10-16) - draft, pushed 06c39c780

Base: ichumuh:cramph. 8 commits: diff drive relative to the drive's parent
(+ binding policy, bound frame kept on re-parent, turn without arcing),
robot_adaptations on every executor (trials write odometry), cramph
CramLanguageNode.wrap, giskardpy HoldOrientation, coraplex
KeepStretchFromTurningWhileMovingItsTool (drive to reference-configuration
stand pose, then hold heading; MovesHandle and base-driving actions skipped),
DetectBeforeGrasp finds the final approach below wrappers.

In progress (2026-10-16, after review): drop the pre-positioning
NavigateAction - user decision: turning/arcing to reach is so bad that the
reach should fail instead; stand poses are the plan's job. Steps: (1) delete
_heading_to_reach/_base_T_tool_in_reference_configuration/_first_tool_goal
[done]; (2) coraplex_test: 7 failed - 3 structural (asserted the deleted
NavigateAction), 2 milk reaches + test_move_to_reach[stretch] needed stand
poses, place_setting (PR2) fails locally on the PR head too, unrelated
[done]; (3) stand poses: milk at (0.33,-1.98|-2.2, yaw pi/2), MoveToReach
Stretch offset (-0.55,0) (PR2 can't reach from it), new test: Stretch facing
away fails without turning (<2 deg) [done; coraplex_test 722 passed, only
place_setting fails]; (4) committed 87572f6db, pushed, PR description
updated [done].
Issue 1 (user asked to fix it, 2026-10-16): _holds_the_base checked an
ancestor's whole subtree, so the 2nd of sibling tool motions (PickAndPlace's
place) went unheld, and any HoldPose above disabled it. Fix: an ancestor
counts only if it is a Parallel with a HoldOrientation of this robot's base
as a direct child. Tests: successive reaches each held; HoldPose beside does
not count. Committed 09ea562c6, pushed, PR description updated [done];
coraplex_test 724 passed, only place_setting (PR2, local) fails.
Not run: experiments, real-Stretch demo (user OK with real-robot change).
Open: wrap not undone when a modify() block raises (documented);
double-transformation warning on Stretch reaches.
Review by ichumuh (2026-10-16, 6 threads), user agreed to fix all: drop
CramLanguageNode.wrap and _hand_over_place_of (+ ChildCannotBeWrappedError,
TopLevelNodeCannotBeWrappedError, ORM regen) for replace(reference, node) /
Action.replace_action_body(node) where `node` may hold the replaced child
(kept, owner wiring cleared); transformation uses
replace_action_body(Parallel([hold, body])); trim its docstring; reword
"run" (= parent of) in touched docstrings only, offer a wider rename in the
reply to thread 6 and leave that thread unresolved. Steps: (1) tests
[todo]; (2) implement [todo]; (3) ORM + test suites [todo]; (4) commit,
push, re-draft, PR description, reply + resolve threads 1-5 [todo].
