PR: https://github.com/cram2/cognitive_robot_abstract_machine/pull/1038 (closes ichumuh#57)

Plan
- ActionServerTask pursues its goal only while running (cancel on pause/end/reset, resend on unpause, paused start sends nothing)
- Address self-review: cancel on cleanup, keep answered goal across pause, no result request for cancelled goal, rename to align_goal_with_life_cycle; reply to all 9 threads (keep the five hook overrides, user decision)

Done
- Fix, tests and PR opened at c5f6a9ae1d; labels bug-fix, api-break (cram2 has no `bug` label)

Next
- Tests first, then fixes, format, krrood check, push, reply/resolve threads
- Discuss the api-break with the giskardpy code owners before requesting review
- Separate issue/PR: NavigateActionServerTask.on_tick reads _result while it is None
