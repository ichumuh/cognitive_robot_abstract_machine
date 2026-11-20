PR: https://github.com/cram2/cognitive_robot_abstract_machine/pull/1038 (closes ichumuh#57)

Plan
- ActionServerTask pursues its goal only while running (cancel on pause/end/reset, resend on unpause, paused start sends nothing)

Done
- Fix, tests and PR opened at c5f6a9ae1d; labels bug-fix, api-break (cram2 has no `bug` label)
- Self-review addressed at 696d3f2c36 (cleanup cancels, answered goal kept across pause, no result request for cancelled goal, rename); all 9 threads replied, 4 resolved; description updated
- Open threads, no change: same-tick reset+restart (unreachable), five hook overrides (kept, user decision), Navigate crashes (follow-up PR), test coroutine (fixture is per-test)

Next
- Discuss the api-break with the giskardpy code owners before requesting review
- Separate issue/PR: NavigateActionServerTask.on_tick reads _result while it is None
