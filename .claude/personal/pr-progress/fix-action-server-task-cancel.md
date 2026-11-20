PR: https://github.com/cram2/cognitive_robot_abstract_machine/pull/1038 (closes ichumuh#57)

Plan
- ActionServerTask pursues its goal only while running (cancel on pause/end/reset, resend on unpause, paused start sends nothing)

Done
- Fix, tests and PR opened at commit on fix-action-server-task-cancel; labels bug-fix, api-break (cram2 has no `bug` label)

Next
- Discuss the api-break with the giskardpy code owners before requesting review
- Separate issue/PR: NavigateActionServerTask.on_tick reads _result while it is None
