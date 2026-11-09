Branch move-giskard-middleware-code on cram/main 8574ddfd0a; draft PR cram2#916 against main, pushed at 018a8a8419.

Plan:
- Move misplaced code out of giskardpy/middleware/ros2 and simplify what moved.

Done:
- 230c3338fc: ros2_interface -> SDT with node parameter, action_client/giskard_tester split, deletions, launch passes the node.
- 018a8a8419: MyActionClient -> GiskardActionClient.
- Draft PR #916 opened (labels Giskard, semDT, Refactor, api-break).

Next:
- Handle CI or review feedback when the user asks.
