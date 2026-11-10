Branch move-giskard-middleware-code; draft PR cram2#916 against main, pushed at 2edce878ba.

Plan:
- Move misplaced code out of giskardpy/middleware/ros2 and simplify what moved.

Done:
- Moves, GiskardActionClient rename, merge of cram/main.
- aee83b2a1c: LucaKro's review - ros_graph.RosGraph, inbox instead of nested callback, TopicNamesAndTypes; threads replied and resolved.
- 2edce878ba: rospy -> background_node (pure rename); PR description and rospy-thread reply updated. Tests 654 passed at 2edce878ba (UR errors local only).

Next:
- Re-review by LucaKro; handle CI or comments when the user asks.
