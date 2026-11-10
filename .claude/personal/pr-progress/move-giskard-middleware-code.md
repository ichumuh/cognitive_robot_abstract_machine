Branch move-giskard-middleware-code; draft PR cram2#916 against main, pushed at aee83b2a1c.

Plan:
- Move misplaced code out of giskardpy/middleware/ros2 and simplify what moved.

Done:
- Moves, rename to GiskardActionClient, merge of cram/main.
- aee83b2a1c: LucaKro's review - ros2_interface -> ros_graph.RosGraph (defaults to registered node), rospy lifecycle only, inbox instead of nested callback, TopicNamesAndTypes. Threads replied and resolved; tests 654 passed at aee83b2a1c (UR errors local only).

Next:
- Follow-up PR: rename rospy to a lifecycle class, replace rospy.get_node() calls.
- Re-review by LucaKro; handle CI or comments when the user asks.
