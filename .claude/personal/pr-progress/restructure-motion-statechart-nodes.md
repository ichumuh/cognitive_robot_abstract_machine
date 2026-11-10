Branch restructure-motion-statechart-nodes; draft PR cram2#924 against main (labels api-break, maintenance), pushed at 0699e0c0e1 (merge of cram/main ecc91acd96, the keyword-only refactor).

Plan:
- Merge giskardpy motion_statechart goals/ into tasks/, move the collision distance monitors to monitors/, rename ros2_nodes topic_monitor -> topics and ros_tasks -> action_servers.

Done:
- 18ed60ac82: all of the above, importers in giskardpy/coraplex/experiments/docs/tests updated, ORM regenerated locally. giskardpy tests 1219 passed; coraplex 687 passed, 3 failed only for missing causallearn.

Next:
- Re-ran after the merge: same results. Review; handle CI or comments when the user asks.
