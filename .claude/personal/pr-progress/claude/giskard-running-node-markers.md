### Plan
- Tests first in test_debug_expression_publisher.py: per-node collection, only RUNNING shown, recompile only on shown-set change, publish rate, copies handed over, markers deleted after run, Giskard adds the extension only with debug_mode + publish_debug_expressions
- Rewrite DebugExpressionPublishing (collect per node, after_tick evaluate at publish_frequency, worker thread publishes newest values)
- Remove DebugExpressionPublisher; add GiskardServerConfig.publish_debug_expressions and wire it in Giskard.setup
- Run giskardpy debug publisher + launching tests in ROS env; krrood format/check

### Done
- Branch + draft PR #48 stacked on #47

### Next
- Write the failing tests
