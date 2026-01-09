import json

from geometry_msgs.msg import WrenchStamped

from giskardpy.motion_statechart.goals.templates import Sequence, Parallel
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.ros2_nodes.force_torque_monitor import (
    ForceImpactMonitor,
)
from giskardpy.motion_statechart.ros2_nodes.topic_monitor import PublishOnStart
from giskardpy.motion_statechart.test_nodes.test_nodes import ConstTrueNode
from giskardpy.ros_executor import Ros2Executor
from semantic_digital_twin.world import World


def test_force_impact_node(rclpy_node):
    topic_name = "force_torque_topic"
    publisher = rclpy_node.create_publisher(WrenchStamped, topic_name, 10)

    msg_below = WrenchStamped()

    msg_above = WrenchStamped()
    msg_above.wrench.force.x = 20.0

    msc = MotionStatechart()
    msc.add_node(
        parallel := Parallel(
            [
                ForceImpactMonitor(topic_name=topic_name, threshold=10),
                Sequence(
                    nodes=[
                        ConstTrueNode(),
                        PublishOnStart(topic_name=topic_name, msg=msg_below),
                        PublishOnStart(topic_name=topic_name, msg=msg_above),
                    ]
                ),
            ]
        )
    )
    msc.add_node(EndMotion.when_true(parallel))

    json_data = msc.to_json()
    json_str = json.dumps(json_data)
    new_json_data = json.loads(json_str)
    msc_copy = MotionStatechart.from_json(new_json_data)

    kin_sim = Ros2Executor(world=World(), ros_node=rclpy_node)
    kin_sim.compile(motion_statechart=msc_copy)

    publisher.publish(msg_above)

    # this should now finish
    kin_sim.tick_until_end()

    msc_copy.draw("muh.pdf")
