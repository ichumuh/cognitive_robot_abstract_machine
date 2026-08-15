"""
Tests for pushing a free-moving body onto a target pose with a single point of contact.
"""

from __future__ import annotations

import math

import numpy
import pytest

from giskardpy.motion_statechart.goals.pushing import (
    PlanarDisplacement,
    PoseTolerance,
    PushContact,
    PushOnce,
    PushSelector,
    PushToPose,
)
from giskardpy.motion_statechart.exceptions import NoImprovingPushError
from giskardpy.motion_statechart.goals.templates import Sequence
from giskardpy.motion_statechart.monitors.cartesian_monitors import PoseReached
from giskardpy.motion_statechart.monitors.progress_monitors import ProgressStalled
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.tasks.cartesian_tasks import (
    CartesianPosition,
    CartesianPositionStraight,
)
from giskardpy.executor import Executor
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types.derivatives import DerivativeMap
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
    Point3,
    Vector3,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    FixedConnection,
    PrismaticConnection,
)
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedom,
    DegreeOfFreedomLimits,
)
from semantic_digital_twin.world_description.world_entity import Body

# %% a square body to push

HALF_WIDTH = 0.1
"""
Half the edge length of the square body the selector tests push around.
"""

GYRATION_RADIUS = HALF_WIDTH * math.sqrt(2 / 3)
"""
The radius of gyration of a square of this half width about its centre.
"""

PUSHER_RADIUS = 0.02
"""
How far the pusher's centre sits off the surface it touches.
"""

STANDOFF_DISTANCE = 0.05
"""
How far behind the contact a push starts.
"""

MINIMUM_PUSH_DISTANCE = 0.01
"""
The shortest a single push may travel past the contact.

Kept well below the errors the tests use, so it never decides the geometry they assert.
"""

MAXIMUM_PUSH_DISTANCE = 0.08
"""
The longest a single push may travel past the contact.
"""

PUSH_GAIN = 0.5
"""
How much of the correction a push could make it aims to make.

Below one so that the tests asserting a push length read the length the selector worked
out rather than one of its two limits.
"""

PUSHING_HEIGHT = 0.02
"""
Height above the ground at which contact is made.
"""

TOLERANCE = PoseTolerance(position=0.02, orientation=0.2)
"""
How close to its target the body has to end up.

The two are in a ratio of a tenth of a metre per radian, so a turn shows up in the tests
as ten times its size in radians.
"""


def square_contacts() -> list[PushContact]:
    """
    The four face midpoints of a square body, each pushed towards the body's centre.

    A square is used rather than a T so that every expected choice can be worked out by
    hand: the four contacts are symmetric, so only the pose error can decide between
    them.

    :return: One contact per face, ordered ``+x``, ``-x``, ``+y``, ``-y``.
    """
    return [
        PushContact(
            point=Point3(x=HALF_WIDTH, y=0.0, z=0.0),
            direction=Vector3(x=-1.0, y=0.0, z=0.0),
        ),
        PushContact(
            point=Point3(x=-HALF_WIDTH, y=0.0, z=0.0),
            direction=Vector3(x=1.0, y=0.0, z=0.0),
        ),
        PushContact(
            point=Point3(x=0.0, y=HALF_WIDTH, z=0.0),
            direction=Vector3(x=0.0, y=-1.0, z=0.0),
        ),
        PushContact(
            point=Point3(x=0.0, y=-HALF_WIDTH, z=0.0),
            direction=Vector3(x=0.0, y=1.0, z=0.0),
        ),
    ]


def offset_contacts() -> list[PushContact]:
    """
    Two contacts on the ``+y`` face, either side of the body's centre.

    Both push in the same direction, so only their lever arm about the centre tells them
    apart - which is exactly what the turning part of the score is supposed to weigh.

    :return: The contact at ``+x`` first, the one at ``-x`` second.
    """
    return [
        PushContact(
            point=Point3(x=HALF_WIDTH, y=HALF_WIDTH, z=0.0),
            direction=Vector3(x=0.0, y=-1.0, z=0.0),
        ),
        PushContact(
            point=Point3(x=-HALF_WIDTH, y=HALF_WIDTH, z=0.0),
            direction=Vector3(x=0.0, y=-1.0, z=0.0),
        ),
    ]


def contacts_across_one_face() -> list[PushContact]:
    """
    Three contacts on the ``+x`` face, all pushing along ``-x``.

    They differ only in how far off the centre line they sit, so they offer the same
    slide and three different turns, which is what a body needing both asks for.

    :return: The contact towards ``+y`` first, then the middle one, then the one towards
        ``-y``.
    """
    quarter_width = HALF_WIDTH / 2
    return [
        PushContact(
            point=Point3(x=HALF_WIDTH, y=offset, z=0.0),
            direction=Vector3(x=-1.0, y=0.0, z=0.0),
        )
        for offset in (quarter_width, 0.0, -quarter_width)
    ]


def build_selector(contacts: list[PushContact]) -> PushSelector:
    """
    A selector over ``contacts`` with the module's constants.

    :param contacts: The contacts the selector chooses between.
    :return: The new selector.
    """
    return PushSelector(
        contacts=contacts,
        centroid=Point3(),
        gyration_radius=GYRATION_RADIUS,
        pusher_radius=PUSHER_RADIUS,
        standoff_distance=STANDOFF_DISTANCE,
        minimum_push_distance=MINIMUM_PUSH_DISTANCE,
        maximum_push_distance=MAXIMUM_PUSH_DISTANCE,
        push_gain=PUSH_GAIN,
        pushing_height=PUSHING_HEIGHT,
    )


def planar_pose(x: float = 0.0, y: float = 0.0, yaw: float = 0.0) -> numpy.ndarray:
    """
    :param x: Position along x.
    :param y: Position along y.
    :param yaw: Heading around z, in radians.
    :return: The matching homogeneous transformation matrix.
    """
    return HomogeneousTransformationMatrix.from_xyz_rpy(x=x, y=y, yaw=yaw).to_np()


def push_length(selected) -> float:
    """
    :param selected: A chosen push.
    :return: How far past the contact it travels, in metres.
    """
    return float(numpy.linalg.norm(selected.follow_through - selected.contact_point))


# %% weighing a slide against a turn


def test_a_turn_is_worth_a_slide_at_the_ratio_of_the_two_tolerances():
    """
    Being turned by the whole orientation tolerance is exactly as much of a miss as
    being displaced by the whole position tolerance, which is what fixes the rate at
    which one converts into the other.
    """
    assert TOLERANCE.rotation_radius == pytest.approx(
        TOLERANCE.position / TOLERANCE.orientation
    )


def test_a_turn_counts_as_the_arc_it_sweeps():
    """
    A displacement is compared as three lengths, so its turn has to be given one.
    """
    displacement = PlanarDisplacement(
        translation=numpy.array([0.3, -0.4]), rotation=2.0
    )

    numpy.testing.assert_allclose(
        displacement.to_lengths(rotation_radius=0.1), [0.3, -0.4, 0.2]
    )


# %% choosing a push


def test_a_displaced_body_is_pushed_from_the_face_it_should_move_away_from():
    """
    Shoving a body along ``-x`` means standing on its ``+x`` face, since a point contact
    can only push.
    """
    contacts = square_contacts()
    selector = build_selector(contacts)

    selected = selector.select(
        root_T_body=planar_pose(x=0.5),
        root_T_target=planar_pose(),
        tolerance=TOLERANCE,
    )

    assert selected.contact is contacts[0]


def test_a_body_displaced_the_other_way_is_pushed_from_the_opposite_face():
    """
    The choice follows the error rather than any preferred face.
    """
    contacts = square_contacts()
    selector = build_selector(contacts)

    selected = selector.select(
        root_T_body=planar_pose(x=-0.5),
        root_T_target=planar_pose(),
        tolerance=TOLERANCE,
    )

    assert selected.contact is contacts[1]


def test_a_turned_body_is_pushed_where_the_push_turns_it_back():
    """
    Two contacts pushing the same way are told apart by their lever arm: turning a body
    clockwise means pushing the side whose torque about the centre is clockwise.
    """
    contacts = offset_contacts()
    selector = build_selector(contacts)

    selected = selector.select(
        root_T_body=planar_pose(yaw=0.4),
        root_T_target=planar_pose(),
        tolerance=TOLERANCE,
    )

    assert selected.contact is contacts[0]


def test_a_body_turned_the_other_way_is_pushed_on_its_other_side():
    """
    The lever arm that turns a body back reverses with the sign of the error.
    """
    contacts = offset_contacts()
    selector = build_selector(contacts)

    selected = selector.select(
        root_T_body=planar_pose(yaw=-0.4),
        root_T_target=planar_pose(),
        tolerance=TOLERANCE,
    )

    assert selected.contact is contacts[1]


def test_a_body_that_is_only_displaced_is_pushed_through_its_centre():
    """
    Pushing off the centre would turn a body that is already pointing the right way, and
    that turn would then have to be undone.
    """
    contacts = contacts_across_one_face()
    selector = build_selector(contacts)

    selected = selector.select(
        root_T_body=planar_pose(x=0.05),
        root_T_target=planar_pose(),
        tolerance=TOLERANCE,
    )

    assert selected.contact is contacts[1]


def test_a_body_that_is_both_displaced_and_turned_is_pushed_off_centre():
    """
    A push slides and turns a body at once, so a body needing both is best served by one
    push doing some of each rather than by two pushes each undoing the other's work.
    """
    contacts = contacts_across_one_face()
    selector = build_selector(contacts)

    selected = selector.select(
        root_T_body=planar_pose(x=0.05, yaw=0.4),
        root_T_target=planar_pose(),
        tolerance=TOLERANCE,
    )

    assert selected.contact is contacts[2]


def test_an_orientation_error_is_measured_the_short_way_around():
    """
    A body turned just past half a revolution is barely turned the other way, so it is
    straightened by turning on rather than back.
    """
    contacts = offset_contacts()
    selector = build_selector(contacts)

    just_past_half_turn = selector.select(
        root_T_body=planar_pose(yaw=math.pi + 0.2),
        root_T_target=planar_pose(),
        tolerance=TOLERANCE,
    )
    just_under_half_turn = selector.select(
        root_T_body=planar_pose(yaw=math.pi - 0.2),
        root_T_target=planar_pose(),
        tolerance=TOLERANCE,
    )

    assert just_past_half_turn.contact is not just_under_half_turn.contact


def test_a_body_already_on_its_target_has_nothing_worth_pushing():
    """
    Every push would move the body away from where it already is, so choosing the least
    bad one would be worse than refusing.
    """
    selector = build_selector(square_contacts())

    with pytest.raises(NoImprovingPushError):
        selector.select(
            root_T_body=planar_pose(),
            root_T_target=planar_pose(),
            tolerance=TOLERANCE,
        )


# %% where the push travels


def test_the_push_runs_from_behind_the_contact_to_beyond_it():
    """
    The three points of a push lie on one line through the contact, spaced by the
    standoff behind it and the push distance past it, and the pusher's own radius keeps
    it off the surface.

    The error here is far larger than one push may correct, so the push runs its full
    permitted length.
    """
    contacts = square_contacts()
    selector = build_selector(contacts)

    selected = selector.select(
        root_T_body=planar_pose(x=0.5),
        root_T_target=planar_pose(),
        tolerance=TOLERANCE,
    )

    contact_surface_x = 0.5 + HALF_WIDTH
    numpy.testing.assert_allclose(
        selected.contact_point,
        [contact_surface_x + PUSHER_RADIUS, 0.0, PUSHING_HEIGHT],
        atol=1e-9,
    )
    numpy.testing.assert_allclose(
        selected.standoff,
        [contact_surface_x + PUSHER_RADIUS + STANDOFF_DISTANCE, 0.0, PUSHING_HEIGHT],
        atol=1e-9,
    )
    numpy.testing.assert_allclose(
        selected.follow_through,
        [
            contact_surface_x + PUSHER_RADIUS - MAXIMUM_PUSH_DISTANCE,
            0.0,
            PUSHING_HEIGHT,
        ],
        atol=1e-9,
    )


def test_the_push_follows_the_body_when_it_is_turned():
    """
    The contacts are given in the body's own frame, so a turned body is pushed on the
    face that has turned with it.
    """
    contacts = square_contacts()
    selector = build_selector(contacts)

    selected = selector.select(
        root_T_body=planar_pose(y=0.5, yaw=math.pi / 2),
        root_T_target=planar_pose(),
        tolerance=TOLERANCE,
    )

    # Turned a quarter turn, the body's +x face points along the root's +y, which is the
    # direction the body has to be pushed away from.
    assert selected.contact is contacts[0]
    numpy.testing.assert_allclose(
        selected.contact_point,
        [0.0, 0.5 + HALF_WIDTH + PUSHER_RADIUS, PUSHING_HEIGHT],
        atol=1e-9,
    )


def test_a_push_only_travels_as_far_as_the_error_it_corrects():
    """
    A push as long as the body is far from its target overshoots, and an overshoot has
    to be undone from the other side, so a small error gets a short push.
    """
    selector = build_selector(square_contacts())
    distance_to_go = MAXIMUM_PUSH_DISTANCE / 2

    selected = selector.select(
        root_T_body=planar_pose(x=distance_to_go),
        root_T_target=planar_pose(),
        tolerance=TOLERANCE,
    )

    assert push_length(selected) == pytest.approx(distance_to_go * PUSH_GAIN)


def test_a_turning_push_grows_with_how_far_the_body_has_to_turn():
    """
    Turning a body twice as far takes twice the push, the same way moving it twice as
    far does.
    """
    selector = build_selector(offset_contacts())

    small_turn = selector.select(
        root_T_body=planar_pose(yaw=0.6),
        root_T_target=planar_pose(),
        tolerance=TOLERANCE,
    )
    large_turn = selector.select(
        root_T_body=planar_pose(yaw=1.2),
        root_T_target=planar_pose(),
        tolerance=TOLERANCE,
    )

    assert push_length(large_turn) == pytest.approx(2 * push_length(small_turn))


def test_a_push_is_never_too_short_to_move_the_body():
    """
    A push shorter than the slack in the contact is taken up by friction without the
    body moving at all, which would leave the same error to correct again next time.
    """
    selector = build_selector(square_contacts())

    selected = selector.select(
        root_T_body=planar_pose(x=MINIMUM_PUSH_DISTANCE / 100),
        root_T_target=planar_pose(),
        tolerance=TOLERANCE,
    )

    assert push_length(selected) == pytest.approx(MINIMUM_PUSH_DISTANCE)


# %% the statechart the goal builds


@pytest.fixture
def pushing_world() -> World:
    """
    A world with a free-moving block, a fixed target marker, and a pusher on three slide
    joints.
    """
    world = World.create_with_root_body("world")
    block = Body(name=PrefixedName("block"))
    target = Body(name=PrefixedName("target"))
    pusher = Body(name=PrefixedName("pusher"))
    links = [Body(name=PrefixedName(f"link_{axis}")) for axis in "xy"]

    with world.modify_world():
        world.add_connection(
            Connection6DoF.create_with_dofs(world=world, parent=world.root, child=block)
        )
        world.add_connection(FixedConnection(parent=world.root, child=target))
        parent = world.root
        axes = [Vector3.X, Vector3.Y, Vector3.Z]
        children = links + [pusher]
        for axis_name, axis_factory, child in zip("xyz", axes, children):
            degree_of_freedom = DegreeOfFreedom(
                name=PrefixedName(f"slide_{axis_name}"),
                limits=DegreeOfFreedomLimits(
                    lower=DerivativeMap(position=-1.0, velocity=-1.0),
                    upper=DerivativeMap(position=1.0, velocity=1.0),
                ),
            )
            world.add_degree_of_freedom(degree_of_freedom)
            connection = PrismaticConnection(
                name=degree_of_freedom.name,
                parent=parent,
                child=child,
                axis=axis_factory(reference_frame=parent),
                raw_dof=degree_of_freedom,
            )
            world.add_connection(connection)
            parent = child
    return world


def push_to_pose(world: World) -> PushToPose:
    """
    :param world: The world holding the block, the target and the pusher.
    :return: A goal pushing the world's block onto its target.
    """
    return PushToPose(
        pushed_body=world.get_kinematic_structure_entity_by_name("block"),
        target_body=world.get_kinematic_structure_entity_by_name("target"),
        pusher=world.get_kinematic_structure_entity_by_name("pusher"),
        selector=build_selector(square_contacts()),
        travel_height=0.06,
        tolerance=TOLERANCE,
    )


def compiled_statechart(world: World, goal: PushToPose) -> MotionStatechart:
    """
    Compile ``goal`` into a statechart, so its children exist and are built.

    :param world: The world the goal acts in.
    :param goal: The goal to compile.
    :return: The statechart holding the compiled goal.
    """
    motion_statechart = MotionStatechart()
    motion_statechart.add_node(goal)
    Executor(context=MotionStatechartContext(world=world)).compile(
        motion_statechart=motion_statechart
    )
    return motion_statechart


def test_one_push_lifts_travels_descends_and_pushes_in_that_order(pushing_world):
    """
    The pusher can only reach another face by going over the body, so a push is four
    steps in a fixed order rather than a single straight run.
    """
    goal = push_to_pose(pushing_world)
    compiled_statechart(pushing_world, goal)

    [push_once] = [node for node in goal.nodes if isinstance(node, PushOnce)]
    [sequence] = [node for node in push_once.nodes if isinstance(node, Sequence)]

    assert [type(node) for node in sequence.nodes] == [
        CartesianPosition,
        CartesianPosition,
        CartesianPosition,
        CartesianPositionStraight,
    ]
    assert [node.name for node in sequence.nodes] == [
        "lift",
        "travel",
        "descend",
        "push",
    ]


def test_every_step_of_a_push_moves_the_pusher(pushing_world):
    """
    Only the pusher is commanded; the block moves because it is in the way, never
    because a task asked it to.
    """
    goal = push_to_pose(pushing_world)
    compiled_statechart(pushing_world, goal)

    [push_once] = [node for node in goal.nodes if isinstance(node, PushOnce)]
    [sequence] = [node for node in push_once.nodes if isinstance(node, Sequence)]

    pusher = pushing_world.get_kinematic_structure_entity_by_name("pusher")
    assert {node.tip_link for node in sequence.nodes} == {pusher}
    assert {node.root_link for node in sequence.nodes} == {pushing_world.root}


def test_a_finished_or_stalled_push_starts_another_one(pushing_world):
    """
    One push rarely lands the body on its target, so the goal resets its push and picks
    a new contact against the pose the body has by then.
    """
    goal = push_to_pose(pushing_world)
    compiled_statechart(pushing_world, goal)

    [push_once] = [node for node in goal.nodes if isinstance(node, PushOnce)]
    [stalled] = [node for node in goal.nodes if isinstance(node, ProgressStalled)]

    reset_dependencies = push_once._reset_condition.node_dependencies
    assert set(reset_dependencies) == {push_once, stalled}


def test_pushing_stops_once_the_body_is_on_its_target(pushing_world):
    """
    The goal watches the body rather than the pusher, since the pusher reaching a point
    says nothing about where the body ended up.
    """
    goal = push_to_pose(pushing_world)
    compiled_statechart(pushing_world, goal)

    [on_target] = [node for node in goal.nodes if isinstance(node, PoseReached)]
    [push_once] = [node for node in goal.nodes if isinstance(node, PushOnce)]

    assert on_target.tip_link is pushing_world.get_kinematic_structure_entity_by_name(
        "block"
    )
    assert on_target.goal_pose.reference_frame is (
        pushing_world.get_kinematic_structure_entity_by_name("target")
    )
    assert push_once._start_condition.node_dependencies == [on_target]


def test_arrival_is_judged_by_the_tolerance_the_pushes_are_chosen_against(
    pushing_world,
):
    """
    A goal that stopped correcting the body before it was close enough to count as
    arrived could never finish, so one tolerance settles both.
    """
    goal = push_to_pose(pushing_world)
    compiled_statechart(pushing_world, goal)

    [on_target] = [node for node in goal.nodes if isinstance(node, PoseReached)]
    [push_once] = [node for node in goal.nodes if isinstance(node, PushOnce)]

    assert on_target.position_threshold == goal.tolerance.position
    assert on_target.orientation_threshold == goal.tolerance.orientation
    assert push_once.tolerance is goal.tolerance
