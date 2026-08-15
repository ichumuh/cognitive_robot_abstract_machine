"""
Where the T block can be pushed, derived from the two boxes it is built from.

Everything here follows from the dimensions in :mod:`experiments.push_t.scene`, so
reshaping the T moves its contacts with it.
"""

from __future__ import annotations

import math

from typing_extensions import List, Sequence

from experiments.push_t.scene import (
    CROSSBAR_SCALE,
    STEM_OFFSET,
    STEM_SCALE,
)
from giskardpy.motion_statechart.goals.pushing import PushContact
from semantic_digital_twin.spatial_types.spatial_types import Point3, Vector3
from semantic_digital_twin.world_description.geometry import Scale

# %% the T's outline

CROSSBAR_HALF_WIDTH = CROSSBAR_SCALE.x / 2
"""
How far the crossbar reaches along either side of the block's frame.
"""

CROSSBAR_HALF_DEPTH = CROSSBAR_SCALE.y / 2
"""
Half the crossbar's depth, so the offset of its two long faces.
"""

STEM_HALF_WIDTH = STEM_SCALE.x / 2
"""
Half the stem's width, so the offset of its two long faces.
"""

STEM_HALF_LENGTH = STEM_SCALE.y / 2
"""
Half the stem's length.
"""

STEM_CENTRE = -STEM_OFFSET
"""
Where the stem's middle sits along the block frame's y axis.
"""

STEM_END = STEM_CENTRE - STEM_HALF_LENGTH
"""
Where the stem's free end sits along the block frame's y axis.
"""

EXPOSED_UNDERSIDE_CENTRE = (STEM_HALF_WIDTH + CROSSBAR_HALF_WIDTH) / 2
"""
Middle of one of the two stretches of crossbar underside the stem leaves exposed.

The stem meets the crossbar in the middle of its underside, so what is left to push on
is a pair of ledges either side of it.
"""

BLOCK_RADIUS = max(
    math.hypot(CROSSBAR_HALF_WIDTH, CROSSBAR_HALF_DEPTH),
    math.hypot(STEM_HALF_WIDTH, abs(STEM_END)),
)
"""
Radius of the smallest circle about the block's frame holding the whole T, in metres.

Whichever way the T is turned it stays inside this circle, so it is what says how close
to something else the T may be placed.
"""

# %% the T's centroid

CROSSBAR_AREA = CROSSBAR_SCALE.x * CROSSBAR_SCALE.y
"""
Face area of the crossbar, used to weigh it against the stem.
"""

STEM_AREA = STEM_SCALE.x * STEM_SCALE.y
"""
Face area of the stem, used to weigh it against the crossbar.
"""

CENTROID_OFFSET = STEM_AREA * STEM_CENTRE / (CROSSBAR_AREA + STEM_AREA)
"""
Where the T's centroid sits along the block frame's y axis.

The block's frame sits on the crossbar rather than on the centroid, and a push's turning
effect is about the centroid, so the two must not be confused.
"""

BLOCK_CENTROID = Point3(x=0.0, y=CENTROID_OFFSET, z=0.0)
"""
The T's centroid, in the block's own frame.
"""


def box_second_moment(scale: Scale, centre_offset: float) -> float:
    """
    How far one of the T's two boxes has its area spread from the block's centroid.

    :param scale: Extents of the box.
    :param centre_offset: How far the box's own centre sits from :data:`BLOCK_CENTROID`.
    :return: The box's second moment of area about the centroid, in metres to the
        fourth.
    """
    own_spread = (scale.x**2 + scale.y**2) / 12
    return scale.x * scale.y * (own_spread + centre_offset**2)


BLOCK_GYRATION_RADIUS = math.sqrt(
    (
        box_second_moment(CROSSBAR_SCALE, -CENTROID_OFFSET)
        + box_second_moment(STEM_SCALE, STEM_CENTRE - CENTROID_OFFSET)
    )
    / (CROSSBAR_AREA + STEM_AREA)
)
"""
The radius of gyration of the T's footprint about its centroid, in metres.

It is the distance at which the whole footprint could be concentrated and still resist
turning as much as it does, so it sets how much of a push off the centre goes into
spinning the T rather than sliding it.
"""


# %% sampling the outline


def face_contacts(
    outward: Vector3,
    face_centre: Point3,
    along: Vector3,
    offsets: Sequence[float],
) -> List[PushContact]:
    """
    Sample one flat stretch of the block's outline at several places.

    :param outward: Unit direction the face points in, in the block's frame.
    :param face_centre: A point on the face, in the block's frame.
    :param along: Unit direction running across the face, in the block's frame.
    :param offsets: How far along ``along`` from ``face_centre`` to sample.
    :return: One contact per offset, each pushing straight into the face.
    """
    return [
        PushContact(
            point=Point3(
                x=face_centre.x + along.x * offset,
                y=face_centre.y + along.y * offset,
                z=0.0,
            ),
            direction=Vector3(x=-outward.x, y=-outward.y, z=0.0),
        )
        for offset in offsets
    ]


def build_push_contacts() -> List[PushContact]:
    """
    Every place the pusher can meet the T, in the block's own frame.

    The long faces are sampled away from their middles as well as at them, so that a
    push that has to turn the block has a lever arm to work with.

    :return: The contacts, running from the crossbar's top face round to the stem's end.
    """
    across = Vector3(x=1.0, y=0.0, z=0.0)
    lengthwise = Vector3(x=0.0, y=1.0, z=0.0)
    quarter_width = CROSSBAR_HALF_WIDTH / 2
    stem_quarter_length = STEM_HALF_LENGTH / 2
    return [
        # The crossbar's top, its full width free to push on.
        *face_contacts(
            outward=lengthwise,
            face_centre=Point3(x=0.0, y=CROSSBAR_HALF_DEPTH, z=0.0),
            along=across,
            offsets=(-quarter_width, 0.0, quarter_width),
        ),
        # The two ledges the stem leaves either side of the crossbar's underside.
        *face_contacts(
            outward=Vector3(x=0.0, y=-1.0, z=0.0),
            face_centre=Point3(x=0.0, y=-CROSSBAR_HALF_DEPTH, z=0.0),
            along=across,
            offsets=(-EXPOSED_UNDERSIDE_CENTRE, EXPOSED_UNDERSIDE_CENTRE),
        ),
        # The crossbar's two ends, where the block's longest lever arms are.
        *face_contacts(
            outward=across,
            face_centre=Point3(x=CROSSBAR_HALF_WIDTH, y=0.0, z=0.0),
            along=lengthwise,
            offsets=(0.0,),
        ),
        *face_contacts(
            outward=Vector3(x=-1.0, y=0.0, z=0.0),
            face_centre=Point3(x=-CROSSBAR_HALF_WIDTH, y=0.0, z=0.0),
            along=lengthwise,
            offsets=(0.0,),
        ),
        # The stem's two long sides.
        *face_contacts(
            outward=across,
            face_centre=Point3(x=STEM_HALF_WIDTH, y=STEM_CENTRE, z=0.0),
            along=lengthwise,
            offsets=(-stem_quarter_length, stem_quarter_length),
        ),
        *face_contacts(
            outward=Vector3(x=-1.0, y=0.0, z=0.0),
            face_centre=Point3(x=-STEM_HALF_WIDTH, y=STEM_CENTRE, z=0.0),
            along=lengthwise,
            offsets=(-stem_quarter_length, stem_quarter_length),
        ),
        # The stem's free end.
        *face_contacts(
            outward=Vector3(x=0.0, y=-1.0, z=0.0),
            face_centre=Point3(x=0.0, y=STEM_END, z=0.0),
            along=across,
            offsets=(0.0,),
        ),
    ]
