"""
Tests for the sizes MuJoCo is given for the shapes of a world.

MuJoCo describes a shape by its half-extents, while a
:class:`~semantic_digital_twin.world_description.geometry.Shape` describes itself by its
full ones, so every converter has a halving to get right and a shape built at twice its
size collides with things its own world says it is nowhere near.
"""

from semantic_digital_twin.adapters.multi_sim import (
    MujocoBoxConverter,
    MujocoCylinderConverter,
    MujocoSphereConverter,
)
from semantic_digital_twin.world_description.geometry import (
    Box,
    Cylinder,
    Scale,
    Sphere,
)

# %% shape sizes


def test_a_box_is_given_its_half_extents():
    box = Box(scale=Scale(0.2, 0.4, 0.6))

    properties = MujocoBoxConverter.convert(box)

    assert properties["size"] == [
        box.scale.x / 2,
        box.scale.y / 2,
        box.scale.z / 2,
    ]


def test_a_cylinder_is_given_its_radius_and_half_its_length():
    cylinder = Cylinder(width=0.03, height=0.25)

    properties = MujocoCylinderConverter.convert(cylinder)

    assert properties["size"][0] == cylinder.radius
    assert properties["size"][1] == cylinder.height / 2


def test_a_sphere_is_given_its_radius():
    sphere = Sphere(radius=0.02)

    properties = MujocoSphereConverter.convert(sphere)

    assert properties["size"][0] == sphere.radius
