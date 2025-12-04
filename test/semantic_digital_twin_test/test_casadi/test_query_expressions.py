import pytest

from krrood.entity_query_language.entity import (
    let,
    entity,
    contains,
)
from krrood.entity_query_language.failures import NoSolutionFound
from krrood.entity_query_language.match import match
from krrood.entity_query_language.quantify_entity import the, a
from semantic_digital_twin.spatial_types import Expression, FloatVariable

from semantic_digital_twin.testing import world_setup
from semantic_digital_twin.world_description.connections import RevoluteConnection
from semantic_digital_twin.world_description.degree_of_freedom import PositionVariable


def test_querying_equations(world_setup):
    results = list(a(match(PositionVariable)()).evaluate())
    expr = results[0] + results[1]
    found_expr = a(
        entity(
            e := let(Expression, domain=None),
            e.is_scalar(),
            contains(e.free_variables(), results[0]),
            contains(e.free_variables(), results[1]),
        )
    ).evaluate()
    result = list(found_expr)
    assert found_expr is expr


def test_no_solution(world_setup):
    with pytest.raises(NoSolutionFound):
        found_expr = the(
            entity(
                c := let(RevoluteConnection, domain=None),
                c.multiplier + 1 > 5,
            )
        ).evaluate()


def test_the_solution(world_setup):
    world, l1, l2, bf, r1, r2 = world_setup
    expected = world.get_connections_by_type(RevoluteConnection)[0]
    found_expr = the(
        entity(
            c := let(RevoluteConnection, domain=None),
            c.dof.variables.position + 1 == 2,
        )
    ).evaluate()
    assert found_expr is expected
