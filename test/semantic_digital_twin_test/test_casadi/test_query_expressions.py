import pytest

from krrood.entity_query_language.entity import (
    let,
    entity,
    contains,
    flatten,
)
from krrood.entity_query_language.failures import NoSolutionFound
from krrood.entity_query_language.match import match
from krrood.entity_query_language.predicate import symbolic_function
from krrood.entity_query_language.quantify_entity import the, a
from semantic_digital_twin.spatial_types import Expression, FloatVariable

from semantic_digital_twin.testing import world_setup
from semantic_digital_twin.world_description.connections import RevoluteConnection
from semantic_digital_twin.world_description.degree_of_freedom import PositionVariable


@symbolic_function
def eql_hash(val):
    return hash(val)


def test_querying_equations(world_setup):
    results = list(a(match(PositionVariable)()).evaluate())
    expr = results[0] + results[1]
    e = let(Expression, domain=None)
    free_vars = eql_hash(flatten(e.free_variables()))
    free_vars2 = eql_hash(flatten(e.free_variables()))
    query = the(
        entity(
            e,
            e.is_scalar(),
            free_vars == eql_hash(results[0]),
            free_vars2 == eql_hash(results[1]),
        )
    )
    found_expr = query.evaluate()
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
