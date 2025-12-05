import pytest
import krrood.entity_query_language.entity as eql
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
from semantic_digital_twin.world_description.connections import (
    RevoluteConnection,
    ActiveConnection1DOF,
)
from semantic_digital_twin.world_description.degree_of_freedom import PositionVariable
import semantic_digital_twin.spatial_types.spatial_types as cas
from semantic_digital_twin.world_description.world_entity import Connection


def test_querying_equations(world_setup):
    results = list(a(match(PositionVariable)()).evaluate())
    expr = results[0] + results[1]
    expr2 = results[0] + results[3]
    e = let(Expression, domain=None)
    free_variables = e.free_variables()
    query = the(
        entity(
            e,
            e.is_scalar(),
            contains(free_variables, results[0]),
            contains(free_variables, results[1]),
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
            c.multiplier + 1 == 2,
        )
    ).evaluate()
    assert found_expr is expected


def test_max(world_setup):
    world, l1, l2, bf, r1, r2 = world_setup
    expected = world.get_connections_by_type(RevoluteConnection)[0]
    max_val = eql.max(let(ActiveConnection1DOF, domain=None).multiplier).evaluate()
    found_expr = a(
        entity(
            c := let(ActiveConnection1DOF, domain=None),
            c.multiplier == max_val,
        )
    ).evaluate()
    found_exprs = list(found_expr)
    assert len(found_exprs) == 2
