import krrood.entity_query_language.entity_result_processors as eql
from krrood.entity_query_language.entity import (
    variable,
    variable_from,
    entity,
)
from ..dataset.semantic_world_like_classes import Cabinet


def test_max_per(handles_and_containers_world):
    world = handles_and_containers_world
    cabinet = variable(Cabinet, domain=world.views)
    drawer = variable_from(cabinet.drawers)

    # We want to find the drawer with the "largest" handle name (alphabetically) per cabinet.
    max_drawer = eql.max(drawer, key=lambda d: d.handle.name).per(cabinet)
    results = list(max_drawer.evaluate())

    # expected: for each cabinet, one result which is the drawer with max handle name
    expected_cabinets = [c for c in world.views if isinstance(c, Cabinet)]
    assert len(results) == len(expected_cabinets)

    for res in results:
        # res should have cabinet and the drawer in bindings
        c = res[cabinet]
        d = res[drawer]
        assert d in c.drawers
        assert d.handle.name == max(cd.handle.name for cd in c.drawers)


def test_multiple_per_variables(handles_and_containers_world):
    world = handles_and_containers_world
    cabinet = variable(Cabinet, domain=world.views)
    drawer = variable_from(cabinet.drawers)

    # Group by both cabinet and drawer (silly, but tests multiple variables)
    count = eql.count(drawer).per(cabinet, drawer)
    results = list(count.evaluate())

    # Each result should have count=1 because each (cabinet, drawer) pair is unique here
    for res in results:
        assert res[count] == 1
        assert res[cabinet] is not None
        assert res[drawer] is not None


def test_sum_per(handles_and_containers_world):
    world = handles_and_containers_world
    cabinet = variable(Cabinet, domain=world.views)
    drawer = variable_from(cabinet.drawers)
    # Give drawers a numeric property to sum. They don't have one, but we can use a key func.
    # Let's sum the length of handle names per cabinet.

    total_characters = eql.sum(drawer, key=lambda d: len(d.handle.name)).per(cabinet)
    results = list(total_characters.evaluate())

    expected_cabinets = [c for c in world.views if isinstance(c, Cabinet)]
    assert len(results) == len(expected_cabinets)

    for res in results:
        c = res[cabinet]
        s = res[total_characters]
        assert s == sum(len(d.handle.name) for d in c.drawers)


def test_count_per(handles_and_containers_world):
    world = handles_and_containers_world
    cabinet = variable(Cabinet, domain=world.views)
    cabinet_drawers = variable_from(cabinet.drawers)
    query = eql.count(cabinet_drawers).per(cabinet)
    result = list(query.evaluate())
    expected = [len(c.drawers) for c in world.views if isinstance(c, Cabinet)]
    assert [r[query] for r in result] == expected

    # without per should be all drawers of all cabinets
    query_all = eql.count(cabinet_drawers)
    results = list(query_all.evaluate())
    assert len(results) == 1
    result_all = results[0]
    expected_all = sum(len(c.drawers) for c in world.views if isinstance(c, Cabinet))
    assert result_all == expected_all


def test_max_count_per(handles_and_containers_world):
    world = handles_and_containers_world
    cabinet = variable(Cabinet, domain=world.views)
    cabinet_drawers = variable_from(cabinet.drawers)
    query = eql.max(eql.count(cabinet_drawers).per(cabinet))
    result = list(query.evaluate())
    assert len(result) == 1
    result_max = result[0][query]
    expected = 0
    for c in world.views:
        if isinstance(c, Cabinet) and len(c.drawers) > expected:
            expected = len(c.drawers)
    assert result_max == expected


def test_max_min_no_variable():
    values = [2, 1, 3, 5, 4]
    value = variable(int, domain=values)

    max_query = eql.max(entity(value))
    assert list(max_query.evaluate())[0] == max(values)

    min_query = eql.min(entity(value))
    assert list(min_query.evaluate())[0] == min(values)


def test_max_min_without_entity():
    values = [2, 1, 3, 5, 4]
    value = variable(int, domain=values)

    max_query = eql.max(value)
    assert list(max_query.evaluate())[0] == max(values)

    min_query = eql.min(value)
    assert list(min_query.evaluate())[0] == min(values)


def test_max_min_with_empty_list():
    empty_list = []
    value = variable(int, domain=empty_list)

    max_query = eql.max(entity(value))
    assert list(max_query.evaluate())[0] is None

    min_query = eql.min(entity(value))
    assert list(min_query.evaluate())[0] is None
