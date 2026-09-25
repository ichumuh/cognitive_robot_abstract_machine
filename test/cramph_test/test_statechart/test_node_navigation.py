from __future__ import annotations

import pytest
from typing_extensions import List

from cramph.composites import Parallel, Sequence
from cramph.executor import StatechartExecutor
from cramph.node import StatechartNode
from cramph.nodes_for_testing import ConstTrueNode
from cramph.statechart import Statechart

# %% the tree the navigation is read from


@pytest.fixture()
def nested_statechart(statechart_executor: StatechartExecutor) -> Statechart:
    """
    :return: A statechart holding one tree, whose middle child is itself a composite::

        root
        |-- first
        |-- middle
        |   |-- inner_first
        |   +-- inner_last
        +-- last

    Its nodes are reached by name through :func:`node_named`.
    """
    statechart = Statechart(context=statechart_executor.context)
    middle = Parallel(
        name="middle",
        nodes=[ConstTrueNode(name="inner_first"), ConstTrueNode(name="inner_last")],
    )
    statechart.add_node(
        Parallel(
            name="root",
            nodes=[ConstTrueNode(name="first"), middle, ConstTrueNode(name="last")],
        )
    )
    return statechart


def node_named(statechart: Statechart, name: str) -> StatechartNode:
    """
    :param statechart: The statechart to search.
    :param name: The name of the node to find.
    :return: The one node of `statechart` carrying that name.
    """
    matches = [node for node in statechart.nodes if node.name == name]
    assert len(matches) == 1, f"{name} names {len(matches)} nodes"
    return matches[0]


def names_of(nodes: List[StatechartNode]) -> List[str]:
    """
    :param nodes: The nodes to name.
    :return: Their names, in the order they were given.
    """
    return [node.name for node in nodes]


# %% children


def test_a_node_that_runs_nothing_has_no_children(nested_statechart: Statechart):
    assert node_named(nested_statechart, "first").children == []


def test_a_composite_answers_the_nodes_it_runs_in_order(
    nested_statechart: Statechart,
):
    root = node_named(nested_statechart, "root")

    assert root.children == root.nodes


def test_children_are_exactly_the_nodes_whose_parent_this_is(
    nested_statechart: Statechart,
):
    for node in nested_statechart.nodes:
        assert node.children == [
            candidate
            for candidate in nested_statechart.nodes
            if candidate.parent_node is node
        ]


# %% descendants


def test_descendants_follow_each_child_with_its_own_subtree(
    nested_statechart: Statechart,
):
    root = node_named(nested_statechart, "root")

    assert names_of(root.descendants) == [
        "first",
        "middle",
        "inner_first",
        "inner_last",
        "last",
    ]


def test_a_node_that_runs_nothing_has_no_descendants(nested_statechart: Statechart):
    assert node_named(nested_statechart, "last").descendants == []


# %% path


def test_the_path_leads_from_the_parent_up_to_the_root(
    nested_statechart: Statechart,
):
    inner_first = node_named(nested_statechart, "inner_first")

    assert inner_first.path == [
        node_named(nested_statechart, "middle"),
        node_named(nested_statechart, "root"),
    ]


def test_a_top_level_node_has_an_empty_path(nested_statechart: Statechart):
    assert node_named(nested_statechart, "root").path == []


def test_the_depth_of_a_node_is_the_length_of_its_path(
    nested_statechart: Statechart,
):
    for node in nested_statechart.nodes:
        assert node.depth == len(node.path)


# %% is_leaf


def test_a_node_that_runs_nothing_is_a_leaf(nested_statechart: Statechart):
    assert node_named(nested_statechart, "inner_last").is_leaf


def test_a_composite_is_not_a_leaf(nested_statechart: Statechart):
    assert not node_named(nested_statechart, "middle").is_leaf


# %% siblings


def test_the_siblings_of_a_child_are_the_other_children_in_order(
    nested_statechart: Statechart,
):
    assert names_of(node_named(nested_statechart, "middle").siblings) == [
        "first",
        "last",
    ]


def test_the_siblings_of_a_top_level_node_are_the_other_top_level_nodes(
    statechart_executor: StatechartExecutor,
):
    statechart = Statechart(context=statechart_executor.context)
    statechart.add_nodes([first := ConstTrueNode(), other := ConstTrueNode()])

    assert first.siblings == [other]


def test_the_only_top_level_node_has_no_siblings(
    statechart_executor: StatechartExecutor,
):
    statechart = Statechart(context=statechart_executor.context)
    statechart.add_node(only := ConstTrueNode())

    assert only.siblings == []


def test_siblings_of_the_same_name_are_told_apart_by_identity(
    statechart_executor: StatechartExecutor,
):
    statechart = Statechart(context=statechart_executor.context)
    statechart.add_node(
        Parallel(
            name="root",
            nodes=[
                first := ConstTrueNode(name="same"),
                second := ConstTrueNode(name="same"),
            ],
        )
    )

    assert first.siblings == [second]
    assert second.siblings == [first]


# %% neighbours


def test_the_siblings_split_into_those_left_and_right_of_this_node(
    nested_statechart: Statechart,
):
    middle = node_named(nested_statechart, "middle")

    assert names_of(middle.left_siblings) == ["first"]
    assert names_of(middle.right_siblings) == ["last"]


def test_the_neighbours_are_the_closest_sibling_on_either_side(
    nested_statechart: Statechart,
):
    middle = node_named(nested_statechart, "middle")

    assert middle.left_neighbour is node_named(nested_statechart, "first")
    assert middle.right_neighbour is node_named(nested_statechart, "last")


def test_the_leftmost_node_has_no_left_neighbour(nested_statechart: Statechart):
    first = node_named(nested_statechart, "first")

    assert first.left_siblings == []
    assert first.left_neighbour is None


def test_the_rightmost_node_has_no_right_neighbour(nested_statechart: Statechart):
    last = node_named(nested_statechart, "last")

    assert last.right_siblings == []
    assert last.right_neighbour is None


# %% a goal that wraps its children


def test_a_sequence_answers_the_children_it_wrapped_its_steps_in(
    statechart_executor: StatechartExecutor,
):
    statechart = Statechart(context=statechart_executor.context)
    sequence = Sequence(nodes=[ConstTrueNode(name="step"), ConstTrueNode(name="next")])
    statechart.add_node(sequence)

    assert sequence.children == sequence.nodes
    assert sequence.children[0].descendants == [node_named(statechart, "step")]
