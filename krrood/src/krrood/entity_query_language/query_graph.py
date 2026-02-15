from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

from typing_extensions import ClassVar, Optional, List, Any, TYPE_CHECKING, Dict

from .conclusion import Conclusion
from .conclusion_selector import ConclusionSelector
from .symbolic import (
    ResultQuantifier,
    Product,
    QueryObjectDescriptor,
    SymbolicExpression,
    Filter,
    Where,
    OrderedBy,
    Aggregator,
    GroupedBy,
    Variable,
    Literal,
    Concatenate,
    DomainMapping,
    Comparator,
    LogicalOperator,
)

try:
    from rustworkx_utils import GraphVisualizer, RWXNode as RXUtilsNode
except ImportError:
    GraphVisualizer = None

import rustworkx as rx


@dataclass
class QueryGraph:
    """
    Represents a query graph for visualizing and introspecting query structures.
    """

    query: SymbolicExpression
    """
    An expression representing the query.
    """
    graph: rx.PyDAG = field(init=False, default_factory=rx.PyDAG)
    """
    The graph representation of the query, used for visualization and introspection.
    """
    expression_node_map: Dict[SymbolicExpression, RWXNode] = field(
        init=False, default_factory=dict
    )
    """
    A mapping from symbolic expressions to their corresponding nodes in the graph.
    """

    def __post_init__(self):
        self.construct_graph()

    def visualize(
        self,
        figsize=(35, 30),
        node_size=7000,
        font_size=25,
        spacing_x: float = 4,
        spacing_y: float = 4,
        curve_scale: float = 0.5,
        layout: str = "tidy",
        edge_style: str = "orthogonal",
        label_max_chars_per_line: Optional[int] = 13,
    ):
        """
        Visualizes the graph using the specified layout and style options.

        Provides a graphical visualization of the graph with customizable options for
        size, layout, spacing, and labeling. Requires the rustworkx_utils library for
        execution.

        :param figsize (tuple of float): Size of the figure in inches (width, height). Default is (35, 30).
        :param node_size (int): Size of the nodes in the visualization. Default is 7000.
        :param font_size (int): Size of the font used for node labels. Default is 25.
        :param spacing_x (float): Horizontal spacing between nodes. Default is 4.
        :param spacing_y (float): Vertical spacing between nodes. Default is 4.
        :param curve_scale (float): Scaling factor for edge curvature. Default is 0.5.
        :param layout (str): Graph layout style (e.g., "tidy"). Default is "tidy".
        :param edge_style (str): Style of the edges (e.g., "orthogonal"). Default is "orthogonal".
        :param label_max_chars_per_line (Optional[int]): Maximum characters per line for node labels. Default is 13.

        :returns: The rendered visualization object.

        :raises: `ModuleNotFoundError` If rustworkx_utils is not installed.
        """
        if not GraphVisualizer:
            raise ModuleNotFoundError(
                "rustworkx_utils is not installed. Please install it with `pip install rustworkx_utils`"
            )
        visualizer = GraphVisualizer(
            node=self.expression_node_map[self.query._root_],
            figsize=figsize,
            node_size=node_size,
            font_size=font_size,
            spacing_x=spacing_x,
            spacing_y=spacing_y,
            curve_scale=curve_scale,
            layout=layout,
            edge_style=edge_style,
            label_max_chars_per_line=label_max_chars_per_line,
        )
        return visualizer.render()

    def construct_graph(
        self,
        expression: Optional[SymbolicExpression] = None,
    ) -> RWXNode:
        """
        Construct the graph representation of the query, used for visualization and introspection.
        """
        expression = expression if expression is not None else self.query._root_

        if expression in self.expression_node_map:
            return self.expression_node_map[expression]

        node = RWXNode(
            expression._name_,
            self.graph,
            color=ColorLegend.from_expression(expression),
            data=expression,
        )
        self.expression_node_map[expression] = node

        if isinstance(expression, ResultQuantifier):
            node.wrap_subtree = True

        self._add_children_to_graph(node)

        return node

    def _add_children_to_graph(
        self,
        parent_node: RWXNode,
    ):
        """
        Adds child nodes to the graph recursively.

        :param parent_node: The parent node of the children to add.
        """
        parent_expression = parent_node.data
        for child in parent_expression._children_:
            child_node = self.construct_graph(child)
            if isinstance(parent_expression, Product) and isinstance(
                parent_expression._parents_[0], QueryObjectDescriptor
            ):
                if child._binding_id_ in [
                    v._binding_id_
                    for v in parent_expression._parents_[0]._selected_variables_
                ]:
                    child_node.enclosed = True
            child_node.parent = parent_node


@dataclass
class ColorLegend:
    name: str = field(default="Other")
    """
    The name of the color legend entry.
    """
    color: str = field(default="white")
    """
    The color associated with the color legend entry.
    """

    @classmethod
    def from_expression(cls, expression: SymbolicExpression) -> ColorLegend:
        name = expression.__class__.__name__
        color = "white"
        match expression:
            case Product() | Filter() | OrderedBy() | GroupedBy():
                color = "#17becf"
            case Aggregator():
                color = "#F54927"
            case ResultQuantifier():
                color = "#9467bd"
            case QueryObjectDescriptor():
                color = "#d62728"
            case Literal():
                color = "#949292"
            case Variable():
                color = "cornflowerblue"
            case Concatenate():
                color = "#949292"
            case DomainMapping():
                name = "DomainMapping"
                color = "#8FC7B8"
            case Comparator():
                color = "#ff7f0e"
            case ConclusionSelector():
                color = "#eded18"
            case LogicalOperator():
                color = "#2ca02c"
            case Conclusion():
                color = "#8cf2ff"
        return cls(name=name, color=color)


@dataclass
class RWXNode(RXUtilsNode):
    """
    A node in the query graph. Overrides the default enclosed name to "Selected Variable".
    """

    enclosed_name: ClassVar[str] = "Selected Variable"
