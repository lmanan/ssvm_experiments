import networkx as nx
from motile.costs import Costs, Weight
from motile.solver import Solver
import numpy as np
from motile.variables import EdgeSelected
from typing import cast


class HyperEdgeDistance(Costs):
    def __init__(
        self,
        position_attribute: str | tuple[str, ...],
        weight: float = 1.0,
        constant: float = 0.0,
    ) -> None:
        self.position_attribute = position_attribute
        self.weight = Weight(weight)
        self.constant = Weight(constant)

    def apply(self, solver: Solver) -> None:
        edge_variables = solver.get_variables(EdgeSelected)
        for key, index in edge_variables.items():
            if type(key[1]) is tuple:  # must be a hyper edge ...
                solver.add_variable_cost(index, 0.0, self.weight)
                solver.add_variable_cost(index, 0.0, self.constant)
            else:  # normal edge
                u, v = cast("tuple[int, int]", key)
                pos_u = self.__get_node_position(solver.graph, u)
                pos_v = self.__get_node_position(solver.graph, v)
                feature = np.linalg.norm(pos_u - pos_v)
                solver.add_variable_cost(index, feature, self.weight)
                solver.add_variable_cost(index, 1.0, self.constant)

    def __get_node_position(self, graph: nx.DiGraph, node: int) -> np.ndarray:
        if isinstance(self.position_attribute, tuple):
            return np.array([graph.nodes[node][p] for p in self.position_attribute])
        else:
            return np.array(graph.nodes[node][self.position_attribute])


class HyperSplit(Costs):
    def __init__(self, weight, position_attribute, constant):
        self.weight = Weight(weight)
        self.constant = Weight(constant)
        self.position_attribute = position_attribute

    def apply(self, solver):
        edge_variables = solver.get_variables(EdgeSelected)
        for key, index in edge_variables.items():
            if type(key[1]) is tuple:
                (start,) = key[0]
                end1, end2 = key[1]
                pos_start = self.__get_node_position(solver.graph, start)
                pos_end1 = self.__get_node_position(solver.graph, end1)
                pos_end2 = self.__get_node_position(solver.graph, end2)
                feature = np.linalg.norm(pos_start - 0.5 * (pos_end1 + pos_end2))
                solver.add_variable_cost(index, feature, self.weight)
                solver.add_variable_cost(index, 1.0, self.constant)
            else:
                solver.add_variable_cost(index, 0.0, self.weight)
                solver.add_variable_cost(index, 0.0, self.constant)

    def __get_node_position(self, graph: nx.DiGraph, node: int) -> np.ndarray:
        if isinstance(self.position_attribute, tuple):
            return np.array([graph.nodes[node][p] for p in self.position_attribute])
        else:
            return np.array(graph.nodes[node][self.position_attribute])
