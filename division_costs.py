import networkx as nx
from motile.costs import Costs, Weight
import numpy as np
from motile.variables import EdgeSelected
from typing import cast


class EdgeDistance(Costs):
    def __init__(self, position_attribute, weight, constant):
        self.position_attribute = position_attribute
        self.weight = Weight(weight)
        self.constant = Weight(constant)

    def apply(self, solver):
        edge_variables = solver.get_variables(EdgeSelected)
        for key, index in edge_variables.items():
            if type(key[1]) is tuple:  # must be a hyper edge ...
                (start,) = key[0]
                end1, end2 = key[1]
                pos_start = self.__get_node_position(solver.graph, start)
                pos_end1 = self.__get_node_position(solver.graph, end1)
                pos_end2 = self.__get_node_position(solver.graph, end2)
                feature = np.linalg.norm(pos_start - 0.5 * (pos_end1 + pos_end2))
                solver.add_variable_cost(index, feature, self.weight)
                solver.add_variable_cost(index, 1.0, self.constant)
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


class AreaSplit(Costs):
    def __init__(self, weight, area_attribute, constant):
        self.weight = Weight(weight)
        self.constant = Weight(constant)
        self.area_attribute = area_attribute

    def apply(self, solver):
        edge_variables = solver.get_variables(EdgeSelected)
        for key, index in edge_variables.items():
            if type(key[1]) is tuple:  # hyper edge
                (start,) = key[0]
                end1, end2 = key[1]
                area_start = self.__get_node_area(solver.graph, start)
                area_end1 = self.__get_node_area(solver.graph, end1)
                area_end2 = self.__get_node_area(solver.graph, end2)
                feature = np.linalg.norm(area_start - (area_end1 + area_end2))
                solver.add_variable_cost(index, feature, self.weight)
                solver.add_variable_cost(index, 1.0, self.constant)
            else:  # normal edge
                u, v = cast("tuple[int, int]", key)
                pos_u = self.__get_node_area(solver.graph, u)
                pos_v = self.__get_node_area(solver.graph, v)
                feature = np.linalg.norm(pos_u - pos_v)
                solver.add_variable_cost(index, feature, self.weight)
                solver.add_variable_cost(index, 1.0, self.constant)

    def __get_node_area(self, graph: nx.DiGraph, node: int) -> np.ndarray:
        if isinstance(self.area_attribute, tuple):
            return np.array([graph.nodes[node][p] for p in self.area_attribute])
        else:
            return np.array(graph.nodes[node][self.area_attribute])
