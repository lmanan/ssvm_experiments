from motile.costs import Costs, Weight
from motile.variables import EdgeSelected
from motile.solver import Solver
import networkx as nx
from scipy.spatial import distance
import numpy as np
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


class EdgeEmbeddingDistance(Costs):
    def __init__(self, edge_embedding_attribute, weight, constant):
        self.edge_embedding_attribute = edge_embedding_attribute
        self.weight = Weight(weight)
        self.constant = Weight(constant)

    def apply(self, solver):
        edge_variables = solver.get_variables(EdgeSelected)
        for key, index in edge_variables.items():
            if type(key[1]) is tuple:
                (start,) = key[0]
                end1, end2 = key[1]
                feature1 = self.__get_edge_embedding(solver.graph, start, end1)
                feature2 = self.__get_edge_embedding(solver.graph, start, end2)
                feature = 0.5 * (feature1 + feature2)
                solver.add_variable_cost(index, feature, self.weight)
                solver.add_variable_cost(index, 1.0, self.constant)
            else:
                u, v = cast("tuple[int, int]", key)
                feature = self.__get_edge_embedding(solver.graph, u, v)
                solver.add_variable_cost(index, feature, self.weight)
                solver.add_variable_cost(index, 1.0, self.constant)

    def __get_edge_embedding(self, graph, node_start, node_end):
        edge = (node_start, node_end)
        if self.edge_embedding_attribute in graph.edges[edge]:
            return np.array([graph.edges[edge][self.edge_embedding_attribute]])
        else:
            return np.array([0.0])


class NodeEmbeddingDistance(Costs):
    def __init__(self, node_embedding_attribute, weight, constant):
        self.node_embedding_attribute = node_embedding_attribute
        self.weight = Weight(weight)
        self.constant = Weight(constant)

    def apply(self, solver):
        edge_variables = solver.get_variables(EdgeSelected)
        for key, index in edge_variables.items():
            if type(key[1]) is tuple:  # must be a hyper edge ...
                (start,) = key[0]
                end1, end2 = key[1]
                embedding_start = self.__get_node_embedding(solver.graph, start)
                embedding_end1 = self.__get_node_embedding(solver.graph, end1)
                embedding_end2 = self.__get_node_embedding(solver.graph, end2)
                feature = distance.cosine(
                    embedding_start, 0.5 * (embedding_end1 + embedding_end2)
                )
                solver.add_variable_cost(index, feature, self.weight)
                solver.add_variable_cost(index, 1.0, self.constant)
            else:  # normal edge
                u, v = cast("tuple[int, int]", key)
                embedding_u = self.__get_node_embedding(solver.graph, u)
                embedding_v = self.__get_node_embedding(solver.graph, v)
                feature = distance.cosine(embedding_u, embedding_v)
                solver.add_variable_cost(index, feature, self.weight)
                solver.add_variable_cost(index, 1.0, self.constant)

    def __get_node_embedding(self, graph: nx.DiGraph, node: int) -> np.ndarray:
        if isinstance(self.node_embedding_attribute, tuple):
            return np.array(
                [graph.nodes[node][p] for p in self.node_embedding_attribute]
            )
        else:
            return np.array(graph.nodes[node][self.node_embedding_attribute])


class TimeGap(Costs):
    def __init__(
        self,
        time_attribute: str | tuple[str, ...],
        weight: float = 1.0,
        constant: float = 0.0,
    ) -> None:
        self.time_attribute = time_attribute
        self.weight = Weight(weight)
        self.constant = Weight(constant)

    def apply(self, solver: Solver) -> None:
        edge_variables = solver.get_variables(EdgeSelected)
        for key, index in edge_variables.items():
            if type(key[1]) is tuple:  # hyper edge
                (start,) = key[0]
                end1, end2 = key[1]
                time_start = self.__get_node_frame(solver.graph, start)
                time_end1 = self.__get_node_frame(solver.graph, end1)
                time_end2 = self.__get_node_frame(solver.graph, end2)
                feature = np.linalg.norm(time_start - 0.5 * (time_end1 + time_end2))
                solver.add_variable_cost(index, feature, self.weight)
                solver.add_variable_cost(index, 1.0, self.constant)
            else:
                u, v = cast("tuple[int, int]", key)
                time_u = self.__get_node_frame(solver.graph, u)
                time_v = self.__get_node_frame(solver.graph, v)
                feature = np.linalg.norm(time_u - time_v)
                solver.add_variable_cost(index, feature, self.weight)
                solver.add_variable_cost(index, 1.0, self.constant)

    def __get_node_frame(self, graph: nx.DiGraph, node: int) -> np.ndarray:
        if isinstance(self.time_attribute, tuple):
            return np.array([graph.nodes[node][p] for p in self.time_attribute])
        else:
            return np.array(graph.nodes[node][self.time_attribute])
