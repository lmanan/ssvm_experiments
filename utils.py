from natsort import natsorted
from pathlib import Path
import tifffile
import networkx as nx
from itertools import combinations
import numpy as np
import json
import motile
from motile.constraints import MaxParents, MaxChildren
from motile.costs import Appear, Disappear
from new_costs import (
    EdgeDistance,
    NodeEmbeddingDistance,
    EdgeEmbeddingDistance,
    TimeGap,
)
from motile_toolbox.candidate_graph import NodeAttr


def load_data(segmentation_dir_name: str, add_channel_axis=True):
    # obtain masks
    filenames = natsorted(list(Path(segmentation_dir_name).glob("*.tif")))
    segmentation = []
    for filename in filenames:
        mask = tifffile.imread(filename)
        segmentation.append(mask)
    segmentation = np.asarray(segmentation)
    if add_channel_axis:
        segmentation = segmentation[:, np.newaxis]
    return segmentation


def add_hyper_edges(candidate_graph: nx.DiGraph):
    nodes_original = list(candidate_graph.nodes)
    for node in nodes_original:
        out_edges = candidate_graph.out_edges(node)
        pairs = list(combinations(out_edges, 2))
        for pair in pairs:
            temporary_node = (
                str(pair[0][0]) + "_" + str(pair[0][1]) + "_" + str(pair[1][1])
            )
            candidate_graph.add_node(temporary_node)
            candidate_graph.add_edge(pair[0][0], temporary_node)
            candidate_graph.add_edge(
                temporary_node,
                pair[0][1],
            )
            candidate_graph.add_edge(
                temporary_node,
                pair[1][1],
            )
    return candidate_graph


def flip_edges(candidate_graph: nx.DiGraph):
    candidate_graph_copy = nx.DiGraph()
    candidate_graph_copy.add_nodes_from(candidate_graph.nodes(data=True))
    for node1, node2, data in candidate_graph.edges(data=True):
        candidate_graph_copy.add_edge(node2, node1, **data)
    return candidate_graph_copy


def add_gt_edges_to_graph(groundtruth_graph: nx.DiGraph, json_file_name: str):
    f = open(json_file_name)
    gt_data = json.load(f)

    parent_daughter_dictionary = {}

    for key, value in gt_data.items():
        parent_id = int(value[1])
        if parent_id in parent_daughter_dictionary.keys():
            pass
        else:
            parent_daughter_dictionary[parent_id] = []
        parent_daughter_dictionary[parent_id].append(
            str(int(np.min(value[0]))) + "_" + str(int(key))
        )
        # t_id ...

    for key, value in gt_data.items():
        frames, parent_id = value
        for index in range(len(frames) - 1):
            groundtruth_graph.add_edge(
                str(frames[index]) + "_" + str(key),
                str(frames[index + 1]) + "_" + str(key),
            )  # intra tracklet
        if parent_id != 0:
            time_parent = int(
                np.max(gt_data[str(parent_id)][0])
            )  # last time where parent showed up
            parent_node = str(time_parent) + "_" + str(parent_id)

            if len(parent_daughter_dictionary[parent_id]) == 1:
                groundtruth_graph.add_edge(
                    parent_node, parent_daughter_dictionary[parent_id][0]
                )
            elif len(parent_daughter_dictionary[parent_id]) == 2:
                temporary_node = parent_node
                for daughter_node in parent_daughter_dictionary[parent_id]:
                    temporary_node += "_" + daughter_node
                groundtruth_graph.add_node(temporary_node)
                groundtruth_graph.add_edge(parent_node, temporary_node)
                for daughter_node in parent_daughter_dictionary[parent_id]:
                    groundtruth_graph.add_edge(temporary_node, daughter_node)
    return groundtruth_graph


def add_costs(
    solver: motile.Solver,
    dT: int,
    node_embedding_exists: bool,
    edge_embedding_exists: bool,
):
    solver.add_costs(
        EdgeDistance(weight=1.0, constant=-20.0, position_attribute=NodeAttr.POS.value),
        name="Edge Distance",
    )
    if dT > 1:
        solver.add_costs(
            TimeGap(weight=1.0, constant=0.0, time_attribute=NodeAttr.TIME.value),
            name="Time Gap",
        )
    if node_embedding_exists:
        solver.add_costs(
            NodeEmbeddingDistance(
                node_embedding_attribute="node embedding", weight=1.0, constant=-0.5
            ),
            name="A.E. Embedding Distance",
        )
    if edge_embedding_exists:
        solver.add_costs(
            EdgeEmbeddingDistance(
                edge_embedding_attribute="edge_embedding", weight=-1.0, constant=0.5
            ),
            name="Attrackt Affinity",
        )
    solver.add_costs(Appear(constant=0.6, ignore_attribute="ignore_appear_cost"))
    solver.add_costs(Disappear(constant=0.6, ignore_attribute="ignore_disappear_cost"))
    return solver


def add_constraints(solver: motile.Solver):
    solver.add_constraints(MaxParents(1))
    solver.add_constraints(MaxChildren(1))
    return solver