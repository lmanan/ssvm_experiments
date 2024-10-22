from natsort import natsorted
from pathlib import Path
import tifffile
import networkx as nx
from itertools import combinations
import numpy as np
import json
import motile
from motile.constraints import MaxParents, MaxChildren, Pin
from motile.costs import Appear, Disappear
from costs import (
    EdgeDistance,
    NodeEmbeddingDistance,
    EdgeEmbeddingDistance,
    TimeGap,
)
from motile.variables import NodeSelected, EdgeSelected, NodeAppear, NodeDisappear
from motile_toolbox.candidate_graph import NodeAttr, EdgeAttr
from typing import List


def save_ilp_result(solution_graph, results_dir_name):
    ilp_results_data = []
    for edge in solution_graph.edges:
        u, v = edge
        if isinstance(u, tuple):
            (u,) = u

        t_u, id_u = u.split("_")
        t_u, id_u = int(t_u), int(id_u)
        if isinstance(v, tuple):
            m, n = v

            t_m, id_m = m.split("_")
            t_m, id_m = int(t_m), int(id_m)

            t_n, id_n = n.split("_")
            t_n, id_n = int(t_n), int(id_n)

            ilp_results_data.append([id_u, t_u, id_m, t_m])
            ilp_results_data.append([id_u, t_u, id_n, t_n])
        else:
            t_v, id_v = v.split("_")
            t_v, id_v = int(t_v), int(id_v)

            ilp_results_data.append([id_u, t_u, id_v, t_v])

    np.savetxt(
        results_dir_name + "/jsons/ilp.csv",
        np.asarray(ilp_results_data),
        delimiter=" ",
        fmt=["%i", "%i", "%i", "%i"],
    )


def set_ground_truth_mask(solver: motile.Solver, gt_attribute: str = "gt"):
    """set_ground_truth_mask.
    This function tries to figure out which variables we have gt annotation
    for.

    """

    mask = np.zeros((solver.num_variables), dtype=np.float32)
    ground_truth = np.zeros_like(mask)
    # if nodes have `gt_attribute` specified, set mask and groundtruth for NodeSelected
    # variables.

    for node, index in solver.get_variables(NodeSelected).items():
        gt = solver.graph.nodes[node].get(gt_attribute, None)
        if gt is not None:
            mask[index] = 1.0
            ground_truth[index] = gt

    # if edges have `gt_attribute` specified, set mask and ground truth for
    # `EdgeSelected` variables.
    # IMPORTANT:
    # If groundtruth annotation value is 1.0, then we can also say
    # that mask and groundtruth for NodeDisappear for starting node
    # is known and mask and ground truth for Node Appear for ending
    # nodes is known.

    for edge, index in solver.get_variables(EdgeSelected).items():
        u, v = edge
        if isinstance(v, tuple):
            (u,) = u
            (v1, v2) = v
            index_v1_appear = solver.get_variables(NodeAppear)[v1]
            index_v2_appear = solver.get_variables(NodeAppear)[v2]
            index_u_disappear = solver.get_variables(NodeDisappear)[u]
            gt = solver.graph.edges[edge].get(gt_attribute, None)
            if gt is not None:
                mask[index] = 1.0
                ground_truth[index] = gt
                if gt == 1.0:
                    mask[index_u_disappear] = 1.0
                    ground_truth[index_u_disappear] = 0
                    mask[index_v1_appear] = 1.0
                    ground_truth[index_v1_appear] = 0
                    mask[index_v2_appear] = 1.0
                    ground_truth[index_v2_appear] = 0
        else:
            index_v_appear = solver.get_variables(NodeAppear)[v]
            index_u_disappear = solver.get_variables(NodeDisappear)[u]
            gt = solver.graph.edges[edge].get(gt_attribute, None)
            if gt is not None:
                mask[index] = 1.0
                ground_truth[index] = gt
                if gt == 1.0:
                    mask[index_u_disappear] = 1.0
                    ground_truth[index_u_disappear] = 0
                    mask[index_v_appear] = 1.0
                    ground_truth[index_v_appear] = 0

    return ground_truth, mask


def get_recursion_limit(candidate_graph):
    max_in_edges = max_out_edges = 0
    for node in candidate_graph.nodes:
        num_next = len(candidate_graph.out_edges(node))
        if num_next > max_out_edges:
            max_out_edges = num_next

        num_prev = len(candidate_graph.in_edges(node))
        if num_prev > max_in_edges:
            max_in_edges = num_prev

    print("+" * 10)
    print(f"Maximum out edges is {max_out_edges}, max in edges {max_in_edges}.")
    temp_limit = np.maximum(max_in_edges, max_out_edges) + 500
    return temp_limit


def load_tif_data(segmentation_dir_name: str, add_channel_axis=True):
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


def load_csv_data(csv_file_name: str, delimiter=" "):
    """Assuming ids are seg_id t z y x p_id"""
    data = np.genfromtxt(fname=csv_file_name, delimiter=delimiter)

    return data


def add_hyper_edges(candidate_graph: nx.DiGraph):
    candidate_graph_copy = nx.DiGraph()
    candidate_graph_copy.add_nodes_from(candidate_graph.nodes(data=True))
    candidate_graph_copy.add_edges_from(candidate_graph.edges(data=True))
    nodes_original = list(candidate_graph_copy.nodes)
    for node in nodes_original:
        out_edges = candidate_graph_copy.out_edges(node)
        pairs = list(combinations(out_edges, 2))
        for pair in pairs:
            temporary_node = (
                str(pair[0][0]) + "_" + str(pair[0][1]) + "_" + str(pair[1][1])
            )
            candidate_graph_copy.add_node(temporary_node)
            candidate_graph_copy.add_edge(pair[0][0], temporary_node)
            candidate_graph_copy.add_edge(
                temporary_node,
                pair[0][1],
            )
            candidate_graph_copy.add_edge(
                temporary_node,
                pair[1][1],
            )
    return candidate_graph_copy


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


def add_gt_edges_to_graph_2(groundtruth_graph: nx.DiGraph, gt_data: np.ndarray):
    """gt data will have last column as parent id column."""
    parent_daughter_dictionary = {}  # parent_id : list of daughter ids
    id_time_dictionary = {}  # id_: time it shows up
    for row in gt_data:
        id_, t, parent_id = int(row[0]), int(row[1]), int(row[-1])
        if parent_id <= 0:  # new track starts
            pass
        else:
            if parent_id in parent_daughter_dictionary:
                parent_daughter_dictionary[parent_id].append(id_)
            else:
                parent_daughter_dictionary[parent_id] = [id_]
    for row in gt_data:
        id_, t, parent_id = int(row[0]), int(row[1]), int(row[-1])
        id_time_dictionary[id_] = t

    for row in gt_data:
        id_, t, parent_id = int(row[0]), int(row[1]), int(row[-1])
        if parent_id > 0:
            parent_time = id_time_dictionary[parent_id]
            start_node = str(parent_time) + "_" + str(parent_id)
            if len(parent_daughter_dictionary[parent_id]) == 1:
                end_node = str(t) + "_" + str(id_)
                groundtruth_graph.add_edge(start_node, end_node)
            elif len(parent_daughter_dictionary[parent_id]) == 2:
                temporary_node = start_node
                for daughter_node in parent_daughter_dictionary[parent_id]:
                    temporary_node += "_" + str(t) + "_" + str(daughter_node)
                groundtruth_graph.add_node(temporary_node)
                groundtruth_graph.add_edge(start_node, temporary_node)
                for daughter_node in parent_daughter_dictionary[parent_id]:
                    end_node = str(t) + "_" + str(daughter_node)
                    groundtruth_graph.add_edge(temporary_node, end_node)
    return groundtruth_graph


def add_costs(
    solver: motile.Solver,
    dT: int,
    use_edge_distance: bool,
    node_embedding_exists: bool,
    edge_embedding_exists: bool,
    mean_edge_distance: float = None,
    std_edge_distance: float = None,
    mean_node_embedding_distance: float = None,
    std_node_embedding_distance: float = None,
    mean_edge_embedding_distance: float = None,
    std_edge_embedding_distance: float = None,
):
    if use_edge_distance:
        solver.add_costs(
            EdgeDistance(
                weight=1.0,
                constant=-20.0,
                position_attribute=NodeAttr.POS.value,
                mean_edge_distance=mean_edge_distance,
                std_edge_distance=std_edge_distance,
            ),
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
                node_embedding_attribute=NodeAttr.NODE_EMBEDDING.value,
                weight=1.0,
                constant=-0.5,
                mean_node_embedding_distance=mean_node_embedding_distance,
                std_node_embedding_distance=std_node_embedding_distance,
            ),
            name="A.E. Embedding Distance",
        )
    if edge_embedding_exists:
        solver.add_costs(
            EdgeEmbeddingDistance(
                edge_embedding_attribute=EdgeAttr.EDGE_EMBEDDING.value,
                weight=-1.0,
                constant=0.5,
                mean_edge_embedding_distance=mean_edge_embedding_distance,
                std_edge_embedding_distance=std_edge_embedding_distance,
            ),
            name="Attrackt Affinity",
        )
    solver.add_costs(Appear(constant=0.6, ignore_attribute="ignore_appear_cost"))
    solver.add_costs(Disappear(constant=0.6, ignore_attribute="ignore_disappear_cost"))
    return solver


def add_constraints(solver: motile.Solver, pin_nodes: bool):
    solver.add_constraints(MaxParents(1))
    solver.add_constraints(MaxChildren(1))
    if pin_nodes:
        solver.add_constraints(Pin(attribute=NodeAttr.PINNED.value))
    return solver


def expand_position(
    data: np.ndarray,
    position: List,
    id_: int,
    nhood: int = 1,
):
    outside = True
    if len(position) == 2:
        H, W = data.shape
        y, x = position
        y, x = int(y), int(x)
        while outside:
            data_ = data[
                np.maximum(y - nhood, 0) : np.minimum(y + nhood + 1, H),
                np.maximum(x - nhood, 0) : np.minimum(x + nhood + 1, W),
            ]
            if 0 in data_.shape:
                nhood += 1
            else:
                outside = False
        data[
            np.maximum(y - nhood, 0) : np.minimum(y + nhood + 1, H),
            np.maximum(x - nhood, 0) : np.minimum(x + nhood + 1, W),
        ] = id_
    elif len(position) == 3:
        D, H, W = data.shape
        z, y, x = position
        z, y, x = int(z), int(y), int(x)
        while outside:
            data_ = data[
                np.maximum(z - nhood, 0) : np.minimum(z + nhood + 1, D),
                np.maximum(y - nhood, 0) : np.minimum(y + nhood + 1, H),
                np.maximum(x - nhood, 0) : np.minimum(x + nhood + 1, W),
            ]
            if 0 in data_.shape:
                nhood += 1
            else:
                outside = False
        data[
            np.maximum(z - nhood, 0) : np.minimum(z + nhood + 1, D),
            np.maximum(y - nhood, 0) : np.minimum(y + nhood + 1, H),
            np.maximum(x - nhood, 0) : np.minimum(x + nhood + 1, W),
        ] = id_
    return data


def add_app_disapp_attributes(track_graph: motile.TrackGraph, t_min, t_max):
    num_nodes_previous = {}
    num_nodes_next = {}
    num_nodes_current = {}
    for t in range(t_min, t_max + 1):
        if t == t_min:
            num_nodes_previous[t_min] = 0
        else:
            num_nodes_previous[t] = len(track_graph.nodes_by_frame(t - 1))

        if t == t_max:
            num_nodes_next[t_max] = 0
        else:
            num_nodes_next[t] = len(track_graph.nodes_by_frame(t + 1))
        num_nodes_current[t] = len(track_graph.nodes_by_frame(t))

    for node, attrs in track_graph.nodes.items():
        time = attrs[NodeAttr.TIME.value]
        if num_nodes_previous[time] == 0 and num_nodes_current[time] != 0:
            track_graph.nodes[node][NodeAttr.IGNORE_APPEAR_COST.value] = True
        if num_nodes_next[time] == 0 and num_nodes_current[time] != 0:
            track_graph.nodes[node][NodeAttr.IGNORE_DISAPPEAR_COST.value] = True
    return track_graph
