from yaml import load, Loader
import json
import sys
import os
import jsonargparse
import numpy as np
from utils import (
    get_recursion_limit,
    set_ground_truth_mask,
    set_feature_mask_app_disapp,
    load_csv_data,
    flip_edges,
    add_hyper_edges,
    add_costs,
    add_constraints,
    add_app_disapp_attributes,
)
from motile_toolbox.candidate_graph import (
    get_candidate_graph_from_points_list,
    NodeAttr,
    EdgeAttr,
)
from motile import TrackGraph, Solver
import pprint
import logging

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s %(name)s %(levelname)-8s %(message)s"
)
logger = logging.getLogger(__name__)


pp = pprint.PrettyPrinter(indent=4)


def train(yaml_config_file_name: str):

    with open(yaml_config_file_name) as stream:
        args = load(stream, Loader=Loader)

    print("+" * 10)
    pp.pprint(args)
    train_csv_file_name = args["train_csv_file_name"]
    num_nearest_neighbours = args["num_nearest_neighbours"]
    max_edge_distance = args["max_edge_distance"]
    direction_candidate_graph = args["direction_candidate_graph"]
    dT = args["dT"]
    train_node_embedding_file_name = args["train_node_embedding_file_name"]
    train_edge_embedding_file_name = args["train_edge_embedding_file_name"]
    regularizer_weight = args["regularizer_weight"]
    pin_nodes = args["pin_nodes"]
    use_edge_distance = args["use_edge_distance"]
    whitening = args["whitening"]
    supervised_json_file_name = args["supervised_json_file_name"]
    results_dir_name = args["results_dir_name"]

    assert direction_candidate_graph in ["forward", "backward"]

    if os.path.exists(results_dir_name + "/jsons/"):
        pass
    else:
        os.makedirs(results_dir_name + "/jsons/")

    print("+" * 10)
    print("Saved args in 'jsons/args.json' file.")
    with open(results_dir_name + "/jsons/args.json", "w") as f:
        json.dump(args, f)

    # ++++++++
    # Step 1 - build `train` candidate graph
    # ++++++++

    train_array = load_csv_data(csv_file_name=train_csv_file_name)

    print("+" * 10)
    print(f"Train array has shape {train_array.shape}.")
    train_t_min = int(np.min(train_array[:, 1]))
    train_t_max = int(np.max(train_array[:, 1]))

    print(
        f"Min train time point is {train_t_min}, Max train time point is {train_t_max}."
    )

    mean_edge_distance = std_edge_distance = None
    mean_node_embedding_distance = std_node_embedding_distance = None
    mean_edge_embedding_distance = std_edge_embedding_distance = None

    train_candidate_graph_initial, mean_edge_distance, std_edge_distance = (
        get_candidate_graph_from_points_list(
            points_list=train_array,
            max_edge_distance=max_edge_distance,
            num_nearest_neighbours=num_nearest_neighbours,
            direction_candidate_graph=direction_candidate_graph,
            dT=dT,
            whitening=whitening,
        )
    )

    print("+" * 10)
    print(
        f"Mean edge distance {mean_edge_distance}, Std edge distance {std_edge_distance}."
    )

    if direction_candidate_graph == "backward":
        train_candidate_graph_initial = flip_edges(train_candidate_graph_initial)

    print("+" * 10)
    print(
        f"Number of nodes in train graph initial is {len(train_candidate_graph_initial.nodes)} and edges is {len(train_candidate_graph_initial.edges)}."
    )

    # add train_node_embedding
    if train_node_embedding_file_name is not None:
        train_node_embedding_data = np.loadtxt(
            train_node_embedding_file_name, delimiter=" "
        )
        for row in train_node_embedding_data:
            id_, t = int(row[0]), int(row[1])
            node_id = str(t) + "_" + str(id_)
            train_candidate_graph_initial.nodes[node_id][
                NodeAttr.NODE_EMBEDDING.value
            ] = row[
                2:
            ]  # seg_id t ...

        if whitening:
            node_embedding_distance_list = []
            for edge_id in train_candidate_graph_initial.edges:
                u, value = edge_id
                node_embedding_u = train_candidate_graph_initial.nodes[u][
                    NodeAttr.NODE_EMBEDDING.value
                ]
                node_embedding_v = train_candidate_graph_initial.nodes[value][
                    NodeAttr.NODE_EMBEDDING.value
                ]
                d = np.linalg.norm(node_embedding_u - node_embedding_v)
                node_embedding_distance_list.append(d)
            mean_node_embedding_distance = np.mean(node_embedding_distance_list)
            std_node_embedding_distance = np.std(node_embedding_distance_list)

            print("+" * 10)
            print(
                f"Mean node embedding distance {mean_node_embedding_distance} std node embedding distance {std_node_embedding_distance}."
            )

    if train_edge_embedding_file_name is not None:
        train_edge_embedding_data = np.loadtxt(
            train_edge_embedding_file_name, delimiter=" "
        )
        for row in train_edge_embedding_data:
            id_a, t_a, id_b, t_b, weight = row
            id_a, t_a, id_b, t_b, weight = (
                int(id_a),
                int(t_a),
                int(id_b),
                int(t_b),
                float(weight),
            )
            node_a = str(t_a) + "_" + str(id_a)
            node_b = str(t_b) + "_" + str(id_b)
            edge_id = (node_a, node_b)
            if edge_id in train_candidate_graph_initial.edges:
                train_candidate_graph_initial.edges[edge_id][
                    EdgeAttr.EDGE_EMBEDDING.value
                ] = weight

        if whitening:
            edge_embedding_distance_list = []
            for edge_id in train_candidate_graph_initial.edges:
                if (
                    EdgeAttr.EDGE_EMBEDDING.value
                    in train_candidate_graph_initial.edges[edge_id]
                ):
                    edge_embedding_distance_list.append(
                        train_candidate_graph_initial.edges[edge_id][
                            EdgeAttr.EDGE_EMBEDDING.value
                        ]
                    )
            mean_edge_embedding_distance = np.mean(edge_embedding_distance_list)
            std_edge_embedding_distance = np.std(edge_embedding_distance_list)

            print("+" * 10)
            print(
                f"Mean edge embedding distance {mean_edge_embedding_distance} std edge embedding distance {std_edge_embedding_distance}."
            )

    # add hyper edges
    train_candidate_graph = add_hyper_edges(
        candidate_graph=train_candidate_graph_initial
    )
    # make track graph
    train_track_graph = TrackGraph(
        nx_graph=train_candidate_graph, frame_attribute="time"
    )
    train_track_graph = add_app_disapp_attributes(
        train_track_graph, train_t_min, train_t_max
    )

    print("+" * 10)
    print(
        f"Number of nodes in train track graph is {len(train_track_graph.nodes)} and edges is {len(train_track_graph.edges)}."
    )

    recursion_limit = get_recursion_limit(candidate_graph=train_candidate_graph)
    if recursion_limit > 1000:
        sys.setrecursionlimit(recursion_limit)

    # ++++++++
    # Step 2 - fit weights on ground truth annotations
    # ++++++++

    with open(supervised_json_file_name) as f:
        supervised_data = json.load(f)

    for in_node in supervised_data.keys():
        daughters = []
        for key, value in supervised_data[in_node].items():
            edge_id = (in_node, key)
            if value == 0.0:
                train_track_graph.edges[edge_id]["gt"] = False
            elif value == 1.0:
                daughters.append(key)

        if len(daughters) == 2:
            for index, daughter in enumerate(daughters):
                edge_id = (in_node, daughters[index])
                train_track_graph.edges[edge_id]["gt"] = False
            edge_id = ((in_node,), (daughters[0], daughters[1]))
            if edge_id in train_track_graph.edges:
                train_track_graph.edges[edge_id]["gt"] = True  # hyper edge

            edge_id = ((in_node,), (daughters[1], daughters[0]))
            if edge_id in train_track_graph.edges:
                train_track_graph.edges[edge_id]["gt"] = True  # hyper edge

        elif len(daughters) == 1:
            edge_id = (in_node, daughters[0])
            train_track_graph.edges[edge_id]["gt"] = True

    solver = Solver(track_graph=train_track_graph)

    node_embedding_exists = False if train_node_embedding_file_name is None else True
    edge_embedding_exists = False if train_edge_embedding_file_name is None else True

    solver = add_costs(
        solver=solver,
        dT=dT,
        use_edge_distance=use_edge_distance,
        node_embedding_exists=node_embedding_exists,
        edge_embedding_exists=edge_embedding_exists,
        mean_edge_distance=mean_edge_distance,
        std_edge_distance=std_edge_distance,
        mean_node_embedding_distance=mean_node_embedding_distance,
        std_node_embedding_distance=std_node_embedding_distance,
        mean_edge_embedding_distance=mean_edge_embedding_distance,
        std_edge_embedding_distance=std_edge_embedding_distance,
    )
    solver = add_constraints(solver=solver, pin_nodes=pin_nodes)

    ground_truth, mask = set_ground_truth_mask(solver)
    train_track_graph = set_feature_mask_app_disapp(
        ground_truth, mask, train_track_graph
    )

    solver.fit_weights(
        gt_attribute="gt",
        regularizer_weight=regularizer_weight,
        max_iterations=1000,
        ground_truth=ground_truth,
        mask=mask,
    )
    ssvm_weights = solver.weights
    weights_by_name = ssvm_weights._weights_by_name

    print("+" * 10)
    print(f"After SSVM fitting, weights are {ssvm_weights}.")
    ssvm_weights_array = ssvm_weights.to_ndarray()
    if ssvm_weights_array[-3] == 0 and ssvm_weights_array[-1] == 0:
        app_bias = ssvm_weights_array[0] * 3.0 + ssvm_weights_array[1]
        disapp_bias = app_bias
        ssvm_weights_array[-3] = app_bias
        ssvm_weights_array[-1] = disapp_bias

    print("+" * 10)
    print(f"After adjustment, weights are {ssvm_weights_array}.")
    fitted_weights_dictionary = {}
    for index, key in enumerate(weights_by_name):
        fitted_weights_dictionary[key[0] + "_" + key[1]] = str(
            ssvm_weights_array[index]
        )

    with open(results_dir_name + "/jsons/weights.json", "w") as f:
        json.dump(fitted_weights_dictionary, f)


if __name__ == "__main__":
    parser = jsonargparse.ArgumentParser()
    parser.add_argument("--yaml_config_file_name", dest="yaml_config_file_name")
    args = parser.parse_args()
    train(yaml_config_file_name=args.yaml_config_file_name)
