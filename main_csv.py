import numpy as np
from utils import (
    add_gt_edges_to_graph_2,
    load_csv_data,
    flip_edges,
    add_hyper_edges,
    add_costs,
    add_constraints,
    expand_position,
    add_app_disapp_attributes,
)
from motile_toolbox.candidate_graph import (
    get_candidate_graph_from_points_list,
    graph_to_nx,
    NodeAttr,
    EdgeAttr,
)
from motile import TrackGraph, Solver
from jsonargparse import ArgumentParser
import pprint
import os
from saving_utils import save_result
from run_traccuracy import compute_metrics
import json
import logging
import networkx as nx
import sys
from yaml import load, Loader

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s %(name)s %(levelname)-8s %(message)s"
)
logger = logging.getLogger(__name__)

pp = pprint.PrettyPrinter(indent=4)


def track(yaml_config_file_name: str):

    with open(yaml_config_file_name) as stream:
        args = load(stream, Loader=Loader)

    print("+" * 10)
    pp.pprint(args)
    print("+" * 10)

    train_csv_file_name = args["train_csv_file_name"]
    val_csv_file_name = args["val_csv_file_name"]
    num_nearest_neighbours = args["num_nearest_neighbours"]
    max_edge_distance = args["max_edge_distance"]
    direction_candidate_graph = args["direction_candidate_graph"]
    dT = args["dT"]
    train_node_embedding_file_name = args["train_node_embedding_file_name"]
    val_node_embedding_file_name = args["val_node_embedding_file_name"]
    train_edge_embedding_file_name = args["train_edge_embedding_file_name"]
    val_edge_embedding_file_name = args["val_edge_embedding_file_name"]
    regularizer_weight = args["regularizer_weight"]
    val_image_shape = args["val_image_shape"]
    pin_nodes = args["pin_nodes"]
    use_edge_distance = args["use_edge_distance"]
    write_tifs = args["write_tifs"]
    ssvm_weights_array = args["ssvm_weights_array"]
    results_dir_name = args["results_dir_name"]

    assert direction_candidate_graph in ["forward", "backward"]

    print("Created jsons directory.")
    if os.path.exists(results_dir_name + "/jsons/"):
        pass
    else:
        os.makedirs(results_dir_name + "/jsons/")

    print("Saved args in 'jsons/args.json' file.")
    with open(results_dir_name + "/jsons/args.json", "w") as f:
        json.dump(args, f)

    # ++++++++
    # Step 1 - build `train` and `val` candidate graphs
    # ++++++++

    print("\nBuilding train and val track graph ...\n")

    if ssvm_weights_array is None:
        train_array = load_csv_data(csv_file_name=train_csv_file_name)

        print(f"Train array has shape {train_array.shape}.")

        train_t_min = int(np.min(train_array[:, 1]))
        train_t_max = int(np.max(train_array[:, 1]))

        print(
            f"Min train time point is {train_t_min}, Max train time point is {train_t_max}."
        )

    val_array = load_csv_data(csv_file_name=val_csv_file_name)

    print(f"Val array has shape {val_array.shape}.")

    val_t_min = int(np.min(val_array[:, 1]))
    val_t_max = int(np.max(val_array[:, 1]))
    print(f"Min val time point is {val_t_min}, Max val time point is {val_t_max}.")

    if ssvm_weights_array is None:
        train_candidate_graph_initial = get_candidate_graph_from_points_list(
            points_list=train_array,
            max_edge_distance=max_edge_distance,
            num_nearest_neighbours=num_nearest_neighbours,
            direction_candidate_graph=direction_candidate_graph,
            dT=dT,
        )

    val_candidate_graph_initial = get_candidate_graph_from_points_list(
        points_list=val_array,
        max_edge_distance=max_edge_distance,
        num_nearest_neighbours=num_nearest_neighbours,
        direction_candidate_graph=direction_candidate_graph,
        dT=dT,
    )

    if direction_candidate_graph == "backward":
        if ssvm_weights_array is None:
            train_candidate_graph_initial = flip_edges(train_candidate_graph_initial)
        val_candidate_graph_initial = flip_edges(val_candidate_graph_initial)

    # add train_node_embedding
    if ssvm_weights_array is None:
        if train_node_embedding_file_name is not None:
            print("Adding train node embedding ...")

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

        if train_edge_embedding_file_name is not None:
            print("Adding train edge embedding ...")

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

    if val_node_embedding_file_name is not None:
        print("Adding val node embedding ...")

        val_embedding_data = np.loadtxt(val_node_embedding_file_name, delimiter=" ")
        for row in val_embedding_data:
            id_, t = int(row[0]), int(row[1])
            node_id = str(t) + "_" + str(id_)
            val_candidate_graph_initial.nodes[node_id][
                NodeAttr.NODE_EMBEDDING.value
            ] = row[
                2:
            ]  # seg_id t ...

    if val_edge_embedding_file_name is not None:
        print("Adding val edge embedding ...")
        val_edge_embedding_data = np.loadtxt(
            val_edge_embedding_file_name, delimiter=" "
        )
        for row in val_edge_embedding_data:
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
            if edge_id in val_candidate_graph_initial.edges:
                val_candidate_graph_initial.edges[edge_id][
                    EdgeAttr.EDGE_EMBEDDING.value
                ] = weight

    # add hyper edges
    if ssvm_weights_array is None:
        train_candidate_graph = add_hyper_edges(
            candidate_graph=train_candidate_graph_initial
        )
    val_candidate_graph = add_hyper_edges(candidate_graph=val_candidate_graph_initial)

    # make track graph
    if ssvm_weights_array is None:
        train_track_graph = TrackGraph(
            nx_graph=train_candidate_graph, frame_attribute="time"
        )
        train_track_graph = add_app_disapp_attributes(
            train_track_graph, train_t_min, train_t_max
        )

        print(
            f"Number of nodes in train graph is {len(train_track_graph.nodes)} and edges is {len(train_track_graph.edges)}."
        )

    val_track_graph = TrackGraph(nx_graph=val_candidate_graph, frame_attribute="time")
    val_track_graph = add_app_disapp_attributes(val_track_graph, val_t_min, val_t_max)

    print(
        f"Number of nodes in val graph is {len(val_track_graph.nodes)} and edges is {len(val_track_graph.edges)}."
    )

    if ssvm_weights_array is None:
        max_out_edges = 0
        max_in_edges = 0
        for node in train_candidate_graph.nodes:
            num_next = len(train_candidate_graph.out_edges(node))
            if num_next > max_out_edges:
                max_out_edges = num_next

            num_prev = len(train_candidate_graph.in_edges(node))
            if num_prev > max_in_edges:
                max_in_edges = num_prev

        print(f"Maximum number of out edges is {max_out_edges}.")
        print(f"Maximum number of in edges {max_in_edges}.")
        temp_limit = np.maximum(max_in_edges, max_out_edges) + 500
        if temp_limit > 1000:
            sys.setrecursionlimit(temp_limit)

    # ++++++++
    # Step 2 - build `gt` graph
    # ++++++++
    if ssvm_weights_array is None:
        print("\nBuilding GT track graph ...\n")

        groundtruth_graph = nx.DiGraph()
        groundtruth_graph.add_nodes_from(train_candidate_graph_initial.nodes(data=True))

        # add edges
        groundtruth_graph = add_gt_edges_to_graph_2(
            groundtruth_graph=groundtruth_graph, gt_data=train_array
        )

        # convert to track graph
        groundtruth_track_graph = TrackGraph(
            nx_graph=groundtruth_graph, frame_attribute="time"
        )

    # ++++++++
    # Step 3 - fit weights on `groundtruth_track_graph`
    # ++++++++

    print("\nFitting weights using SSVM...\n")

    # add `gt` attribute on edge
    if ssvm_weights_array is None:
        for node_id in train_track_graph.nodes:
            for edge_id in list(train_track_graph.next_edges[node_id]):
                if edge_id in groundtruth_track_graph.edges:
                    train_track_graph.edges[edge_id]["gt"] = True
                else:
                    train_track_graph.edges[edge_id]["gt"] = False

        # fit weights
        solver = Solver(track_graph=train_track_graph)
        node_embedding_exists = (
            False if train_node_embedding_file_name is None else True
        )
        edge_embedding_exists = (
            False if train_edge_embedding_file_name is None else True
        )

        solver = add_costs(
            solver=solver,
            dT=dT,
            use_edge_distance=use_edge_distance,
            node_embedding_exists=node_embedding_exists,
            edge_embedding_exists=edge_embedding_exists,
        )
        solver = add_constraints(solver=solver, pin_nodes=pin_nodes)
        solver.fit_weights(
            gt_attribute="gt",
            regularizer_weight=regularizer_weight,
            max_iterations=1000,
        )
        ssvm_weights = solver.weights
        weights_by_name = ssvm_weights._weights_by_name
        print(f"After SSVM fitting, weights are {ssvm_weights}")
        ssvm_weights_array = ssvm_weights.to_ndarray()
        fitted_weights_dictionary = {}
        for i, key in enumerate(weights_by_name):
            fitted_weights_dictionary[key[0] + "_" + key[1]] = str(
                ssvm_weights_array[i]
            )

        with open(results_dir_name + "/jsons/weights.json", "w") as f:
            json.dump(fitted_weights_dictionary, f)

    # ++++++++
    # Step 4 - apply weights on `val_track_graph` and compute scores ...
    # ++++++++

    print("\nApplying weights on val track graph and getting traccuracy metrics ...\n")

    solver = Solver(track_graph=val_track_graph)
    node_embedding_exists = False if val_node_embedding_file_name is None else True
    edge_embedding_exists = False if val_edge_embedding_file_name is None else True
    solver = add_costs(
        solver=solver,
        dT=dT,
        use_edge_distance=use_edge_distance,
        node_embedding_exists=node_embedding_exists,
        edge_embedding_exists=edge_embedding_exists,
    )
    solver = add_constraints(solver=solver, pin_nodes=pin_nodes)
    solver.weights.from_ndarray(ssvm_weights_array)

    solution = solver.solve(verbose=True)
    solution_graph = solver.get_selected_subgraph(solution)

    ilp_results_data = []
    for edge in solution_graph.edges:
        u, v = edge
        if isinstance(u, tuple):
            u = u[0]

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

    print(
        f"After optimization, we selected {len(solution_graph.nodes)} nodes and {len(solution_graph.edges)} edges."
    )

    # since val_segmentation is not available, as csvs are available.

    val_segmentation = np.zeros(
        (val_t_max + 1, *tuple(val_image_shape)), dtype=np.uint64
    )
    for node, attrs in val_candidate_graph_initial.nodes.items():
        t, id_ = node.split("_")
        t, id_ = int(t), int(id_)
        position = attrs[NodeAttr.POS.value]
        val_segmentation[t] = expand_position(
            data=val_segmentation[t], position=position, id_=id_
        )

    print("Saving results ...")
    new_mapping, res_track, tracked_masks, G = save_result(
        solution_nx_graph=graph_to_nx(solution_graph),
        segmentation=val_segmentation,
        output_tif_dir_name=results_dir_name,
        write_tifs=write_tifs,
    )

    print("Computing scores ...")

    val_gt_graph = nx.DiGraph()
    val_gt_graph.add_nodes_from(val_candidate_graph_initial.nodes(data=True))
    val_gt_graph = add_gt_edges_to_graph_2(
        groundtruth_graph=val_gt_graph, gt_data=val_array
    )

    # convert to track graph
    val_gt_track_graph = TrackGraph(nx_graph=val_gt_graph, frame_attribute="time")
    print(
        f"Number of nodes in the test imaging dataset is {len(val_gt_track_graph.nodes)} and edges is {len(val_gt_track_graph.edges)}."
    )

    for node in val_gt_track_graph.nodes:
        pos = val_gt_track_graph.nodes[node]["pos"]
        if len(pos) == 2:
            y, x = pos
            val_gt_track_graph.nodes[node]["y"] = y
            val_gt_track_graph.nodes[node]["x"] = x
        elif len(pos) == 3:
            z, y, x = pos
            val_gt_track_graph.nodes[node]["z"] = z
            val_gt_track_graph.nodes[node]["y"] = y
            val_gt_track_graph.nodes[node]["x"] = x

    compute_metrics(
        gt_segmentation=val_segmentation,
        gt_nx_graph=graph_to_nx(val_gt_track_graph),
        predicted_segmentation=tracked_masks,
        pred_nx_graph=G,
        results_dir_name=results_dir_name,
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--yaml_config_file_name", dest="yaml_config_file_name")
    args = parser.parse_args()
    track(yaml_config_file_name=args.yaml_config_file_name)
