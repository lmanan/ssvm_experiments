from yaml import load, Loader
import json
import jsonargparse
import os
import sys
import numpy as np
import networkx as nx
from utils import (
    get_recursion_limit,
    save_ilp_result,
    add_gt_edges_to_graph_2,
    load_csv_data,
    flip_edges,
    add_hyper_edges,
    add_costs,
    add_constraints,
    add_app_disapp_attributes,
)

from motile_toolbox.candidate_graph import (
    get_candidate_graph_from_points_list,
    graph_to_nx,
    NodeAttr,
    EdgeAttr,
)
from motile import TrackGraph, Solver
from saving_utils import save_result
from run_traccuracy import compute_metrics

import pprint

pp = pprint.PrettyPrinter(indent=4)


def infer(yaml_config_file_name):

    with open(yaml_config_file_name) as stream:
        args = load(stream, Loader=Loader)

    print("+" * 10)
    pp.pprint(args)

    test_csv_file_names = args["test_csv_file_names"]
    num_nearest_neighbours = args["num_nearest_neighbours"]
    max_edge_distance = args["max_edge_distance"]
    direction_candidate_graph = args["direction_candidate_graph"]
    dT = args["dT"]
    test_node_embedding_file_names = args["test_node_embedding_file_names"]
    test_edge_embedding_file_names = args["test_edge_embedding_file_names"]
    pin_nodes = args["pin_nodes"]
    pin_edges = args["pin_edges"]
    use_edge_distance = args["use_edge_distance"]
    write_tifs = args["write_tifs"]
    ssvm_weights_file_path = args["ssvm_weights_file_path"]
    results_dir_names = args["results_dir_names"]
    whitening = args["whitening"]
    use_different_weights_hyper = args["use_different_weights_hyper"]
    voxel_size = args["voxel_size"]
    voxel_size = [float(x) for x in voxel_size]
    supervised_csv_file_names = args["supervised_csv_file_names"]

    assert len(test_csv_file_names) == len(results_dir_names)

    for index_result, test_csv_file_name in enumerate(test_csv_file_names):

        mean_edge_distance = args["mean_edge_distance"]
        std_edge_distance = args["std_edge_distance"]
        mean_node_embedding_distance = args["mean_node_embedding_distance"]
        std_node_embedding_distance = args["std_node_embedding_distance"]
        mean_edge_embedding_distance = args["mean_edge_embedding_distance"]
        std_edge_embedding_distance = args["std_edge_embedding_distance"]

        print("+" * 10)
        print(f"Processing {test_csv_file_name}.")

        results_dir_name = results_dir_names[index_result]
        assert direction_candidate_graph in ["forward", "backward"]

        if os.path.exists(results_dir_name + "/jsons/"):
            pass
        else:
            os.makedirs(results_dir_name + "/jsons/")

        with open(results_dir_name + "/jsons/args.json", "w") as f:
            json.dump(args, f)

        # ++++++++
        # Step 1 - build test candidate graph
        # ++++++++

        test_array = load_csv_data(
            csv_file_name=test_csv_file_name, voxel_size=voxel_size
        )

        print("+" * 10)
        print(f"Test array has shape {test_array.shape}.")
        test_t_min = int(np.min(test_array[:, 1]))
        test_t_max = int(np.max(test_array[:, 1]))

        if mean_edge_distance is None and std_edge_distance is None:
            test_candidate_graph_initial, mean_edge_distance, std_edge_distance = (
                get_candidate_graph_from_points_list(
                    points_list=test_array,
                    max_edge_distance=max_edge_distance,
                    num_nearest_neighbours=num_nearest_neighbours,
                    direction_candidate_graph=direction_candidate_graph,
                    dT=dT,
                    whitening=whitening,
                )
            )
        else:
            test_candidate_graph_initial, _, _ = get_candidate_graph_from_points_list(
                points_list=test_array,
                max_edge_distance=max_edge_distance,
                num_nearest_neighbours=num_nearest_neighbours,
                direction_candidate_graph=direction_candidate_graph,
                dT=dT,
                whitening=whitening,
            )

        print("+" * 10)
        print(
            f"Mean edge distance is {mean_edge_distance} and std edge distance is {std_edge_distance}."
        )

        if direction_candidate_graph == "backward":
            test_candidate_graph_initial = flip_edges(test_candidate_graph_initial)

        print("+" * 10)
        print(
            f"Number of nodes in test graph initial is {len(test_candidate_graph_initial.nodes)} and edges is {len(test_candidate_graph_initial.edges)}. "
        )

        if test_node_embedding_file_names is not None:
            if len(test_node_embedding_file_names) == 1:
                test_node_embedding_file_name = test_node_embedding_file_names[0]
            elif len(test_node_embedding_file_names) == len(results_dir_names):
                test_node_embedding_file_name = test_node_embedding_file_names[
                    index_result
                ]
            test_embedding_data = np.loadtxt(
                test_node_embedding_file_name, delimiter=" "
            )
            for row in test_embedding_data:
                id_, t = int(row[0]), int(row[1])
                node_id = str(t) + "_" + str(id_)

                if node_id in test_candidate_graph_initial.nodes:
                    test_candidate_graph_initial.nodes[node_id][
                        NodeAttr.NODE_EMBEDDING.value
                    ] = row[
                        2:
                    ]  # seg_id t ...

            if (
                whitening
                and mean_node_embedding_distance is None
                and std_node_embedding_distance is None
            ):
                node_embedding_distance_list = []
                for edge_id in test_candidate_graph_initial.edges:
                    u, v = edge_id
                    node_embedding_u = test_candidate_graph_initial.nodes[u][
                        NodeAttr.NODE_EMBEDDING.value
                    ]
                    node_embedding_v = test_candidate_graph_initial.nodes[v][
                        NodeAttr.NODE_EMBEDDING.value
                    ]
                    d = np.linalg.norm(node_embedding_u - node_embedding_v)
                    node_embedding_distance_list.append(d)
                mean_node_embedding_distance = np.mean(node_embedding_distance_list)
                std_node_embedding_distance = np.std(node_embedding_distance_list)

                print("+" * 10)
                print(
                    f"Mean node embedding distance is {mean_node_embedding_distance} and std node embedding distance is {std_node_embedding_distance}."
                )

        if test_edge_embedding_file_names is not None:
            if len(test_edge_embedding_file_names) == 1:
                test_edge_embedding_file_name = test_edge_embedding_file_names[0]
            elif len(test_edge_embedding_file_names) == len(results_dir_names):
                test_edge_embedding_file_name = test_edge_embedding_file_names[
                    index_result
                ]

            test_edge_embedding_data = np.loadtxt(
                test_edge_embedding_file_name, delimiter=" "
            )
            for row in test_edge_embedding_data:
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
                if edge_id in test_candidate_graph_initial.edges:
                    test_candidate_graph_initial.edges[edge_id][
                        EdgeAttr.EDGE_EMBEDDING.value
                    ] = weight

            # say if only a subset of edges were provided in the text file,
            # then ...

            for edge_id in test_candidate_graph_initial.edges:
                if (
                    EdgeAttr.EDGE_EMBEDDING.value
                    in test_candidate_graph_initial.edges[edge_id]
                ):
                    pass  # already set from text file
                else:
                    test_candidate_graph_initial.edges[edge_id][
                        EdgeAttr.EDGE_EMBEDDING.value
                    ] = 0.0

            if (
                whitening
                and mean_edge_embedding_distance is None
                and std_edge_embedding_distance is None
            ):
                edge_embedding_distance_list = []
                for edge_id in test_candidate_graph_initial.edges:
                    if (
                        EdgeAttr.EDGE_EMBEDDING.value
                        in test_candidate_graph_initial.edges[edge_id]
                    ):
                        edge_embedding_distance_list.append(
                            test_candidate_graph_initial.edges[edge_id][
                                EdgeAttr.EDGE_EMBEDDING.value
                            ]
                        )
                mean_edge_embedding_distance = np.mean(edge_embedding_distance_list)
                std_edge_embedding_distance = np.std(edge_embedding_distance_list)

                print("+" * 10)
                print(
                    f"Mean edge embedding distance is {mean_edge_embedding_distance} and std edge embedding distance is {std_edge_embedding_distance}."
                )

        test_candidate_graph = add_hyper_edges(
            candidate_graph=test_candidate_graph_initial
        )

        test_track_graph = TrackGraph(
            nx_graph=test_candidate_graph, frame_attribute="time"
        )
        # ++++++
        # Step 1a - figure out out edges to pin
        # ++++++

        if supervised_csv_file_names[index_result] is not None:

            test_gt_graph = nx.DiGraph()
            test_gt_graph.add_nodes_from(test_candidate_graph_initial.nodes(data=True))
            test_gt_graph = add_gt_edges_to_graph_2(
                groundtruth_graph=test_gt_graph, gt_data=test_array
            )
            test_gt_track_graph = TrackGraph(
                nx_graph=test_gt_graph, frame_attribute="time"
            )

            supervised_data = np.loadtxt(
                supervised_csv_file_names[index_result], delimiter=" "
            )
            if len(supervised_data.shape) == 1:
                supervised_data = [supervised_data]

            for row in supervised_data:
                id_, t = int(row[0]), int(row[1])
                node_id = str(t) + "_" + str(id_)
                out_edges = list(test_track_graph.next_edges[node_id])
                for out_edge in out_edges:
                    test_track_graph.edges[out_edge][
                        EdgeAttr.PINNED.value
                    ] = False  # initially set everything to False.
                out_gt_edges = list(test_gt_track_graph.next_edges[node_id])
                for out_gt_edge in out_gt_edges:
                    if out_gt_edge in test_track_graph.edges:
                        print(f"Pinning gt edge {out_gt_edge}.")
                        test_track_graph.edges[out_gt_edge][
                            EdgeAttr.PINNED.value
                        ] = True
                    else:
                        print(f"gt edge {out_gt_edge} not in test track_graph.")
        # get hyper edge statistics

        hyper_edge_distances = []
        hyper_edge_embedding_distances = []
        for edge in test_track_graph.edges:
            if isinstance(edge[1], tuple):
                (u,) = edge[0]
                (v1, v2) = edge[1]
                u_pos = np.array(test_track_graph.nodes[u][NodeAttr.POS.value])
                v1_pos = np.array(test_track_graph.nodes[v1][NodeAttr.POS.value])
                v2_pos = np.array(test_track_graph.nodes[v2][NodeAttr.POS.value])

                d = np.linalg.norm(u_pos - 0.5 * (v1_pos + v2_pos))
                hyper_edge_distances.append(d)
                if test_edge_embedding_file_names is not None:
                    u_v1_attrackt = test_track_graph.edges[(u, v1)][
                        EdgeAttr.EDGE_EMBEDDING.value
                    ]

                    u_v2_attrackt = test_track_graph.edges[(u, v2)][
                        EdgeAttr.EDGE_EMBEDDING.value
                    ]

                    d2 = np.mean([u_v1_attrackt, u_v2_attrackt])
                    hyper_edge_embedding_distances.append(d2)

        mean_hyper_edge_distance = np.mean(hyper_edge_distances)
        std_hyper_edge_distance = np.std(hyper_edge_distances)

        if test_edge_embedding_file_names is not None:
            mean_hyper_edge_embedding_distance = np.mean(hyper_edge_embedding_distances)
            std_hyper_edge_embedding_distance = np.std(hyper_edge_embedding_distances)

            print(
                f"Mean hyper edge distance is {mean_hyper_edge_distance} and std is {std_hyper_edge_distance}."
            )
            print(
                f"Mean hyper edge embedding distance is {mean_hyper_edge_embedding_distance} and std is {std_hyper_edge_embedding_distance}."
            )
        else:
            mean_hyper_edge_embedding_distance = None
            std_hyper_edge_embedding_distance = None
        test_track_graph = add_app_disapp_attributes(
            test_track_graph, test_t_min, test_t_max
        )

        print("+" * 10)
        print(
            f"Number of nodes in test track graph is {len(test_track_graph.nodes)} and edges is {len(test_track_graph.edges)}."
        )

        recursion_limit = get_recursion_limit(candidate_graph=test_candidate_graph)
        if recursion_limit > 1000:
            sys.setrecursionlimit(recursion_limit)

        # ++++++++
        # Step 2 - apply weights on the test candidate graph
        # ++++++++

        solver = Solver(track_graph=test_track_graph)
        node_embedding_exists = (
            False if test_node_embedding_file_names is None else True
        )
        edge_embedding_exists = (
            False if test_edge_embedding_file_names is None else True
        )
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
            mean_hyper_edge_distance=mean_hyper_edge_distance,
            std_hyper_edge_distance=std_hyper_edge_distance,
            mean_hyper_edge_embedding_distance=mean_hyper_edge_embedding_distance,
            std_hyper_edge_embedding_distance=std_hyper_edge_embedding_distance,
            use_different_weights_hyper=use_different_weights_hyper,
        )
        solver = add_constraints(
            solver=solver, pin_nodes=pin_nodes, pin_edges=pin_edges
        )

        with open(ssvm_weights_file_path, "r") as file:
            data = json.load(file)
            if not use_edge_distance:
                if edge_embedding_exists and not node_embedding_exists:
                    ssvm_weights_array = np.array(
                        [
                            float(data["Attrackt Affinity_weight"]),
                            float(data["Attrackt Affinity_constant"]),
                            float(data["Appear_weight"]),
                            float(data["Appear_constant"]),
                            float(data["Disappear_weight"]),
                            float(data["Disappear_constant"]),
                        ]
                    )
                elif node_embedding_exists and not edge_embedding_exists:
                    ssvm_weights_array = np.array(
                        [
                            float(data["A.E. Embedding Distance_weight"]),
                            float(data["A.E. Embedding Distance_constant"]),
                            float(data["Appear_weight"]),
                            float(data["Appear_constant"]),
                            float(data["Disappear_weight"]),
                            float(data["Disappear_constant"]),
                        ]
                    )
            else:
                if not use_different_weights_hyper:
                    if edge_embedding_exists and not node_embedding_exists:
                        if dT > 1:
                            ssvm_weights_array = np.array(
                                [
                                    float(data["Edge Distance_weight"]),
                                    float(data["Edge Distance_constant"]),
                                    float(data["Time Gap_weight"]),
                                    float(data["Time Gap_constant"]),
                                    float(data["Attrackt Affinity_weight"]),
                                    float(data["Attrackt Affinity_constant"]),
                                    float(data["Appear_weight"]),
                                    float(data["Appear_constant"]),
                                    float(data["Disappear_weight"]),
                                    float(data["Disappear_constant"]),
                                ]
                            )
                        else:
                            ssvm_weights_array = np.array(
                                [
                                    float(data["Edge Distance_weight"]),
                                    float(data["Edge Distance_constant"]),
                                    float(data["Attrackt Affinity_weight"]),
                                    float(data["Attrackt Affinity_constant"]),
                                    float(data["Appear_weight"]),
                                    float(data["Appear_constant"]),
                                    float(data["Disappear_weight"]),
                                    float(data["Disappear_constant"]),
                                ]
                            )
                    elif not edge_embedding_exists and node_embedding_exists:
                        if dT > 1:
                            ssvm_weights_array = np.array(
                                [
                                    float(data["Edge Distance_weight"]),
                                    float(data["Edge Distance_constant"]),
                                    float(data["Time Gap_weight"]),
                                    float(data["Time Gap_constant"]),
                                    float(data["A.E. Embedding Distance_weight"]),
                                    float(data["A.E. Embedding Distance_constant"]),
                                    float(data["Appear_weight"]),
                                    float(data["Appear_constant"]),
                                    float(data["Disappear_weight"]),
                                    float(data["Disappear_constant"]),
                                ]
                            )
                        else:
                            ssvm_weights_array = np.array(
                                [
                                    float(data["Edge Distance_weight"]),
                                    float(data["Edge Distance_constant"]),
                                    float(data["A.E. Embedding Distance_weight"]),
                                    float(data["A.E. Embedding Distance_constant"]),
                                    float(data["Appear_weight"]),
                                    float(data["Appear_constant"]),
                                    float(data["Disappear_weight"]),
                                    float(data["Disappear_constant"]),
                                ]
                            )
                    else:
                        if dT > 1:
                            ssvm_weights_array = np.array(
                                [
                                    float(data["Edge Distance_weight"]),
                                    float(data["Edge Distance_constant"]),
                                    float(data["Time Gap_weight"]),
                                    float(data["Time Gap_constant"]),
                                    float(data["Appear_weight"]),
                                    float(data["Appear_constant"]),
                                    float(data["Disappear_weight"]),
                                    float(data["Disappear_constant"]),
                                ]
                            )
                        else:
                            ssvm_weights_array = np.array(
                                [
                                    float(data["Edge Distance_weight"]),
                                    float(data["Edge Distance_constant"]),
                                    float(data["Appear_weight"]),
                                    float(data["Appear_constant"]),
                                    float(data["Disappear_weight"]),
                                    float(data["Disappear_constant"]),
                                ]
                            )
                else:
                    if edge_embedding_exists and not node_embedding_exists:
                        if dT > 1:
                            ssvm_weights_array = np.array(
                                [
                                    float(data["Edge Distance Regular_weight"]),
                                    float(data["Edge Distance Regular_constant"]),
                                    float(data["Edge Distance Hyper_weight"]),
                                    float(data["Edge Distance Hyper_constant"]),
                                    float(data["Time Gap_weight"]),
                                    float(data["Time Gap_constant"]),
                                    float(data["Attrackt Affinity Regular_weight"]),
                                    float(data["Attrackt Affinity Regular_constant"]),
                                    float(data["Attrackt Affinity Hyper_weight"]),
                                    float(data["Attrackt Affinity Hyper_constant"]),
                                    float(data["Appear_weight"]),
                                    float(data["Appear_constant"]),
                                    float(data["Disappear_weight"]),
                                    float(data["Disappear_constant"]),
                                ]
                            )
                        else:
                            ssvm_weights_array = np.array(
                                [
                                    float(data["Edge Distance Regular_weight"]),
                                    float(data["Edge Distance Regular_constant"]),
                                    float(data["Edge Distance Hyper_weight"]),
                                    float(data["Edge Distance Hyper_constant"]),
                                    float(data["Attrackt Affinity Regular_weight"]),
                                    float(data["Attrackt Affinity Regular_constant"]),
                                    float(data["Attrackt Affinity Hyper_weight"]),
                                    float(data["Attrackt Affinity Hyper_constant"]),
                                    float(data["Appear_weight"]),
                                    float(data["Appear_constant"]),
                                    float(data["Disappear_weight"]),
                                    float(data["Disappear_constant"]),
                                ]
                            )
                    elif not edge_embedding_exists and node_embedding_exists:
                        if dT > 1:
                            ssvm_weights_array = np.array(
                                [
                                    float(data["Edge Distance Regular_weight"]),
                                    float(data["Edge Distance Regular_constant"]),
                                    float(data["Edge Distance Hyper_weight"]),
                                    float(data["Edge Distance Hyper_constant"]),
                                    float(data["Time Gap_weight"]),
                                    float(data["Time Gap_constant"]),
                                    float(
                                        data["A.E. Embedding Distance Regular_weight"]
                                    ),
                                    float(
                                        data["A.E. Embedding Distance Regular_constant"]
                                    ),
                                    float(data["A.E. Embedding Distance Hyper_weight"]),
                                    float(
                                        data["A.E. Embedding Distance Hyper_constant"]
                                    ),
                                    float(data["Appear_weight"]),
                                    float(data["Appear_constant"]),
                                    float(data["Disappear_weight"]),
                                    float(data["Disappear_constant"]),
                                ]
                            )
                        else:
                            ssvm_weights_array = np.array(
                                [
                                    float(data["Edge Distance Regular_weight"]),
                                    float(data["Edge Distance Regular_constant"]),
                                    float(data["Edge Distance Hyper_weight"]),
                                    float(data["Edge Distance Hyper_constant"]),
                                    float(
                                        data["A.E. Embedding Distance Regular_weight"]
                                    ),
                                    float(
                                        data["A.E. Embedding Distance Regular_constant"]
                                    ),
                                    float(data["A.E. Embedding Distance Hyper_weight"]),
                                    float(
                                        data["A.E. Embedding Distance Hyper_constant"]
                                    ),
                                    float(data["Appear_weight"]),
                                    float(data["Appear_constant"]),
                                    float(data["Disappear_weight"]),
                                    float(data["Disappear_constant"]),
                                ]
                            )
                    else:
                        if dT > 1:
                            ssvm_weights_array = np.array(
                                [
                                    float(data["Edge Distance Regular_weight"]),
                                    float(data["Edge Distance Regular_constant"]),
                                    float(data["Edge Distance Hyper_weight"]),
                                    float(data["Edge Distance Hyper_constant"]),
                                    float(data["Time Gap_weight"]),
                                    float(data["Time Gap_constant"]),
                                    float(data["Appear_weight"]),
                                    float(data["Appear_constant"]),
                                    float(data["Disappear_weight"]),
                                    float(data["Disappear_constant"]),
                                ]
                            )
                        else:
                            ssvm_weights_array = np.array(
                                [
                                    float(data["Edge Distance Regular_weight"]),
                                    float(data["Edge Distance Regular_constant"]),
                                    float(data["Edge Distance Hyper_weight"]),
                                    float(data["Edge Distance Hyper_constant"]),
                                    float(data["Appear_weight"]),
                                    float(data["Appear_constant"]),
                                    float(data["Disappear_weight"]),
                                    float(data["Disappear_constant"]),
                                ]
                            )

        solver.weights.from_ndarray(ssvm_weights_array)
        solution = solver.solve(verbose=True)
        solution_graph = solver.get_selected_subgraph(solution)

        save_ilp_result(solution_graph, results_dir_name)

        print("+" * 10)
        print(
            f"After optimization, we selected {len(solution_graph.nodes)} nodes and {len(solution_graph.edges)} edges."
        )

        # ++++++++
        # Step 3 - traccuracy numbers
        # ++++++++

        # gt_segmentation = np.zeros(
        #    (test_t_max + 1, *tuple(test_image_shape)), dtype=np.uint64
        # )
        # for node, attrs in test_candidate_graph_initial.nodes.items():
        #    t, id_ = node.split("_")
        #    t, id_ = int(t), int(id_)
        #    position = attrs[NodeAttr.POS.value]
        #    gt_segmentation[t] = expand_position(
        #        data=gt_segmentation[t], position=position, id_=id_
        #    )

        new_mapping, res_track, tracked_masks, tracked_graph = save_result(
            solution_nx_graph=graph_to_nx(solution_graph),
            segmentation_shape=None,  # gt_segmentation.shape,
            output_tif_dir_name=results_dir_name,
            write_tifs=write_tifs,
        )

        tracked_array = []
        solution_nx_graph = graph_to_nx(solution_graph)
        for edge in solution_nx_graph.edges:
            u, v = edge
            t_u, id_u = u.split("_")
            t_v, id_v = v.split("_")
            t_u, id_u, t_v, id_v = int(t_u), int(id_u), int(t_v), int(id_v)

            # check if track starts at u
            in_edges = solution_nx_graph.in_edges(u)
            if len(in_edges) == 0:
                tracked_array.append(
                    [
                        id_u,
                        t_u,
                        test_candidate_graph.nodes[u]["pos"][0],
                        test_candidate_graph.nodes[u]["pos"][1],
                        0,
                    ]
                )

            tracked_array.append(
                [
                    id_v,
                    t_v,
                    test_candidate_graph.nodes[v]["pos"][0],
                    test_candidate_graph.nodes[v]["pos"][1],
                    id_u,
                ]
            )

        tracked_array = np.asarray(tracked_array)
        np.savetxt(
            fname=results_dir_name + "/track-detections.csv",
            X=tracked_array,
            fmt=["%i", "%i", "%.3f", "%.3f", "%i"],
        )

        test_gt_graph = nx.DiGraph()
        test_gt_graph.add_nodes_from(test_candidate_graph_initial.nodes(data=True))
        test_gt_graph = add_gt_edges_to_graph_2(
            groundtruth_graph=test_gt_graph, gt_data=test_array
        )

        # convert to track graph
        test_gt_track_graph = TrackGraph(nx_graph=test_gt_graph, frame_attribute="time")
        print(
            f"Number of nodes in the groundtruth test dataset is {len(test_gt_track_graph.nodes)} and edges is {len(test_gt_track_graph.edges)}"
        )

        for node in test_gt_track_graph.nodes:
            pos = test_gt_track_graph.nodes[node]["pos"]
            if len(pos) == 2:
                y, x = pos
                test_gt_track_graph.nodes[node]["y"] = y
                test_gt_track_graph.nodes[node]["x"] = x
            elif len(pos) == 3:
                z, y, x = pos
                test_gt_track_graph.nodes[node]["z"] = int(z)
                test_gt_track_graph.nodes[node]["y"] = int(y)
                test_gt_track_graph.nodes[node]["x"] = int(x)

        compute_metrics(
            gt_segmentation=None,  # gt_segmentation,
            gt_nx_graph=graph_to_nx(test_gt_track_graph),
            predicted_segmentation=None,  # tracked_masks,
            pred_nx_graph=tracked_graph,
            results_dir_name=results_dir_name,
        )


if __name__ == "__main__":
    parser = jsonargparse.ArgumentParser()
    parser.add_argument("--yaml_config_file_name", dest="yaml_config_file_name")
    args = parser.parse_args()
    infer(yaml_config_file_name=args.yaml_config_file_name)
