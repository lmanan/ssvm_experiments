import numpy as np
from utils import (
    add_gt_edges_to_graph_2,
    load_csv_data,
    flip_edges,
    add_hyper_edges,
    add_costs,
    add_constraints,
    expand_position,
)
from motile_toolbox.candidate_graph import (
    get_candidate_graph_from_points_list,
    graph_to_nx,
    NodeAttr,
)
from motile import TrackGraph, Solver
import jsonargparse
import pprint
import os
from saving_utils import save_result
from run_traccuracy import compute_metrics
import json
import logging
import networkx as nx

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s %(name)s %(levelname)-8s %(message)s"
)
logger = logging.getLogger(__name__)

pp = pprint.PrettyPrinter(indent=4)


def track(
    train_segmentation_dir_name: str | None,
    val_segmentation_dir_name: str | None,
    train_csv_file_name: str | None,
    val_csv_file_name: str | None,
    direction_candidate_graph: str,
    results_dir_name: str,
    dT: int,
    num_nearest_neighbours: int | None,
    max_edge_distance: float | None,
    val_image_shape: tuple,
    pin_nodes: bool,
    train_node_embedding_file_name: str | None = None,
    val_node_embedding_file_name: str | None = None,
    train_edge_embedding_file_name: str | None = None,
    val_edge_embedding_file_name: str | None = None,
    ssvm_weights_array: np.ndarray | None = None,
    regularizer_weight: float = 100.0,
):

    assert direction_candidate_graph in ["forward", "backward"]

    # ++++++++
    # Step 1 - build `train` and `val` candidate graphs
    # ++++++++

    print("\nBuilding train and val track graph ...\n")

    train_array = load_csv_data(csv_file_name=train_csv_file_name)
    val_array = load_csv_data(csv_file_name=val_csv_file_name)

    # just considering 100 time points (0-99) for now
    # val_array = val_array[:7773, :]
    val_t_min = int(np.min(val_array[:, 1]))
    val_t_max = int(np.max(val_array[:, 1]))
    print(f"Min time point is {val_t_min}, Max time point is {val_t_max}")
    val_num_frames = val_t_max - val_t_min + 1

    print(
        f"train array has shape {train_array.shape}, val array has shape {val_array.shape}"
    )

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
        train_candidate_graph_initial = flip_edges(train_candidate_graph_initial)
        val_candidate_graph_initial = flip_edges(val_candidate_graph_initial)

    # add hyper edges
    train_candidate_graph = add_hyper_edges(
        candidate_graph=train_candidate_graph_initial
    )
    val_candidate_graph = add_hyper_edges(candidate_graph=val_candidate_graph_initial)

    # make track graph
    train_track_graph = TrackGraph(
        nx_graph=train_candidate_graph, frame_attribute="time"
    )
    val_track_graph = TrackGraph(nx_graph=val_candidate_graph, frame_attribute="time")

    print(
        f"Number of nodes in train graph is {len(train_track_graph.nodes)} and edges is {len(train_track_graph.edges)}"
    )
    print(
        f"Number of nodes in val graph is {len(val_track_graph.nodes)} and edges is {len(val_track_graph.edges)}"
    )

    # ++++++++
    # Step 2 - build `gt` graph
    # ++++++++
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
        node_embedding_exists=node_embedding_exists,
        edge_embedding_exists=edge_embedding_exists,
    )
    solver = add_constraints(solver=solver, pin_nodes = pin_nodes)
    solver.weights.from_ndarray(ssvm_weights_array)

    solution = solver.solve(verbose=True)
    solution_graph = solver.get_selected_subgraph(solution)

    print(
        f"After optimization, we selected {len(solution_graph.nodes)} nodes and {len(solution_graph.edges)} edges"
    )

    # since val_segmentation is not available, as csvs are available.

    val_segmentation = np.zeros((val_num_frames, *tuple(val_image_shape)), dtype=np.int64)  # TODO
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
        write_tifs=False,
    )

    print("Computing scores ...")

    val_gt_graph = nx.DiGraph()
    val_gt_graph.add_nodes_from(val_candidate_graph_initial.nodes(data=True))
    val_gt_graph = add_gt_edges_to_graph_2(
        groundtruth_graph=val_gt_graph, gt_data=val_array
    )

    # convert to track graph
    val_gt_track_graph = TrackGraph(nx_graph=val_gt_graph, frame_attribute="time")

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
    parser = jsonargparse.ArgumentParser()
    parser.add_argument(
        "--train_segmentation_dir_name",
        dest="train_segmentation_dir_name",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--val_segmentation_dir_name",
        dest="val_segmentation_dir_name",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--train_csv_file_name", dest="train_csv_file_name", type=str, default=None
    )

    parser.add_argument(
        "--val_csv_file_name", dest="val_csv_file_name", type=str, default=None
    )

    parser.add_argument(
        "--num_nearest_neighbours", dest="num_nearest_neighbours", type=int
    )

    parser.add_argument(
        "--max_edge_distance", dest="max_edge_distance", default=None, type=float
    )

    parser.add_argument(
        "--direction_candidate_graph",
        dest="direction_candidate_graph",
        default="backward",
        type=str,
    )
    parser.add_argument("--dT", dest="dT", default=1, type=int)
    parser.add_argument(
        "--train_node_embedding_file_name",
        dest="train_node_embedding_file_name",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--val_node_embedding_file_name",
        dest="val_node_embedding_file_name",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--train_edge_embedding_file_name",
        dest="train_edge_embedding_file_name",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--val_edge_embedding_file_name",
        dest="val_edge_embedding_file_name",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--results_dir_name", dest="results_dir_name", default="results/", type=str
    )

    parser.add_argument(
        "--regularizer_weight", dest="regularizer_weight", type=float, default=100.0
    )

    parser.add_argument("--pin_nodes", dest="pin_nodes", type=bool)
    parser.add_argument("--val_image_shape", dest="val_image_shape", nargs="+", type=int) 
    print("+" * 10)
    args = parser.parse_args()
    pp.pprint(args)
    print("+" * 10)

    if os.path.exists(args.results_dir_name + "/jsons/"):
        pass
    else:
        os.makedirs(args.results_dir_name + "/jsons/")

    track(
        train_segmentation_dir_name=args.train_segmentation_dir_name,
        val_segmentation_dir_name=args.val_segmentation_dir_name,
        train_csv_file_name=args.train_csv_file_name,
        val_csv_file_name=args.val_csv_file_name,
        num_nearest_neighbours=args.num_nearest_neighbours,
        max_edge_distance=args.max_edge_distance,
        results_dir_name=args.results_dir_name,
        direction_candidate_graph=args.direction_candidate_graph,
        dT=args.dT,
        train_node_embedding_file_name=args.train_node_embedding_file_name,
        val_node_embedding_file_name=args.val_node_embedding_file_name,
        train_edge_embedding_file_name=args.train_edge_embedding_file_name,
        val_edge_embedding_file_name=args.val_edge_embedding_file_name,
        regularizer_weight=args.regularizer_weight,
        val_image_shape = args.val_image_shape,
        pin_nodes = args.pin_nodes
    )
