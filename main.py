import numpy as np
from utils import (
    add_gt_edges_to_graph,
    load_data,
    flip_edges,
    add_hyper_edges,
    add_costs,
    add_constraints,
)
from motile_toolbox.candidate_graph import (
    get_candidate_graph,
    nodes_from_segmentation,
    graph_to_nx,
)
from motile import TrackGraph, Solver
import jsonargparse
import pprint
import os
from saving_utils import save_result
from run_traccuracy import compute_metrics
import json
import logging

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s %(name)s %(levelname)-8s %(message)s"
)
logger = logging.getLogger(__name__)

pp = pprint.PrettyPrinter(indent=4)


def track(
    train_segmentation_dir_name: str,
    val_segmentation_dir_name: str,
    direction_candidate_graph: str,
    results_dir_name: str,
    dT: int,
    num_nearest_neighbours: int | None,
    max_edge_distance: float | None,
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
    train_segmentation = load_data(segmentation_dir_name=train_segmentation_dir_name)
    val_segmentation = load_data(segmentation_dir_name=val_segmentation_dir_name)

    # get `train_candidate_graph`
    train_candidate_graph, _ = get_candidate_graph(
        segmentation=train_segmentation,
        num_nearest_neighbours=num_nearest_neighbours,
        max_edge_distance=max_edge_distance,
        direction_candidate_graph=direction_candidate_graph,
    )

    # get `train_candidate_graph`
    val_candidate_graph, _ = get_candidate_graph(
        segmentation=val_segmentation,
        num_nearest_neighbours=num_nearest_neighbours,
        max_edge_distance=max_edge_distance,
        direction_candidate_graph=direction_candidate_graph,
    )

    if direction_candidate_graph == "backward":
        train_candidate_graph = flip_edges(train_candidate_graph)
        val_candidate_graph = flip_edges(val_candidate_graph)

    # add hyper edges
    train_candidate_graph = add_hyper_edges(candidate_graph=train_candidate_graph)
    val_candidate_graph = add_hyper_edges(candidate_graph=val_candidate_graph)

    # make track graph
    train_track_graph = TrackGraph(
        nx_graph=train_candidate_graph, frame_attribute="time"
    )
    val_track_graph = TrackGraph(nx_graph=val_candidate_graph, frame_attribute="time")

    # ++++++++
    # Step 2 - build `gt` graph
    # ++++++++
    print("\nBuilding GT track graph ...\n")

    groundtruth_graph, node_frame_dict = nodes_from_segmentation(train_segmentation)

    # add edges
    groundtruth_graph = add_gt_edges_to_graph(
        groundtruth_graph=groundtruth_graph,
        json_file_name=train_segmentation_dir_name + "/man_track.json",
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
        solver = add_constraints(solver=solver)
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
    solver = add_constraints(solver=solver)
    solver.weights.from_ndarray(ssvm_weights_array)

    solution = solver.solve(verbose=True)
    solution_graph = solver.get_selected_subgraph(solution)

    new_mapping, res_track, tracked_masks = save_result(
        solution_nx_graph=graph_to_nx(solution_graph),
        segmentation=val_segmentation[:, 0],
        output_tif_dir_name=results_dir_name,
    )

    print("Computing scores ...")
    compute_metrics(
        gt_json_file_name=val_segmentation_dir_name + "/man_track.json",
        gt_segmentation=val_segmentation[:, 0],
        predicted_json_file_name=results_dir_name + "/jsons/res_track.json",
        predicted_segmentation=tracked_masks,
        results_dir_name=results_dir_name,
    )


if __name__ == "__main__":
    parser = jsonargparse.ArgumentParser()
    parser.add_argument(
        "--train_segmentation_dir_name", dest="train_segmentation_dir_name", type=str
    )
    parser.add_argument(
        "--val_segmentation_dir_name", dest="val_segmentation_dir_name", type=str
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
        num_nearest_neighbours=args.num_nearest_neighbours,
        max_edge_distance=args.max_edge_distance,
        results_dir_name=args.results_dir_name,
        direction_candidate_graph=args.direction_candidate_graph,
        dT=args.dT,
        train_node_embedding_file_name=args.train_node_embedding_file_name,
        val_node_embedding_file_name=args.val_node_embedding_file_name,
        train_edge_embedding_file_name=args.train_edge_embedding_file_name,
        val_edge_embedding_file_name=args.val_edge_embedding_file_name,
        ssvm_weights_array=None,
        regularizer_weight=args.regularizer_weight,
    )
