import tifffile
import numpy as np
from natsort import natsorted
import argparse
from motile_toolbox.candidate_graph import (
    get_candidate_graph,
    nodes_from_segmentation,
    NodeAttr,
    graph_to_nx,
)
from motile.track_graph import TrackGraph
from motile.solver import Solver
from motile.constraints import MaxParents, MaxChildren
from motile.costs import Appear, Disappear, Split, EdgeDistance
import logging
from saving_utils import save_result_tifs_res_track
from pathlib import Path
from run_traccuracy import compute_metrics
from division_costs import NormalEdgeDistance, HyperEdgeDistance
from itertools import combinations
import sys
from datetime import date


logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s %(name)s %(levelname)-8s %(message)s"
)
logger = logging.getLogger(__name__)


def add_hyper_elements(candidate_graph):
    nodes_original = list(candidate_graph.nodes)
    for node in nodes_original:
        out_edges = candidate_graph.out_edges(node)
        pairs = list(combinations(out_edges, 2))
        for pair in pairs:
            candidate_graph.add_node(
                str(pair[0][0]) + "_" + str(pair[0][1]) + "_" + str(pair[1][1])
            )
            candidate_graph.add_edge(
                pair[0][0],
                str(pair[0][0] + "_" + str(pair[0][1]) + "_" + str(pair[1][1])),
            )
            candidate_graph.add_edge(
                str(pair[0][0]) + "_" + str(pair[0][1]) + "_" + str(pair[1][1]),
                pair[0][1],
            )
            candidate_graph.add_edge(
                str(pair[0][0]) + "_" + str(pair[0][1]) + "_" + str(pair[1][1]),
                pair[1][1],
            )
    return candidate_graph


def add_costs_and_constraints(solver, symmetric_division_cost):
    solver.add_constraints(MaxParents(1))
    if symmetric_division_cost:
        solver.add_constraints(MaxChildren(1))
    else:
        solver.add_constraints(MaxChildren(2))

    if symmetric_division_cost:
        solver.add_costs(
            NormalEdgeDistance(
                weight=1.0,
                constant=-20.0,
                position_attribute=NodeAttr.POS.value,
            ),
            name="Normal Edge Distance",
        )
        solver.add_costs(
            HyperEdgeDistance(
                weight=1.0, constant=0.0, position_attribute=NodeAttr.POS.value
            ),
            name="Hyper Edge Distance",
        )
    else:
        solver.add_costs(
            EdgeDistance(
                weight=1.0, constant=-20.0, position_attribute=NodeAttr.POS.value
            ),
            name="Position",
        )
        solver.add_costs(Split(constant=0.5), name="Division")

    solver.add_costs(Appear(constant=0.6, ignore_attribute="ignore_appear_cost"))
    solver.add_costs(Disappear(constant=0.6, ignore_attribute="ignore_disappear_cost"))
    return solver


def track(
    train_segmentation_dir_name: str,
    val_segmentation_dir_name: str,
    max_edge_distance: float,
    regularizer_weight: float,
    symmetric_division_cost: bool,
):
    """
    This function does four things:

    Step 1: First it sets up a `train_candidate_graph` and a
    `val_candidate_graph` using the available GT detections
    and by connecting neighbors within `max_edge_distance` radius.

    Step 2: Next, the actual `groundtruth graph` and the
    `groundtruth_track_graph`  is created using the `man_track.txt`
    text file provided by CTC website for this dataset (available within the
    `track_segmentation_dir_name` directory).

    Step 3: A fraction of the nodes are randomly sampled and the outgoing edges
    from these nodes are either specified to be `True` if they are indeed the
    ground truth edge or `False` if they are a candidate, non ground truth edge.
    Next, we try to identify the best weights on the train dataset using SSVM.

    Step 4: Using the obtained weights, the solver is solved over the
    val_track_graph. A call is made to run_traccuracy.py and then numbers are
    computed.

    Parameters
    ----------
    train_segmentation_dir_name : str
        train_segmentation_dir_name is the path to the `TRA` directory containing all
        tif images used for fitting weights.
    val_segmentation_dir_name : str
        val_segmentation_dir_name is the path to the `TRA` directory containing all
        tif images used on which results are needed.
    max_edge_distance : float
        max_edge_distance is used to connect nodes that lie at an L2 distance
        lesser than max_edge_distance while constructing a candidate graph
        (see Step 1 above).
    regularizer_weight: float

    symmetric_division_cost: bool, default=False
        If this is set to `True`, hyper-edges are used to connect a parent node
        with two daughter nodes.
        This leads to a slightly different set of constraints.
        Additionally, the cost for division is different.
    """

    today = date.today()
    sys.stdout = open(val_segmentation_dir_name + "_" + str(today), "w")
    # Step 1 ------------------------------

    # obtain train masks
    train_filenames = natsorted(list(Path(train_segmentation_dir_name).glob("*.tif")))
    train_segmentation = []
    for train_filename in train_filenames:
        train_segmentation.append(tifffile.imread(train_filename))
    train_segmentation = np.asarray(train_segmentation)
    train_segmentation = train_segmentation[
        :, np.newaxis
    ]  # requires a hypothesis channel

    # obtain val masks
    val_filenames = natsorted(list(Path(val_segmentation_dir_name).glob("*.tif")))
    val_segmentation = []
    for val_filename in val_filenames:
        val_segmentation.append(tifffile.imread(val_filename))
    val_segmentation = np.asarray(val_segmentation)
    val_segmentation = val_segmentation[:, np.newaxis]  # requires a hypothesis channel

    # get train candidate graph
    train_candidate_graph, _ = get_candidate_graph(
        segmentation=train_segmentation, max_edge_distance=max_edge_distance, iou=False
    )
    if symmetric_division_cost:
        print(" Adding hyper elements to train candidate graph ...")
        train_candidate_graph = add_hyper_elements(train_candidate_graph)

    # get val candidate graph
    val_candidate_graph, _ = get_candidate_graph(
        segmentation=val_segmentation, max_edge_distance=max_edge_distance, iou=False
    )

    if symmetric_division_cost:
        print(" Adding hyper elements to val candidate graph ...")
        val_candidate_graph = add_hyper_elements(val_candidate_graph)

    # build a train track graph
    train_track_graph = TrackGraph(
        nx_graph=train_candidate_graph, frame_attribute="time"
    )

    # build a val track graph
    val_track_graph = TrackGraph(nx_graph=val_candidate_graph, frame_attribute="time")

    # Step 2 ------------------------------

    # build ground truth track graph

    # add nodes
    groundtruth_graph, node_frame_dict = nodes_from_segmentation(train_segmentation)

    # add edges
    gt_data = np.loadtxt(
        Path(train_segmentation_dir_name).joinpath("man_track.txt"), delimiter=" "
    )
    parent_daughter_dict = {}
    gt_dict = {}

    for row in gt_data:
        id_, t_st, t_end, parent_id = int(row[0]), int(row[1]), int(row[2]), int(row[3])
        gt_dict[id_] = [t_st, t_end]

    for row in gt_data:
        id_, t_st, t_end, parent_id = int(row[0]), int(row[1]), int(row[2]), int(row[3])
        if parent_id in parent_daughter_dict.keys():
            pass
        else:
            parent_daughter_dict[parent_id] = []
        parent_daughter_dict[parent_id].append(id_)

    for row in gt_data:
        id_, t_st, t_end, parent_id = int(row[0]), int(row[1]), int(row[2]), int(row[3])

        for t in range(t_st, t_end):
            groundtruth_graph.add_edge(
                str(t) + "_" + str(id_), str(t + 1) + "_" + str(id_)
            )

        if parent_id != 0:
            time_parent = gt_dict[parent_id][1]
            temp_node = str(time_parent) + "_" + str(parent_id)
            for daughter_index in range(len(parent_daughter_dict[parent_id])):
                temp_node += (
                    "_"
                    + str(t_st)
                    + "_"
                    + str(parent_daughter_dict[parent_id][daughter_index])
                )
            if symmetric_division_cost:
                groundtruth_graph.add_node(temp_node)
                groundtruth_graph.add_edge(
                    str(gt_dict[parent_id][1]) + "_" + str(parent_id), temp_node
                )
                for daughter_index in range(len(parent_daughter_dict[parent_id])):
                    groundtruth_graph.add_edge(
                        temp_node,
                        str(t_st)
                        + "_"
                        + str(parent_daughter_dict[parent_id][daughter_index]),
                    )
            else:
                groundtruth_graph.add_edge(
                    str(gt_dict[parent_id][1]) + "_" + str(parent_id),
                    str(t_st) + "_" + str(id_),
                )

    # build a groundtruth track graph

    groundtruth_track_graph = TrackGraph(
        nx_graph=groundtruth_graph, frame_attribute="time"
    )

    # Step 3 ---------------------------------

    ## fitting weights now ...

    np.random.seed(42)

    for node_id in train_track_graph.nodes:
        for edge_id in list(train_track_graph.next_edges[node_id]):
            if edge_id in groundtruth_track_graph.edges:
                train_track_graph.edges[edge_id]["gt"] = True
            else:
                train_track_graph.edges[edge_id]["gt"] = False

    # fit weights
    solver = Solver(track_graph=train_track_graph)
    solver = add_costs_and_constraints(solver, symmetric_division_cost)
    solver.fit_weights(
        gt_attribute="gt", regularizer_weight=regularizer_weight, max_iterations=1000
    )
    optimal_weights = solver.weights
    print(f"After fitting, optimal weights are {optimal_weights}")

    # Step 4 ---------------------------------

    solver = Solver(track_graph=val_track_graph)
    solver = add_costs_and_constraints(solver, symmetric_division_cost)
    solver.weights.from_ndarray(optimal_weights.to_ndarray())

    # save weights to a json

    solution = solver.solve(verbose=True)
    solution_graph = solver.get_selected_subgraph(solution)

    new_mapping, res_track, new_segmentations = save_result_tifs_res_track(
        graph_to_nx(solution_graph),
        val_segmentation[:, 0],
        output_tif_dir="01_RES_SSVM",
    )
    print("Computing scores ...")
    compute_metrics(val_segmentation_dir_name=val_segmentation_dir_name)
    sys.stdout.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_segmentation_dir_name",
        dest="train_segmentation_dir_name",
        default="./Fluo-N2DL-HeLa/02_GT/TRA/",
    )
    parser.add_argument(
        "--val_segmentation_dir_name",
        dest="val_segmentation_dir_name",
        default="./Fluo-N2DL-HeLa/01_GT/TRA/",
    )
    parser.add_argument("--max_edge_distance", dest="max_edge_distance", default=50)
    parser.add_argument(
        "--regularizer_weight", dest="regularizer_weight", default=1e2, type=float
    )
    parser.add_argument(
        "--symmetric_division_cost",
        dest="symmetric_division_cost",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    args = parser.parse_args()
    print("+" * 10)
    print(args)
    print("+" * 10)
    track(
        train_segmentation_dir_name=args.train_segmentation_dir_name,
        val_segmentation_dir_name=args.val_segmentation_dir_name,
        max_edge_distance=args.max_edge_distance,
        regularizer_weight=args.regularizer_weight,
        symmetric_division_cost=args.symmetric_division_cost,
    )
