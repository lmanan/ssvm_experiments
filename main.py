import tifffile
import numpy as np
from glob import glob
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
import subprocess
from saving_utils import save_result_tifs_res_track

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(name)s %(levelname)-8s %(message)s"
)
logger = logging.getLogger(__name__)


def add_costs_and_constraints(solver):
    solver.add_constraints(MaxParents(1))
    solver.add_constraints(MaxChildren(2))
    solver.add_costs(
        EdgeDistance(
            weight=10.0, constant=-10.0, position_attribute=NodeAttr.POS.value
        ),
        name="Position",
    )
    solver.add_costs(Split(constant=0.5), name="Division")
    solver.add_costs(Appear(constant=0.6))
    solver.add_costs(Disappear(constant=0.6))
    return solver


def track(
    train_segmentation_dir_name: str,
    val_segmentation_dir_name: str,
    max_edge_distance: float,
    regularizer_weight: float,
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
    """
    # Step 1 ------------------------------

    # obtain train masks
    train_filenames = natsorted(glob(train_segmentation_dir_name + "/*.tif"))
    train_segmentation = []
    for train_filename in train_filenames:
        train_segmentation.append(tifffile.imread(train_filename))
    train_segmentation = np.asarray(train_segmentation)
    train_segmentation = train_segmentation[
        :, np.newaxis
    ]  # requires a hypothesis channel

    # obtain val masks
    val_filenames = natsorted(glob(val_segmentation_dir_name + "/*.tif"))
    val_segmentation = []
    for val_filename in val_filenames:
        val_segmentation.append(tifffile.imread(val_filename))
    val_segmentation = np.asarray(val_segmentation)
    val_segmentation = val_segmentation[:, np.newaxis]  # requires a hypothesis channel

    # get train candidate graph
    train_candidate_graph, _ = get_candidate_graph(
        segmentation=train_segmentation, max_edge_distance=max_edge_distance, iou=False
    )

    # get val candidate graph
    val_candidate_graph, _ = get_candidate_graph(
        segmentation=val_segmentation, max_edge_distance=max_edge_distance, iou=False
    )

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
    gt_data = np.loadtxt(train_segmentation_dir_name + "/man_track.txt", delimiter=" ")

    gt_dict = {}
    for row in gt_data:
        id_, t_st, t_end, parent_id = int(row[0]), int(row[1]), int(row[2]), int(row[3])
        gt_dict[id_] = [t_st, t_end]

    for row in gt_data:
        id_, t_st, t_end, parent_id = int(row[0]), int(row[1]), int(row[2]), int(row[3])

        for t in range(t_st, t_end - 1):
            groundtruth_graph.add_edge(
                str(t) + "_" + str(id_), str(t + 1) + "_" + str(id_)
            )

        if parent_id != 0:
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
    solver = add_costs_and_constraints(solver)
    solver.fit_weights(
        gt_attribute="gt", regularizer_weight=regularizer_weight, max_iterations=1000
    )
    optimal_weights = solver.weights
    print(f"After fitting, optimal weights are {optimal_weights}")

    # Step 4 ---------------------------------

    solver = Solver(track_graph=val_track_graph)
    solver = add_costs_and_constraints(solver)
    solver.weights.from_ndarray(optimal_weights.to_ndarray())

    solution = solver.solve(verbose=True)
    solution_graph = solver.get_selected_subgraph(solution)

    new_mapping, res_track, new_segmentations = save_result_tifs_res_track(
        graph_to_nx(solution_graph),
        val_segmentation[:, 0],
        output_tif_dir="01_RES_SSVM",
    )
    print("Computing scores ...")
    subprocess.call(["python ./run_traccuracy.py"], shell=True)


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
    args = parser.parse_args()
    print("+" * 10)
    print(args)
    print("+" * 10)
    track(
        train_segmentation_dir_name=args.train_segmentation_dir_name,
        val_segmentation_dir_name=args.val_segmentation_dir_name,
        max_edge_distance=args.max_edge_distance,
        regularizer_weight=args.regularizer_weight,
    )
