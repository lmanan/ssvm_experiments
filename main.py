import tifffile
import numpy as np
from natsort import natsorted
from typing import cast
import jsonargparse
from motile_toolbox.candidate_graph import (
    get_candidate_graph,
    nodes_from_segmentation,
    NodeAttr,
    graph_to_nx,
)
from motile.variables import EdgeSelected
from motile.track_graph import TrackGraph
from motile.solver import Solver
from motile.constraints import MaxParents, MaxChildren
from motile.costs import Costs, Weight, Appear, Disappear, Split
import logging
from pathlib import Path
from division_costs import EdgeDistance
from itertools import combinations
import json
import networkx as nx
from run_traccuracy import compute_metrics
from saving_utils import save_result_tifs_json
import os

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s %(name)s %(levelname)-8s %(message)s"
)
logger = logging.getLogger(__name__)


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


def add_hyper_elements(candidate_graph):
    nodes_original = list(candidate_graph.nodes)
    for node in nodes_original:
        out_edges = candidate_graph.out_edges(node)
        pairs = list(combinations(out_edges, 2))
        for pair in pairs:
            temp_node = str(pair[0][0]) + "_" + str(pair[0][1]) + "_" + str(pair[1][1])
            candidate_graph.add_node(temp_node)
            candidate_graph.add_edge(pair[0][0], temp_node)
            candidate_graph.add_edge(
                temp_node,
                pair[0][1],
            )
            candidate_graph.add_edge(
                temp_node,
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
            EdgeDistance(
                weight=1.0,
                constant=-20.0,
                position_attribute=NodeAttr.POS.value,
            ),
            name="Edge Distance",
        )

        solver.add_costs(
            TimeGap(weight=1.0, constant=0.0, time_attribute=NodeAttr.TIME.value),
            name="Time Gap",
        )
    else:
        solver.add_costs(
            EdgeDistance(
                weight=1.0, constant=-20.0, position_attribute=NodeAttr.POS.value
            ),
            name="Edge Distance",
        )
        solver.add_costs(
            TimeGap(weight=1.0, constant=0.0, time_attribute=NodeAttr.TIME.value),
            name="Time Gap",
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
    dT: int,
    results_dir: str,
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

    # Step 1 ------------------------------

    # obtain train masks
    train_filenames = natsorted(list(Path(train_segmentation_dir_name).glob("*.tif")))
    train_segmentation = []
    for train_filename in train_filenames:
        ma = tifffile.imread(train_filename)
        train_segmentation.append(ma)
    train_segmentation = np.asarray(train_segmentation)
    train_segmentation = train_segmentation[
        :, np.newaxis
    ]  # requires a hypothesis channel
    train_segmentation_short = train_segmentation[:46]

    # obtain val masks
    val_filenames = natsorted(list(Path(val_segmentation_dir_name).glob("*.tif")))
    val_segmentation = []
    for val_filename in val_filenames:
        ma = tifffile.imread(val_filename)
        val_segmentation.append(ma)
    val_segmentation = np.asarray(val_segmentation)
    val_segmentation = val_segmentation[:, np.newaxis]  # requires a hypothesis channel

    # get train candidate graph
    train_candidate_graph, _ = get_candidate_graph(
        segmentation=train_segmentation_short,  # TODO
        max_edge_distance=max_edge_distance,
        iou=False,
        dT=dT,
    )

    if symmetric_division_cost:
        print(" Adding hyper elements to train candidate graph ...")
        train_candidate_graph = add_hyper_elements(train_candidate_graph)

    # get val candidate graph
    val_candidate_graph, _ = get_candidate_graph(
        segmentation=val_segmentation,
        max_edge_distance=max_edge_distance,
        iou=False,
        dT=dT,
    )

    if symmetric_division_cost:
        print(" Adding hyper elements to val candidate graph ...")
        val_candidate_graph = add_hyper_elements(val_candidate_graph)

    print("Converting train candidate graph to train track graph")

    # build a train track graph
    train_track_graph = TrackGraph(
        nx_graph=train_candidate_graph, frame_attribute="time"
    )

    print("Converting val candidate graph to val track graph")
    # build a val track graph
    val_track_graph = TrackGraph(nx_graph=val_candidate_graph, frame_attribute="time")

    # Step 2 ------------------------------

    # build ground truth track graph

    # add nodes
    groundtruth_graph, node_frame_dict = nodes_from_segmentation(train_segmentation)

    # add edges
    f = open(train_segmentation_dir_name + "/man_track.json")
    gt_data = json.load(f)

    parent_daughter_dict = {}

    for key, value in gt_data.items():
        parent_id = int(value[1])
        if parent_id in parent_daughter_dict.keys():
            pass
        else:
            parent_daughter_dict[parent_id] = []
        parent_daughter_dict[parent_id].append(
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

            temp_node = parent_node
            for daughter_node in parent_daughter_dict[parent_id]:
                temp_node += "_" + daughter_node

            if symmetric_division_cost:
                if len(parent_daughter_dict[parent_id]) == 1:
                    groundtruth_graph.add_edge(
                        parent_node, parent_daughter_dict[parent_id][0]
                    )
                elif len(parent_daughter_dict[parent_id]) == 2:
                    groundtruth_graph.add_node(temp_node)
                    groundtruth_graph.add_edge(parent_node, temp_node)
                    for daughter_node in parent_daughter_dict[parent_id]:
                        groundtruth_graph.add_edge(temp_node, daughter_node)
            else:
                groundtruth_graph.add_edge(
                    parent_node, str(int(np.min(value[0]))) + "_" + str(key)
                )

    ## build a groundtruth track graph

    print("Converting train gt graph to train track graph")
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
    weights_by_name = optimal_weights._weights_by_name
    print(f"After fitting, optimal weights are {optimal_weights}")
    optimal_weights_array = optimal_weights.to_ndarray()

    fitted_weights_dic = {}
    for i, key in enumerate(weights_by_name):
        fitted_weights_dic[key[0] + "_" + key[1]] = str(optimal_weights_array[i])

    with open(results_dir + "/weights.json", "w") as fp:
        json.dump(fitted_weights_dic, fp)

    # Step 4 ---------------------------------

    solver = Solver(track_graph=val_track_graph)
    solver = add_costs_and_constraints(solver, symmetric_division_cost)
    solver.weights.from_ndarray(optimal_weights_array)

    solution = solver.solve(verbose=True)
    solution_graph = solver.get_selected_subgraph(solution)

    new_mapping, res_track, tracked_masks = save_result_tifs_json(
        solution_nx_graph=graph_to_nx(solution_graph),
        segmentation=val_segmentation[:, 0],
        output_tif_dir=results_dir,
    )

    print("Computing scores ...")
    compute_metrics(
        gt_json_file_name=val_segmentation_dir_name + "/man_track.json",
        gt_segmentation=val_segmentation[:, 0],
        pred_json_file_name=results_dir + "/man_track.json",
        pred_segmentation=tracked_masks,
        results_dir=results_dir,
    )


if __name__ == "__main__":
    parser = jsonargparse.ArgumentParser()
    parser.add_argument(
        "--train_segmentation_dir_name",
        dest="train_segmentation_dir_name",
        default="./Fluo-N2DL-HeLa/02_GT0.05/",
    )
    parser.add_argument(
        "--val_segmentation_dir_name",
        dest="val_segmentation_dir_name",
        default="./Fluo-N2DL-HeLa/01_GT0.05/",
    )
    parser.add_argument(
        "--max_edge_distance", dest="max_edge_distance", type=float, default=50
    )
    parser.add_argument(
        "--regularizer_weight", dest="regularizer_weight", default=1e2, type=float
    )
    parser.add_argument(
        "--symmetric_division_cost",
        dest="symmetric_division_cost",
        type=bool,
        default=False,
    )
    parser.add_argument(
        "--results_dir_name", dest="results_dir_name", default="Fluo-N2DL-HeLa_results/"
    )

    parser.add_argument("--dT", dest="dT", type=int, default=1)
    args = parser.parse_args()
    print("+" * 10)
    print(args)
    print("+" * 10)

    if os.path.exists(args.results_dir_name):
        pass
    else:
        os.makedirs(args.results_dir_name)

    parser.save(
        args, args.results_dir_name + "/args.json", format="json", overwrite=True
    )

    track(
        train_segmentation_dir_name=args.train_segmentation_dir_name,
        val_segmentation_dir_name=args.val_segmentation_dir_name,
        max_edge_distance=args.max_edge_distance,
        regularizer_weight=args.regularizer_weight,
        symmetric_division_cost=args.symmetric_division_cost,
        dT=args.dT,
        results_dir=args.results_dir_name,
    )
