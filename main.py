import tifffile
import numpy as np
from glob import glob
from natsort import natsorted
import argparse
from motile_toolbox.candidate_graph import get_candidate_graph, EdgeAttr
from motile_toolbox.candidate_graph import nodes_from_segmentation
from motile.track_graph import TrackGraph
from motile.solver import Solver
from motile.constraints import MaxChildren, MaxParents
from motile.variables import NodeSelected, EdgeSelected
from motile.costs.appear import Appear
from motile.costs import EdgeSelection
import logging

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s %(name)s %(levelname)-8s %(message)s"
)
logger = logging.getLogger(__name__)


def track(segmentation_dir_name: str, max_edge_distance: float, frac: float = 0.1):
    """
    This function does three things:

    Step 1. First it sets up a candidate graph using available GT detections and by
    connecting neighbors within `max_edge_distance`. Then a solution is obtained
    using default weights for IoU edge cost and appearance node cost.

    Step 2. Next, the actual ground truth graph is created using the `man_track.txt`
    text file provided by CTC website for this dataset (available within the
    `segmentation_dir_name` directory).

    Step 3. A fraction of the nodes are randomly sampled and the outgoing edges
    from these nodes are either specified to be `True` if they are indeed the
    ground truth edge or `False` if they are a candidate, non ground truth edge. Next, we try to
    identify best weights using SSVM.


    Parameters
    ----------
    segmentation_dir_name : str
        segmentation_dir_name is the path to the `TRA` directory containing all
        tif images.
    max_edge_distance : float
        max_edge_distance is used to connect nodes that lie at an L2 distance
        lesser than max_edge_distance while constructing a candidate graph
        (see Step 1 above).
    frac: float (between 0.0 and 1.0)
        Fraction of the nodes for which the "gt" attribute is specified, while
        fitting weights using SSVM. Default = 0.1
    """
    # Step 1 ------------------------------

    # obtain masks
    filenames = natsorted(glob(segmentation_dir_name + "/*.tif"))
    segmentation = []
    for filename in filenames:
        segmentation.append(tifffile.imread(filename))
    segmentation = np.asarray(segmentation)
    segmentation = segmentation[:, np.newaxis]  # requires a hypothesis channel

    # get candidate graph
    candidate_graph, _ = get_candidate_graph(
        segmentation=segmentation, max_edge_distance=max_edge_distance, iou=True
    )

    # get number of nodes and edges in candidate graph
    print(f"Number of edges in candidate graph is {len(candidate_graph.edges())}")
    print(f"Number of nodes in candidate graph is {len(candidate_graph.nodes())}")

    # build a track graph
    track_graph = TrackGraph(nx_graph=candidate_graph, frame_attribute="time")

    # create solver
    solver = Solver(track_graph=track_graph)

    # add constraints
    solver.add_constraints(MaxParents(max_parents=1))
    solver.add_constraints(MaxChildren(max_children=2))

    # add costs
    solver.add_costs(
        EdgeSelection(weight=-0.1, attribute=EdgeAttr.IOU.value), name="IoU"
    )
    solver.add_costs(Appear(constant=0.6))

    ## solve
    solution = solver.solve(verbose=True)
    nodes = solver.get_variables(NodeSelected)
    edges = solver.get_variables(EdgeSelected)
    selected_nodes = [node for node in track_graph.nodes if solution[nodes[node]] > 0.5]
    selected_edges = [edge for edge in track_graph.edges if solution[edges[edge]] > 0.5]
    print(f"Number of selected edges in solution graph is {len(selected_edges)}")
    print(f"Number of selected nodes in solution graph is {len(selected_nodes)}")

    # Step 2 ------------------------------

    # build ground truth track graph
    # add nodes
    groundtruth_graph, node_frame_dict = nodes_from_segmentation(segmentation)

    # add edges
    gt_data = np.loadtxt(segmentation_dir_name + "/man_track.txt", delimiter=" ")

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
    # get number of nodes and edges in ground truth track graph
    print(f"Number of edges in groundtruth graph is {len(groundtruth_graph.edges())}")
    print(f"Number of nodes in groundtruth graph is {len(groundtruth_graph.nodes())}")

    # Step 3 ---------------------------------

    ## fitting weights now ...

    # Let's say around 10 percent (specified by `frac`)of the edges are correct.
    # We randomly sample 10 percent (or, `frac`) of the nodes and then set the actual (g.t.)
    # outgoing edge to be True and the other (candidate, non g.t.) ones to be False.

    for node_id in track_graph.nodes:
        t = np.random.rand(1)
        if t <= frac:
            for edge_id in list(track_graph.next_edges[node_id]):
                if edge_id in groundtruth_track_graph.edges:
                    track_graph.edges[edge_id]["gt"] = True
                else:
                    track_graph.edges[edge_id]["gt"] = False

    # fit weights
    solver.fit_weights(gt_attribute="gt", regularizer_weight=1e-3, max_iterations=1000)
    optimal_weights = solver.weights
    print(f"After fitting, optimal weights are {optimal_weights}")

    solver = Solver(track_graph=track_graph)
    solver.add_constraints(MaxParents(1))
    solver.add_constraints(MaxChildren(2))

    solver.add_costs(
        EdgeSelection(weight=-0.1, attribute=EdgeAttr.IOU.value), name="IoU"
    )
    solver.add_costs(Appear(constant=0.6))
    solver.weights.from_ndarray(optimal_weights.to_ndarray())

    solution = solver.solve(verbose=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--segmentation_dir_name",
        dest="segmentation_dir_name",
        default="./Fluo-N2DL-HeLa/01_GT/TRA/",
    )
    parser.add_argument("--max_edge_distance", dest="max_edge_distance", default=50)
    args = parser.parse_args()
    track(
        segmentation_dir_name=args.segmentation_dir_name,
        max_edge_distance=args.max_edge_distance,
    )
