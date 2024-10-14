import jsonargparse
import numpy as np
from motile_toolbox.candidate_graph import get_candidate_graph_from_points_list
import networkx as nx
import json
from yaml import load, Loader

np.random.seed(42)


def flip_edges(candidate_graph: nx.DiGraph):
    candidate_graph_copy = nx.DiGraph()
    candidate_graph_copy.add_nodes_from(candidate_graph.nodes(data=True))
    for node1, node2, data in candidate_graph.edges(data=True):
        candidate_graph_copy.add_edge(node2, node1, **data)
    return candidate_graph_copy


def add_gt_edges_to_graph(groundtruth_graph: nx.DiGraph, gt_data: np.ndarray):
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
            end_node = str(t) + "_" + str(id_)
            groundtruth_graph.add_edge(start_node, end_node)
    return groundtruth_graph


def cumulate_attrackt_scores(
    attrackt_csv_file_name: str,
    ilp_csv_file_name: str,
    detections_csv_file_name: str,
    output_csv_file_name: str,
    backward_output_gt_file_name: str,
    forward_output_gt_file_name: str,
    select_num_rows: int,
    sampling: str,
    num_nearest_neighbours: int = 5,
    dT: int = 1,
):

    detections_data = np.loadtxt(detections_csv_file_name, delimiter=" ")
    candidate_graph, _, _ = get_candidate_graph_from_points_list(
        points_list=detections_data,
        max_edge_distance=None,
        num_nearest_neighbours=num_nearest_neighbours,
        direction_candidate_graph="backward",
        dT=dT,
    )
    candidate_graph = flip_edges(candidate_graph)

    groundtruth_graph = nx.DiGraph()
    groundtruth_graph.add_nodes_from(candidate_graph.nodes(data=True))
    groundtruth_graph = add_gt_edges_to_graph(
        groundtruth_graph=groundtruth_graph, gt_data=detections_data
    )

    attrackt_data = np.loadtxt(
        attrackt_csv_file_name, delimiter=" "
    )  # id_t t id_tp1 tp1 weight
    attrackt_dictionary = {}
    for row in attrackt_data:

        if int(row[0]) == -1 or int(row[1]) == -1:
            pass
        else:
            # check if in dictionary
            node_id = str(int(row[1])) + "_" + str(int(row[0]))
            if node_id in attrackt_dictionary:
                pass
            else:
                attrackt_dictionary[node_id] = 0.0
            attrackt_dictionary[node_id] += float(row[-1])

    ilp_data = np.loadtxt(ilp_csv_file_name, delimiter=" ")
    ilp_dictionary = {}
    for row in ilp_data:
        id_u, t_u, id_v, t_v = row.astype(int)
        node_u = str(t_u) + "_" + str(id_u)
        if node_u in ilp_dictionary:
            ilp_dictionary[node_u] += 1.0
        else:
            ilp_dictionary[node_u] = 1.0

    final_array = []
    for key in attrackt_dictionary:
        deficit = attrackt_dictionary[key]
        if key in ilp_dictionary:
            deficit -= ilp_dictionary[key]
        deficit = np.abs(deficit)
        time, id_ = key.split("_")
        if key in ilp_dictionary:
            final_array.append(
                [
                    int(id_),
                    int(time),
                    attrackt_dictionary[key],
                    ilp_dictionary[key],
                    float(deficit),
                ]
            )
        else:
            final_array.append(
                [int(id_), int(time), attrackt_dictionary[key], 0.0, float(deficit)]
            )

    final_array = np.asarray(final_array)
    if sampling == "preferred":
        final_array_sorted = final_array[final_array[:, -1].argsort()[::-1]]
    elif sampling == "random":
        num_rows = final_array.shape[0]
        indices = np.arange(0, num_rows)
        np.random.shuffle(indices)  # in place
        final_array_sorted = final_array[indices]

    np.savetxt(
        fname=output_csv_file_name,
        X=final_array_sorted,
        delimiter=" ",
        fmt=["%d", "%d", "%.5f", "%.5f", "%.5f"],
    )

    final_array_filtered = final_array_sorted[:select_num_rows, :]

    backward_supervision_dictionary = {}
    for row in final_array_filtered:
        node_id = str(int(row[1])) + "_" + str(int(row[0]))
        out_edges = candidate_graph.out_edges(node_id)
        out_gt_edges = groundtruth_graph.out_edges(node_id)  # list of size 0, 1, 2

        for out_edge in out_edges:

            (u, v) = out_edge
            if v in backward_supervision_dictionary:
                pass
            else:
                backward_supervision_dictionary[v] = {}
            if out_edge in out_gt_edges:
                backward_supervision_dictionary[v][u] = 1.0
                in_edges = candidate_graph.in_edges(v)
                for in_edge in in_edges:
                    m, _ = in_edge
                    if in_edge != out_edge:
                        backward_supervision_dictionary[v][m] = 0.0
            else:
                backward_supervision_dictionary[v][u] = 0.0

    with open(backward_output_gt_file_name, "w") as outfile:
        json.dump(backward_supervision_dictionary, outfile)

    forward_supervision_dictionary = {}
    for row in final_array_filtered:
        node_id = str(int(row[1])) + "_" + str(int(row[0]))
        out_edges = candidate_graph.out_edges(node_id)
        out_gt_edges = groundtruth_graph.out_edges(node_id)  # list of size 0, 1, 2
        for out_edge in out_edges:
            (u, v) = out_edge
            if u in forward_supervision_dictionary:
                pass
            else:
                forward_supervision_dictionary[u] = {}

            if out_edge in out_gt_edges:
                forward_supervision_dictionary[u][v] = 1.0
                in_edges = candidate_graph.in_edges(v)
                for in_edge in in_edges:
                    m, _ = in_edge
                    if m in forward_supervision_dictionary:
                        pass
                    else:
                        forward_supervision_dictionary[m] = {}

                    if in_edge != out_edge:
                        forward_supervision_dictionary[m][v] = 0.0
            else:
                forward_supervision_dictionary[u][v] = 0.0

    with open(forward_output_gt_file_name, "w") as outfile:
        json.dump(forward_supervision_dictionary, outfile)


if __name__ == "__main__":
    parser = jsonargparse.ArgumentParser()
    parser.add_argument("--yaml_config_file_name", dest="yaml_config_file_name")
    args_ = parser.parse_args()
    with open(args_.yaml_config_file_name) as stream:
        args = load(stream, Loader=Loader)

    print(args)

    cumulate_attrackt_scores(
        attrackt_csv_file_name=args["attrackt_csv_file_name"],
        ilp_csv_file_name=args["ilp_csv_file_name"],
        output_csv_file_name=args["output_csv_file_name"],
        backward_output_gt_file_name=args["backward_output_gt_file_name"],
        forward_output_gt_file_name=args["forward_output_gt_file_name"],
        detections_csv_file_name=args["detections_csv_file_name"],
        select_num_rows=int(args["select_num_rows"]),
        sampling=args["sampling"],
    )
