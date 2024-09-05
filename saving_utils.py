from tqdm import tqdm
from pathlib import Path
import tifffile
import numpy as np
import networkx as nx
import json


def save_result_tifs_json(
    solution_nx_graph: nx.DiGraph, segmentation: np.ndarray, output_tif_dir: str
):

    tracked_masks = np.zeros_like(segmentation)
    new_mapping = {}  # <t_id> in segmentation mask: id in tracking mask
    res_track = (
        {}
    )  # id in tracking mask: ([t0, t1, ..., tN], parent_id) in tracking mask
    id_counter = 1
    for in_node, out_node in tqdm(solution_nx_graph.edges()):
        t_in, id_in = in_node.split("_")
        t_out, id_out = out_node.split("_")
        t_in, id_in = int(t_in), int(id_in)
        t_out, id_out = int(t_out), int(id_out)
        num_out_edges = len(solution_nx_graph.out_edges(in_node))
        if num_out_edges == 1:
            if in_node in new_mapping.keys():
                # i.e. continuation of an existing edge
                if t_out not in res_track[new_mapping[in_node]][0]:
                    res_track[new_mapping[in_node]][0].append(
                        t_out
                    )  # include the end time for this tracklet
                tracked_masks[t_in][segmentation[t_in] == id_in] = new_mapping[in_node]
                new_mapping[out_node] = new_mapping[in_node]
                tracked_masks[t_out][segmentation[t_out] == id_out] = new_mapping[
                    out_node
                ]
            else:
                # i.e. start of a new edge
                res_track[id_counter] = ([t_in, t_out], 0)
                new_mapping[in_node] = id_counter
                new_mapping[out_node] = id_counter
                tracked_masks[t_in][segmentation[t_in] == id_in] = id_counter
                tracked_masks[t_out][segmentation[t_out] == id_out] = id_counter
                id_counter += 1
        elif num_out_edges == 2:
            out_edge1, out_edge2 = solution_nx_graph.out_edges(in_node)
            _, out_node1 = out_edge1
            _, out_node2 = out_edge2
            t_out1, id_out1 = out_node1.split("_")
            t_out1, id_out1 = int(t_out1), int(id_out1)
            t_out2, id_out2 = out_node2.split("_")
            t_out2, id_out2 = int(t_out2), int(id_out2)
            if in_node in new_mapping.keys():
                # i.e. in node was connected by one outgoing edge previously
                if t_in not in res_track[new_mapping[in_node]][0]:
                    res_track[new_mapping[in_node]][0].append(t_in)
                tracked_masks[t_in][segmentation[t_in] == id_in] = new_mapping[in_node]
                if out_node1 not in new_mapping:
                    new_mapping[out_node1] = id_counter
                    tracked_masks[t_out1][segmentation[t_out1] == id_out1] = id_counter
                    res_track[id_counter] = ([t_out1], new_mapping[in_node])
                    id_counter += 1
                if out_node2 not in new_mapping:
                    new_mapping[out_node2] = id_counter
                    tracked_masks[t_out2][segmentation[t_out2] == id_out2] = id_counter
                    res_track[id_counter] = ([t_out2], new_mapping[in_node])
                    id_counter += 1
            else:
                res_track[id_counter] = ([t_in], 0)
                new_mapping[in_node] = id_counter
                tracked_masks[t_in][segmentation[t_in] == id_in] = id_counter
                id_counter += 1
                if out_node1 not in new_mapping:
                    new_mapping[out_node1] = id_counter
                    tracked_masks[t_out1][segmentation[t_out1] == id_out1] = id_counter
                    res_track[id_counter] = ([t_out1], new_mapping[in_node])
                    id_counter += 1
                if out_node2 not in new_mapping:
                    new_mapping[out_node2] = id_counter
                    tracked_masks[t_out2][segmentation[t_out2] == id_out2] = id_counter
                    res_track[id_counter] = ([t_out2], new_mapping[in_node])
                    id_counter += 1

    # in case there are edges that start and end at the same node, we do a
    # little more
    for node in solution_nx_graph.nodes():
        t, id_node = node.split("_")
        t, id_node = int(t), int(id_node)
        if node in new_mapping.keys():
            pass
        else:
            res_track[id_counter] = ([t], 0)
            new_mapping[node] = id_counter

            tracked_masks[t][segmentation[t] == id_node] = id_counter
            id_counter += 1

    # ensure that path where tifs will be saved, exists.
    if Path(output_tif_dir).exists():
        filenames = list(Path(output_tif_dir).glob("*.tif"))
        for filename in filenames:
            Path(filename).unlink()
    else:
        Path(output_tif_dir).mkdir()
    # write tifs
    for i in range(tracked_masks.shape[0]):
        tifffile.imwrite(
            Path(output_tif_dir) / ("mask" + str(i).zfill(3) + ".tif"),
            tracked_masks[i].astype(np.uint16),
        )
    # write man_track.json
    with open(output_tif_dir + "/man_track.json", "w") as fp:
        json.dump(res_track, fp)

    return new_mapping, res_track, tracked_masks
