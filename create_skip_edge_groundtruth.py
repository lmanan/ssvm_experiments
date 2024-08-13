from glob import glob
import numpy as np
import tifffile
from argparse import ArgumentParser
from natsort import natsorted
import os
import json
from tqdm import tqdm

np.random.seed(42)


def create_new_groundtruth(
    dir_name: str, man_track_file_name: str, drop_out_fraction: float
):
    """create_new_groundtruth.
    This function drops out a percentage of random segmentations.
    We use this function to test what cost would work well with skip_edges.

    Parameters
    ----------
    dir_name : str
        dir_name contains the ground truth tracked tif files.
    man_track_file_name : str
        man_track_file_name is the name of the file mentioning the daughter parent
        relationships.
    drop_out_fraction : float, 0<=drop_out_fraction<1
        drop_out_fraction is the fraction of nodes/cells dropped per frame.
    """
    filenames = natsorted(glob(dir_name + "/*.tif"))
    print(f"Number of time frames is {len(filenames)}")
    daughter_parent_data = np.loadtxt(man_track_file_name, delimiter=" ")
    print(f"man track data has shape {daughter_parent_data.shape}")

    print(f"dropout fraction has value {drop_out_fraction}")

    out_segmentation_dir_name = os.path.dirname(dir_name)
    out_segmentation_dir_name += str(drop_out_fraction)
    print(f"Dropped out tifs will be saved here {out_segmentation_dir_name}")
    if os.path.exists(out_segmentation_dir_name):
        pass
    else:
        os.makedirs(out_segmentation_dir_name)

    daughter_parent_dic = {}
    for row in daughter_parent_data:
        daughter_parent_dic[int(row[0])] = int(row[3])

    man_track_data = {}
    for t, filename in enumerate(tqdm(filenames)):
        mask = tifffile.imread(filename)
        ids = np.unique(mask)
        ids = ids[ids != 0]
        for id_ in ids:
            if id_ in man_track_data.keys():
                man_track_data[int(id_)][0].append(t)
            else:
                man_track_data[int(id_)] = (
                    [t],
                    daughter_parent_dic[id_],
                )  # list of frames, parent id
        np.random.shuffle(ids)
        drop_out_num_ids = int(len(ids) * drop_out_fraction)
        for i in range(drop_out_num_ids):
            mask[mask == ids[i]] = 0
            man_track_data[ids[i]][0].remove(t)
        tifffile.imwrite(
            out_segmentation_dir_name + "/" + str(t).zfill(3) + ".tif", mask
        )

    with open(out_segmentation_dir_name + "/man_track.json", "w") as outfile:
        json.dump(man_track_data, outfile)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--groundtruth_segmentation_dir_name",
        dest="groundtruth_segmentation_dir_name",
        default="Fluo-N2DL-HeLa/02_GT/TRA",
    )
    parser.add_argument(
        "--groundtruth_daughter_parent_file_name",
        dest="groundtruth_daughter_parent_file_name",
        default="Fluo-N2DL-HeLa/02_GT/TRA/man_track.txt",
    )
    parser.add_argument(
        "--dropout_fraction", dest="dropout_fraction", type=float, default=0.05
    )
    args = parser.parse_args()
    create_new_groundtruth(
        dir_name=args.groundtruth_segmentation_dir_name,
        man_track_file_name=args.groundtruth_daughter_parent_file_name,
        drop_out_fraction=args.dropout_fraction,
    )
