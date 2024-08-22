from traccuracy import run_metrics
from traccuracy.loaders import load_deepcell_data
from traccuracy.matchers import CTCMatcher
from traccuracy.metrics import CTCMetrics, DivisionMetrics
import pprint
import argparse

pp = pprint.PrettyPrinter(indent=4)


def compute_metrics(
    gt_json_file_name,
    gt_segmentation,
    pred_json_file_name,
    pred_segmentation,
    results_dir,
):
    gt_data = load_deepcell_data(
        masks=gt_segmentation, json_file_name=gt_json_file_name, name="gt"
    )
    pred_data = load_deepcell_data(
        masks=pred_segmentation, json_file_name=pred_json_file_name, name="pred"
    )
    ctc_results = run_metrics(
        gt_data=gt_data,
        pred_data=pred_data,
        matcher=CTCMatcher(),
        metrics=[CTCMetrics(), DivisionMetrics()],
    )
    pp.pprint(ctc_results)
    import json

    with open(results_dir + "/results.json", "w") as fp:
        json.dump(ctc_results, fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_json_file_name", dest="gt_json_file_name")
    parser.add_argument("--gt_segmentation", dest="gt_segmentation")
    parser.add_argument("--pred_json_file_name", dest="pred_json_file_name")
    parser.add_argument("--pred_segmentation", dest="pred_segmentation")
    parser.add_argument("--results_dir", dest="results_dir")
    args = parser.parse_args()
    compute_metrics(
        args.gt_json_file_name,
        args.gt_segmentation,
        args.pred_json_file_name,
        args.pred_segmentation,
        args.results_dir,
    )
