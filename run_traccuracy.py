import os
from traccuracy import run_metrics
from traccuracy.loaders import load_ctc_data
from traccuracy.matchers import CTCMatcher
from traccuracy.metrics import CTCMetrics, DivisionMetrics
import pprint

pp = pprint.PrettyPrinter(indent=4)

gt_data=load_ctc_data(data_dir='./Fluo-N2DL-HeLa/01_GT/TRA', track_path='./Fluo-N2DL-HeLa/01_GT/TRA/man_track.txt', name= 'HeLa-01_GT')

pred_data = load_ctc_data(data_dir='./01_RES_SSVM', track_path='./01_RES_SSVM/res_track.txt', name='HeLa-01_RES')


ctc_results = run_metrics(gt_data=gt_data, pred_data = pred_data, matcher = CTCMatcher(), metrics=[CTCMetrics(), DivisionMetrics()])
pp.pprint(ctc_results)



print("| Fraction | Appearance | Disappearance | Division | IoU | Position |  AOGM | DET | TRA | FN edges | FN nodes | FP edges | FP nodes | NS nodes | WS edges | Div F1 | Div Prec | Div Rec | FN Div | FP Div | MBC | TP Div |")
print("| -------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- |------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- |------- |------- |------- | ------- |")
print(f"|1.0 | <span class=\"checked\">✔</span>   | <span class=\"checked\">✔</span>     | <span class=\"checked\">✔</span>     | <span class=\"checked\">✔</span>     | <span class=\"checked\">✔</span>| {ctc_results[0]['results']['AOGM']}|{ctc_results[0]['results']['DET']}|{ctc_results[0]['results']['TRA']}|{ctc_results[0]['results']['fn_edges']}|{ctc_results[0]['results']['fn_nodes']}|{ctc_results[0]['results']['fp_edges']}|{ctc_results[0]['results']['fp_nodes']}|{ctc_results[0]['results']['ns_nodes']}|{ctc_results[0]['results']['ws_edges']}|{ctc_results[1]['results']['Frame Buffer 0']['Division F1']}|{ctc_results[1]['results']['Frame Buffer 0']['Division Precision']}|{ctc_results[1]['results']['Frame Buffer 0']['Division Recall']}|{ctc_results[1]['results']['Frame Buffer 0']['False Negative Divisions']}|{ctc_results[1]['results']['Frame Buffer 0']['False Positive Divisions']}|{ctc_results[1]['results']['Frame Buffer 0']['Mitotic Branching Correctness']}|{ctc_results[1]['results']['Frame Buffer 0']['True Positive Divisions']}")

