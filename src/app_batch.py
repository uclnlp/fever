#!/usr/bin/env python
import os
from fever_io import load_fever_train
from get_evidence import get_evidence, tofeverformat
from converter import convert
from fever_io import save_jsonl, read_jsonl
from pipeline import __run_python
import constants
import logging
import argparse
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from neural_aggregator import Net, PredictedLabelsDataset, predict

# these two imports are necessary to unpickle objects!!! (get_evidence)
from doc_ir_model import doc_ir_model
from line_ir_model import line_ir_model

root_dir = "/hexaf"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--infile", required=True)
args = parser.parse_args()

# load input file
logger.info("loading index files...")
n_docs = 30
n_sents = 15
instances = load_fever_train(path=args.infile, howmany=9999999999)
docs, evidence = get_evidence(instances, n_docs=30, n_sents=15)
pred_ev= tofeverformat(instances, docs, evidence)
save_jsonl(pred_ev, "/tmp/predicted_evidences.jsonl")

# this is not needed unless you train rte model
# # convert to snli format
prependlinum = False
prependtitle = True
use_ir_pred = False
n_sentences = 5
snli_format_instances = convert(
    instances,
    prependlinum=prependlinum,
    prependtitle=prependtitle,
    use_ir_prediction=use_ir_pred,
    n_sentences=str(n_sentences))
save_jsonl(snli_format_instances, "/tmp/converted_instances.jsonl")

# # inference rte
n_sentences = 15
options = [
    "/tmp/predicted_evidences.jsonl", "/tmp/rte_predictions.jsonl",
    "--saved_reader", constants.reader,
    "--prependtitle", "--n_sentences", str(n_sentences)
]
script = ["../fever/jack_reader.py"] + options
os.chdir(os.path.join(root_dir, "jack"))
__run_python(script, gpu=True, env={"PYTHONPATH": "."})
os.chdir("../fever")

# neural_aggregator
# TODO: make sure that rte_pred contains "ev_scores" key. (ref. neural_aggregator.py l66)
# TODO: training
layers = [36, 100, 100]
net = Net(layers=[int(width) for width in layers])
net.load_state_dict(torch.load(os.path.join(constants.index_dir, "aggregator.pt")))

test_set = PredictedLabelsDataset("/tmp/rte_predictions.jsonl", n_sentences=9, use_ev_scores=True)
test_dataloader = DataLoader(
    test_set, batch_size=64, shuffle=False, num_workers=0)
aggregated_labels= predict(test_dataloader, net)

# rerank
pred_ev = read_jsonl("/tmp/predicted_evidences.jsonl") # tmp
logger.info("running reranking")
n_sentences = 15
ev_ids = [ev["id"] for ev in pred_ev]
pred_ev = [ev["predicted_sentences"] for ev in pred_ev]
rte_pred = [p["predicted_labels"] for p in read_jsonl("/tmp/rte_predictions.jsonl")]

predictions = []
assert len(rte_pred) == len(aggregated_labels) == len(
    pred_ev) == len(ev_ids), "{}, {}, {}, {}".format(
        len(rte_pred), len(aggregated_labels), len(pred_ev),
        len(ev_ids))
for idx, (id, ev, rte_labels, aggr_label) in enumerate(
        zip(ev_ids, pred_ev, rte_pred, aggregated_labels)):
        rte_labels = rte_labels[0]  # remove additional dimension
        predictions.append({
            "id": id,
            "rte_preds": rte_labels,
            "predicted_label": aggr_label,
            "predicted_evidence": ev[:n_sentences]
        })

out_preds = list()
for pred in predictions:
    if len(pred["rte_preds"]) != len(pred["predicted_evidence"]):
        pred["rte_preds"] = pred["rte_preds"][:len(pred["predicted_evidence"])]

    # no reranking if num of rte preds are lower than 5
    if len(pred["rte_preds"]
        ) > 5 and pred["predicted_label"] != "NOT ENOUGH INFO":
        correct_ev_flags = (pred["predicted_label"] == np.array(
            pred["rte_preds"]))
        correct_ev_args = np.reshape(
            np.argwhere(correct_ev_flags == True), (-1))
        incorrect_ev_args = np.reshape(
            np.argwhere(correct_ev_flags == False), (-1))

        correct_evs = [
            pred["predicted_evidence"][idx] for idx in correct_ev_args
        ]
        incorrect_evs = [
            pred["predicted_evidence"][idx] for idx in incorrect_ev_args
        ]
        out_ev = (correct_evs + incorrect_evs)

    out_dict = {"id": pred["id"], "predicted_sentences": out_ev}
    out_preds.append(out_dict)

save_jsonl(out_preds, "/hexaf/fever/result.jsonl")
