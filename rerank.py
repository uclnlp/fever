import argparse
import json
import numpy as np


def load_jsonl(path, key=None):
    out = list()
    if key is None:
        with open(path, "r") as f:
            for line in f:
                out.append(json.loads(line))
    else:
        with open(path, "r") as f:
            for line in f:
                out.append(json.loads(line)[key])
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rte_predictions", required=True, help="test.predictions.jsonl")
    parser.add_argument("--aggregated_labels", required=True, help="test_aggregated_labels.jsonl")
    parser.add_argument("--predicted_evidences", required=True, help="test.sentences.scores.p15.s30.jsonl")
    parser.add_argument("--reranked_evidences", required=True)
    parser.add_argument("--n_sentences", required=True, type=int)
    args = parser.parse_args()
    print(args)

    rte_predictions = load_jsonl(args.rte_predictions, key="predicted_labels")
    aggregated_labels = load_jsonl(args.aggregated_labels, key="predicted")
    predicted_evidences = load_jsonl(args.predicted_evidences, key="predicted_sentences")
    ids = load_jsonl(args.predicted_evidences, key="id")

    predictions = []
    assert len(rte_predictions) == len(aggregated_labels) == len(predicted_evidences) == len(ids), "{}, {}, {}, {}".format(len(rte_predictions), len(aggregated_labels), len(predicted_evidences), len(ids))
    for idx, (id, ev, rte_labels, aggr_label) in enumerate(zip(ids, predicted_evidences, rte_predictions, aggregated_labels)):
        # temporal fix (suspicious)
        if len(rte_labels) == 1:
            rte_labels = rte_labels[0]
        predictions.append({"id": id, "rte_preds": rte_labels, "predicted_label": aggr_label, "predicted_evidence": ev[:args.n_sentences]})

    out_preds = list()
    for pred in predictions:
        if len(pred["rte_preds"]) != len(pred["predicted_evidence"]):
            # print("warning: pred['rte_preds'] != len(pred['predicted_evidence'])")
            pred["rte_preds"] = pred["rte_preds"][:len(pred["predicted_evidence"])]

        # no reranking if num of rte preds are lower than 5
        if len(pred["rte_preds"]) > 5 and pred["predicted_label"] != "NOT ENOUGH INFO":
            correct_ev_flags = (pred["predicted_label"] == np.array(pred["rte_preds"]))
            correct_ev_args = np.reshape(np.argwhere(correct_ev_flags==True), (-1))
            incorrect_ev_args = np.reshape(np.argwhere(correct_ev_flags==False), (-1))

            correct_evs = [pred["predicted_evidence"][idx] for idx in correct_ev_args]
            incorrect_evs = [pred["predicted_evidence"][idx] for idx in incorrect_ev_args]
            out_ev = (correct_evs + incorrect_evs)
        else:
            out_ev = pred["predicted_evidence"]

        out_dict = {
            "id": pred["id"],
            "predicted_sentences": out_ev
        }
        out_preds.append(out_dict)

    print("saving file at {}".format(args.reranked_evidences))
    with open(args.reranked_evidences, "w") as f:
        for out_pred in out_preds:
            f.write(json.dumps(out_pred) + "\n")


