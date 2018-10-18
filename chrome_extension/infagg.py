from flask import Flask, jsonify, abort, request
from jack import readers
from jack.core import QASetting
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


app = Flask(__name__)
dam_reader = readers.reader_from_file("reader", dropout=0.0)



def convert_label(label, inverse=False):
    fever2snli = {
        "SUPPORTS": "entailment",
        "REFUTES": "contradiction",
        "NOT ENOUGH INFO": "neutral"
    }
    snli2fever = {snli: fever for fever, snli in fever2snli.items()}
    if not inverse:
        assert label in fever2snli
        return fever2snli[label]
    else:
        assert label in snli2fever
        return snli2fever[label]

def reshape(preds_list, preds_length):
    """reshape prediction instances
    >> preds_list = [obj, obj, obj, obj, obj, obj]
    >> preds_length = [3, 1, 2]
    >> reshape(preds_list, preds_length)
    [[obj, obj, obj], [obj], [obj, obj]]
    """
    reshaped = list()
    pointer = 0
    for i, length in enumerate(preds_length):
        preds = preds_list[pointer: pointer + length]
        pointer += length
        reshaped.append(preds)
    return reshaped


def flatten(bumpy_2d_list):
    flattened = list()
    for list_ in bumpy_2d_list:
        flattened.extend(list_)
    return flattened


def predict(reader, all_settings, batch_size):
    # pointer loops from 0 to less than (or equal to) len(all_settings) with step batch_size
    preds_list = list()
    for pointer in range(0, len(all_settings), batch_size):
        batch_settings = all_settings[pointer: pointer + batch_size]
        preds_list.extend(reader(batch_settings))
    return preds_list

def score(claim, evidence_list):
    settings = [QASetting(question=claim, support=[evidence]) for evidence in evidence_list]
    preds_list = predict(dam_reader, settings, 32)
    pred_labels_list = [[pred.text for pred in preds_instance] for preds_instance in preds_list]
    scores = [[float(pred.score) for pred in preds_instance] for preds_instance in preds_list]
    dic = {
        "scores": scores,
        "predicted_labels": [[convert_label(pred_label, inverse=True) for pred_label in pred_labels] for pred_labels in pred_labels_list]
        }
    return dic

class Net(nn.Module):
    def __init__(self, layers=[15, 10, 5]):
        super(Net, self).__init__()
        assert len(layers) == 3, "currently, only supports 3 layer MLP"
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(layers[0], layers[1])
        self.fc2 = nn.Linear(layers[1], layers[2])
        self.fc3 = nn.Linear(layers[2], 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net(layers=[int(width) for width in [21,100,100]])
net.load_state_dict(torch.load("aggregator.pt"))

label2idx = {"SUPPORTS": 0, "REFUTES": 1, "NOT ENOUGH INFO": 2, "DUMMY LABEL": 3}
idx2label = {idx: label for label, idx in label2idx.items()}
supports = idx2label[0]
refutes = idx2label[1]
nei = idx2label[2]

def aggregate(rte_out):
    results = list()
    with torch.no_grad():
        for claim_pred in rte_out:
            features_per_predicted_evidence = []
            for per_evidence_scores in claim_pred["scores"][:7]:
                if not per_evidence_scores: # sometimes this is empty.
                    new_features = [0.0, 0.0, 0.0]
                else:
                    # default case
                    new_features = per_evidence_scores
                features_per_predicted_evidence.extend(new_features)
            for missing_evidence in range(7 - len(claim_pred["scores"])):
                features_per_predicted_evidence.extend([0.0, 0.0, 0.0])
            input = torch.Tensor(np.array([features_per_predicted_evidence]))
            neural_preds = net(input.float())
            _, pred_labels = torch.max(neural_preds, 1)
            for neural_pred in pred_labels:
                results.append({"label": idx2label[int(neural_pred)]})

    return results


@app.route('/api/factcheck', methods=['POST'])
def get_predictions():
    if not request.json:
        abort(400)
    data=request.json
    claim=data["claim"]
    evidence_list=[ev["element"]["text"] for ev in data["evidences"]]
    claim_pred=score(claim, evidence_list)
    for i, (ev,labs) in enumerate(zip(claim_pred["scores"],claim_pred["predicted_labels"])):
        maxscore=0
        maxidx=-1
        for idx, s in enumerate(ev):
            if s > maxscore:
                maxscore=s
                maxidx=idx
        data["evidences"][i]["group_fact_check"]=labs[maxidx]
    label=aggregate([claim_pred])
    data["global_fc"]=label[0]["label"]
    return jsonify(data)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=4000)
