import json
from logging.config import dictConfig
from fever.api.web_server import fever_web_api
import os
import logging

from jack import readers
from jack.core import QASetting
import torch
import torch.nn as nn
import torch.nn.functional as F


import constants
from ir import get_retrieval_method
from rte import get_predictor
import pickle
from fever_io import titles_to_jsonl_num

hexaf_reader = readers.reader_from_file(constants.results_dir + "/base+sampling2+evscores+rerank+train+dev+test-shared_test.ver0727_newaggr_submission/reader", dropout=0.0)


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
    preds_list = predict(hexaf_reader, settings, 32)
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


def predict_single(predictor, retrieval_method, instance):
    evidence = retrieval_method.get_sentences_for_claim(instance["claim"])

    test_instance = predictor._json_to_instance({"claim":instance["claim"], "predicted_sentences":evidence})
    predicted = predictor.predict_instance(test_instance)

    max_id = predicted["label_logits"].index(max(predicted["label_logits"]))

    return {
        "predicted_label":predictor._model.vocab.get_token_from_index(max_id,namespace="labels"),
        "predicted_evidence": evidence
    }


def hexaf_fever():
    logger = logging.getLogger()
    dictConfig({
        'version': 1,
        'formatters': {'default': {
            'format': '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
        }},
        'handlers': {'wsgi': {
            'class': 'logging.StreamHandler',
            'stream': 'ext://sys.stderr',
            'formatter': 'default'
        }},
        'root': {
            'level': 'INFO',
            'handlers': ['wsgi']
        },
        'allennlp': {
            'level': 'INFO',
            'handlers': ['wsgi']
        },
    })

    logger.info("My sample FEVER application")
    config = json.load(open(os.getenv("CONFIG_PATH","configs/predict_docker.json")))


    # load fever data
    logger.info("Load FEVER Document database")
    with open(constants.index_dir + "/edocs.bin","rb") as rb:
        edocs=pickle.load(rb)
    with open(constants.index_dir + "/doc_ir_model.bin","rb") as rb:
        dmodel=pickle.load(rb)
    t2jnum=titles_to_jsonl_num()
    with open(constants.index_dir + "/line_ir_model.bin","rb") as rb:
        lmodel=pickle.load(rb)

        retrieval_method = get_retrieval_method(dmodel, lmodel, edocs, t2jnum)

    hexaf_reader = readers.reader_from_file(constants.results_dir + "/base+sampling2+evscores+rerank+train+dev+test-shared_test.ver0727_newaggr_submission/reader", dropout=0.0)

    aggregator_path = constants.index_dir + "/aggregator.pt"
    predictor = get_predictor(hexaf_reader, aggregator_path)



    # The prediction function that is passed to the web server for FEVER2.0
    def baseline_predict(instances):
        predictions = []
        for instance in instances:
            predictions.append(predict_single(predictor, retrieval_method, instance))
        return predictions

    return fever_web_api(baseline_predict)
