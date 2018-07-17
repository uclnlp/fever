"""
analyse predictions
this file is located at fever-baselines/src/
"""

import os
import json
import argparse
import pandas
import random
import numpy as np
from tqdm import tqdm
from util import abs_path
from fever_io import titles_to_jsonl_num, load_doclines, read_jsonl, save_jsonl, get_evidence_sentence_list


def analyse(predictions, actual, out_file):
    print_confusion_mat(predictions, actual)
    save_wrong_instances(predictions, actual, out_file)


def save_simple_result(path, score, acc, precision, recall):
    save = {
        "FEVER score": round(score, 4),
        "Accuracy": round(acc, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4)
    }
    with open(path, "w") as f:
        json.dump(save, f)


def print_confusion_mat(predictions, actual):
    # table = {"actual_label": { "pred_label": count, "pred_label": "count"}, ...}
    confusion_mat = np.zeros((3, 3), dtype=np.int32)
    eye = np.eye(3)
    label2idx = {"SUPPORTS": 0, "REFUTES": 1, "NOT ENOUGH INFO": 2}
    labels = [
        label for label, val in sorted(label2idx.items(), key=lambda x: x[1])
    ]

    for pred_instance, actual_instance in zip(predictions, actual):
        confusion_mat[label2idx[actual_instance["label"]], label2idx[
            pred_instance["predicted_label"]]] += 1

    print("actual \ predicted")
    df = pandas.DataFrame(confusion_mat, labels, labels)
    print(df)

    pre = np.squeeze(
        np.sum(np.multiply(confusion_mat, eye), axis=0) / np.sum(
            confusion_mat, axis=0))
    rec = np.squeeze(
        np.sum(np.multiply(confusion_mat, eye), axis=0) / np.sum(
            confusion_mat, axis=1))

    print("precision")
    for pre_instance, label in zip(pre, labels):
        print("{}:\t{:.4}".format(label, pre_instance))

    print("recall")
    for rec_instance, label in zip(rec, labels):
        print("{}:\t{:.4}".format(label, rec_instance))


def shorten_labels(labels):
    """just return initial character of each label
    ["SUPPORTS", "REFUTES"] -> ["S", "R"]
    """
    return [
        "NEI" if label == "NOT ENOUGH INFO" else label[:3] for label in labels
    ]
    # return ["NEI    " if label == "NOT ENOUGH INFO" else label for label in labels]


def compare_evidences(actual_ev, pred_ev):
    """
    pred_ev: [["title", linum], ["title", linum], ...]
    actual_ev: [[[id, id, "title", linum]], [[id, id, "title", linum], [id, id, "title", linum]]]
    currently, not distinguish partial ev and full ev.
    just comparing whether the ev is making part of actual_ev
    """
    actual_evidences = [[title, linum] for evidence_set in actual_ev
                        for _, _, title, linum in evidence_set]

    contained = list()
    for pred_ev_instance in pred_ev:
        if pred_ev_instance in actual_evidences:
            contained.append(True)
        else:
            contained.append(False)

    return contained


def convert(booleans):
    return ["O" if boolean else "X" for boolean in booleans]


def resolve_evidences(evidences, t2l2s, actual=True):
    """
    actual: [[[id, id, "title", linum], [id, id, "title", linum]], [[id, id, "title", linum]], ... ]
    not actual ( predicted ):
    [(title, linum), (title, linum), ...]
    """
    if actual:
        evidence_sentences = list()
        for evidence_set in evidences:
            evidence_linum = [(title, linum)
                              for _, _, title, linum in evidence_set]
            for title, _ in evidence_linum:
                if title is not None and title not in t2l2s:
                    pass
                    # print(title, len(evidences), evidences)

            evidence_linum = [(title, linum)
                              for _, _, title, linum in evidence_set
                              if title in t2l2s]

            # continue if evidence_linum is empty
            if not evidence_linum:
                evidence_sentences.append(["**Not Found**"])
                continue

            sentences = get_evidence_sentence_list(evidence_linum, t2l2s)

            evidence_sentences.append(sentences)

    else:
        evidence_sentences = get_evidence_sentence_list(evidences, t2l2s)

    return evidence_sentences


import re
__pattern = re.compile('\w+|[^\w\s]')


def tokenize(text, pattern=__pattern):
    return __pattern.findall(text)


def find_unk(sentence, vocab):
    tokens = set(tokenize(sentence))
    unk_tokens = [token for token in tokens if token not in vocab]
    return unk_tokens


def save_wrong_instances(actual_file, predicted_labels_file,
                         predicted_evidence_file, out_file):
    label_predictions = read_jsonl(predicted_labels_file)
    ev_predictions = read_jsonl(predicted_evidence_file)
    actual = read_jsonl(actual_file)

    all_titles = list()
    for ev_pred, act in zip(ev_predictions, actual):
        ev_titles = [title for title, _ in ev_pred["predicted_sentences"]]
        act_titles = [
            title for evidence_set in act["evidence"]
            for _, _, title, _ in evidence_set
        ]
        titles = ev_titles + act_titles
        all_titles.extend(titles)

    print("loading wiki data...")
    t2jnum = titles_to_jsonl_num(
        wikipedia_dir=abs_path("data/wiki-pages/wiki-pages/"),
        doctitles=abs_path("data/doctitles"))
    t2l2s = load_doclines(all_titles, t2jnum)

    counter = 0
    observations = list()

    print("loading vocabulary list...")
    import pickle
    with open("vocab_list.db", "rb") as f:
        vocab = pickle.load(f)

    pos_counter = 0
    neg_counter = 0
    print("processing predictions...")
    for label_pred, ev_pred, act in tqdm(
            zip(label_predictions, ev_predictions, actual)):
        actual_label = act["label"]
        assert actual_label == label_pred["actual"]

        pred_label = label_pred["predicted"]
        if pred_label != actual_label:
            continue

        counter += 1
        actual_ev = act["evidence"]
        pred_labels = label_pred["prediction_list"]
        pred_ev = ev_pred["predicted_sentences"]
        pred_ev_sent = resolve_evidences(pred_ev, t2l2s, actual=False)

        claim = act["claim"]
        ev_contained = convert(compare_evidences(actual_ev, pred_ev))
        actual_ev_sent = resolve_evidences(actual_ev, t2l2s)
        assert not (actual_label != "NOT ENOUGH INFO"
                    and len(actual_ev_sent) != len(actual_ev))

        pred_sentence = " ".join(pred_ev_sent)
        ac_sentence = " ".join(sent for sentences in actual_ev_sent
                               for sent in sentences
                               if sent != "**Not Found**")
        unk_words = find_unk(pred_sentence + " " + ac_sentence, vocab)

        if pred_label == actual_label:
            pos_counter += 1
        else:
            neg_counter += 1

        # overwrite when label is NEI
        if actual_label == "NOT ENOUGH INFO":
            ev_contained = ["-" for e in ev_contained]

        # # skip for NEI or no correct evidence.
        # if ev_contained == ["X"] * 5 and ev_contained != ["-"] * 5:
        #     continue

        label_pred_ev = [
            "<{}> <{}> {}".format(label, contained, ev)
            for label, contained, ev in zip(
                shorten_labels(pred_labels), ev_contained, pred_ev)
        ]
        actual_ev = ev_pred["evidence"]

        observations.append({
            "id": act["id"],
            "claim": claim,
            "predicted_evidences": label_pred_ev,
            "predicted_sentences": pred_ev_sent,
            "predicted_label": pred_label,
            "actual_evidence": actual_ev,
            "actual_sentences": actual_ev_sent,
            "actual_label": actual_label,
            "unk_words": unk_words
        })

    random.shuffle(observations)
    save_jsonl_pretty_print(observations, out_file)
    print("pos_counter", pos_counter)
    print("neg_counter", neg_counter)
    print("wrong labels:", counter)


def save_submission_file(predictions, path):
    out = list()
    for pred in predictions:
        out.append({
            "id": pred["id"],
            "predicted_label": pred["predicted_label"],
            "predicted_evidence": pred["predicted_evidence"]
         })
    save_jsonl(out, path)


def save_jsonl_pretty_print(dictionaries, path, print_message=True):
    """save jsonl file from list of dictionaries
    """
    if os.path.exists(path):
        raise OSError("file {} already exists".format(path))

    if print_message:
        print("saving at {}".format(path))
    with open(path, "a") as out_file:
        for instance in dictionaries:
            out_file.write(str(json.dumps(instance, indent=4)) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predicted_labels", type=str)
    parser.add_argument("--predicted_evidence", type=str)
    parser.add_argument("--actual", type=str)
    parser.add_argument("--out_file", type=str)

    args = parser.parse_args()

    ids = []
    predicted_labels = []
    predicted_evidence = []
    actual = []

    with open(args.predicted_labels, "r") as predictions_file:
        for line in predictions_file:
            predicted_labels.append(json.loads(line)["predicted"])

    with open(args.predicted_evidence, "r") as predictions_file:
        for line in predictions_file:
            predicted_evidence.append(json.loads(line)["predicted_sentences"])
            id.append(json.loads(line)["id"])

    with open(args.actual, "r") as actual_file:
        for line in actual_file:
            actual.append(json.loads(line))

    predictions = []
    for id, ev, label in zip(ids, predicted_evidence, predicted_labels):
        predictions.append({
            "id": id,
            "predicted_evidence": ev,
            "predicted_label": label
        })

    print_confusion_mat(predictions, actual)
    save_wrong_instances(args.actual, args.predicted_labels,
                         args.predicted_evidence, args.out_file)
