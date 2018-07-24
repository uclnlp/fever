import argparse
from util import abs_path
from converter import titles_to_jsonl_num, convert_label
from fever_io import load_doclines, read_jsonl, save_jsonl, get_evidence_sentence_list
from tqdm import tqdm
from jack import readers
from jack.core import QASetting

# make everything deterministic
import random
import numpy as np
import tensorflow as tf
tf.set_random_seed(1)
np.random.seed(1)
random.seed(1)


def read_ir_result(path, n_sentences=5, prependlinum=False, prependtitle=False, concatev=False):
    """
    Returns
    instances: list of dictionary
    update instance['predicted_sentences'] with list of evidences (list of str)
    """
    short_evidences_counter = 0
    instances = read_jsonl(path)
    # only read n_sentences
    for instance in instances:
        if len(instance["predicted_sentences"]) < n_sentences:
            short_evidences_counter += 1
        instance["predicted_sentences"] = instance["predicted_sentences"][:n_sentences]
    print("short_evidences: {} / {}".format(short_evidences_counter, len(instances)))

    t2jnum = titles_to_jsonl_num(
        wikipedia_dir=abs_path("data/wiki-pages/wiki-pages/"),
        doctitles=abs_path("data/doctitles"))
    titles = list()

    # make list of titles
    for instance in instances:
        titles.extend([title for title, _ in instance["predicted_sentences"]])

    # load title2line2sentences
    t2l2s = load_doclines(titles, t2jnum)

    for instance in instances:
        if concatev:
            instance["evidence"] = [" ".join(get_evidence_sentence_list(
                instance["predicted_sentences"], t2l2s, prependlinum=prependlinum, prependtitle=prependtitle))]
        else:
            instance["evidence"] = get_evidence_sentence_list(
                instance["predicted_sentences"], t2l2s, prependlinum=prependlinum, prependtitle=prependtitle)

    return instances


def aggregate_preds(prediction, only_use_topev=False):
    """return the most popular verdict
    """
    vote = dict()
    pred_list = [pred[0].text for pred in prediction]
    scores = [float(pred[0].score) for pred in prediction]

    for rank, pred in enumerate(pred_list):
        if pred not in vote:
            vote[pred] = 1
        else:
            vote[pred] += 1

    supports = convert_label("SUPPORTS")
    refutes = convert_label("REFUTES")
    nei = convert_label("NOT ENOUGH INFO")

    # believe more-likely evidence if both supports and refutes appears in the pred_list
    if supports in vote and refutes in vote:
        for pred in pred_list:
            if pred in [supports, refutes]:
                final_verdict = pred
                break
    elif supports in vote:
        final_verdict = supports
    elif refutes in vote:
        final_verdict = refutes
    else:
        final_verdict = nei

    if only_use_topev:
        final_verdict = pred_list[0]

    return (final_verdict, scores, pred_list)


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
    for pointer in tqdm(range(0, len(all_settings), batch_size)):
        batch_settings = all_settings[pointer: pointer + batch_size]
        n_settings = [len(settings_) for settings_ in batch_settings]
        preds_list.extend(reshape(reader(flatten(batch_settings)), n_settings))
    return preds_list


def save_predictions(instances, preds_list, path):
    store = list()
    for instance, preds in zip(instances, preds_list):
        id = instance["id"]
        label = instance["label"]
        claim = instance["claim"]
        pred_sents = instance["evidence"] # refer to read_ir_result
        pred_labels = [pred[0].text for pred in preds]
        scores = [float(pred[0].score) for pred in preds]
        dic = {
            "id": id,
            "label": label,
            "scores": scores,
            "predicted_labels": [convert_label(pred_label, inverse=True) for pred_label in pred_labels],
            "claim": claim,
            "predicted_sentences": pred_sents
        }

        # scores of ir part
        if "scored_sentences" in instance:
            dic["ev_scores"] = instance["scored_sentences"]

        store.append()

    save_jsonl(store, path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("read claim/evidence and output verdict")
    parser.add_argument(
        "in_file",
        help="input file path for rte (e.g., dev.sentences.p5.s5.jsonl)")
    parser.add_argument("save_preds", help="specify file name to save prediction")
    parser.add_argument(
        "--saved_reader", help="path to saved reader directory")
    parser.add_argument(
        "--concatev", action="store_true", help="concat evidences")
    parser.add_argument(
        "--prependlinum", action="store_true", help="prepend linum when perform get_evidence_sentence_list")
    parser.add_argument(
        "--prependtitle", action="store_true", help="prepend title when perform get_evidence_sentence_list")
    parser.add_argument("--only_use_topev", action="store_true", help="only use top evidence for prediction")
    parser.add_argument("--n_sentences", type=int, default=5, help="how many sentences to read for prediction")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size for inference")
    args = parser.parse_args()

    print("loading reader from file:", args.saved_reader)
    dam_reader = readers.reader_from_file(args.saved_reader, dropout=0.0)

    results = list()
    preds_length = list()
    all_settings = list()
    instances = read_ir_result(args.in_file, n_sentences=args.n_sentences, prependlinum=args.prependlinum, prependtitle=args.prependtitle, concatev=args.concatev)
    for instance in instances:
        evidence_list = instance["evidence"]
        claim = instance["claim"]
        settings = [QASetting(question=claim, support=[evidence]) for evidence in evidence_list]
        all_settings.append(settings)

    preds_list = predict(dam_reader, all_settings, args.batch_size)
    save_predictions(instances, preds_list, path=args.save_preds)
