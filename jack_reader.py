import argparse
from util import abs_path
from converter import titles_to_jsonl_num, convert_label
from fever_io import load_doclines, read_jsonl, save_jsonl, get_evidence_sentence_list
from tqdm import tqdm
from jack import readers
from jack.core import QASetting


def read_ir_result(path, prependlinum=False, concatev=False):
    """
    Returns
    instances: list of dictionary
    update instance['predicted_sentences'] with list of evidences (list of str)
    """
    instances = read_jsonl(path)
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
                instance["predicted_sentences"], t2l2s, prependlinum=prependlinum))]
        else:
            instance["evidence"] = get_evidence_sentence_list(
                instance["predicted_sentences"], t2l2s, prependlinum=prependlinum)

    return instances


def aggregate_preds(prediction, only_use_topev=False):
    """return the most popular verdict
    """
    vote = dict()
    pred_list = [pred[0][0].text for pred in prediction]
    for rank, pred in enumerate(pred_list):
        pred = pred[0][0]
        if pred.text not in vote:
            vote[pred.text] = 1
        else:
            vote[pred.text] += 1

    supports = convert_label("SUPPORTS")
    refutes = convert_label("REFUTES")
    nei = convert_label("NOT ENOUGH INFO")
    if supports in vote and refutes in vote:
        final_verdict = max(vote, key=vote.get)
    elif supports in vote:
        final_verdict = supports
    elif refutes in vote:
        final_verdict = refutes
    else:
        final_verdict = nei

    pred_from_top_evidence = prediction[0][0][0].text

    if only_use_topev:
        final_verdict = pred_from_top_evidence

    score = vote[final_verdict]

    return (final_verdict, score, pred_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("read claim/evidence and output verdict")
    parser.add_argument(
        "in_file",
        help="input file path for rte (e.g., dev.sentences.p5.s5.jsonl)")
    parser.add_argument("out_file", help="output prediction file")
    parser.add_argument(
        "--saved_reader", help="path to saved reader directory")
    parser.add_argument(
        "--concatev", action="store_true", help="concat evidences")
    parser.add_argument(
        "--prependlinum", action="store_true", help="prepend linum when perform get_evidence_sentence_list")
    parser.add_argument("--only_use_topev", action="store_true", help="only use top evidence for prediction")
    args = parser.parse_args()

    print("loading reader from file:", args.saved_reader)
    dam_reader = readers.reader_from_file(args.saved_reader)

    results = list()
    for instance in tqdm(read_ir_result(args.in_file, prependlinum=args.prependlinum, concatev=args.concatev)):
        evidence_list = instance["evidence"]
        claim = instance["claim"]
        preds = list()
        for evidence in evidence_list:
            preds.append(dam_reader([QASetting(question=claim, support=[evidence])]))

        prediction, score, prediction_list = aggregate_preds(preds, args.only_use_topev)
        results.append({
            "actual": instance["label"],
            "predicted":
            convert_label(prediction, inverse=True),
            "score":
            score,
            "prediction_list":
            convert_label(prediction_list, inverse=True)
        })
    save_jsonl(results, abs_path(args.out_file))
