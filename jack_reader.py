import argparse
from util import abs_path
from converter import titles_to_jsonl_num, convert_label
from fever_io import load_doclines, read_jsonl, save_jsonl, get_evidence_sentence, get_evidence_sentence_list
from jack import readers
from jack.core import QASetting


def read_ir_result(path):
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
        instance["evidence"] = get_evidence_sentence_list(
            instance["predicted_sentences"], t2l2s)

    return instances


def aggregate_preds(pred):
    """return the most popular verdict
    """
    vote = dict()
    for rank, pred in enumerate(pred):
        pred = pred[0]
        if pred.text not in vote:
            vote[pred.text] = 1
        else:
            vote[pred.text] += 1

    popular_verdict = max(vote, key=vote.get)
    score = vote[popular_verdict]
    pred_from_top_evidence = pred[0][0].text

    return (popular_verdict, score, pred_from_top_evidence)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("read claim/evidence and output verdict")
    parser.add_argument(
        "in_file",
        help="input file path for rte (e.g., dev.sentences.p5.s5.jsonl)")
    parser.add_argument("out_file", help="output prediction file")
    parser.add_argument(
        "--saved_reader", help="path to saved reader directory")
    parser.add_argument("--cutoff", default=None, help="if not None, model only reads specified number of evidences")
    args = parser.parse_args()

    dam_reader = readers.reader_from_file(args.saved_reader)

    if args.cutoff:
        cutoff = int(args.cutoff)
    else:
        cutoff = None

    nli_settings = list()
    results = list()
    instances = read_ir_result(args.in_file)
    claims = [instance["claim"] for instance in instances]
    evidences = [instance["evidence"] for instance in instances]
    actual = [instance["label"]for instance in instances]
    for claim, evidence_list in zip(claims, evidences):
        preds = list()
        for evidence in evidence_list:
            preds.append(dam_reader([QASetting(question=claim, support=evidence)]))
            
        prediction, score, pred_from_top_ev = aggregate_preds(preds)
        results.append({
            "actual": actual,
            "predicted":
            convert_label(prediction, inverse=True),
            "score":
            score,
            "pred_from_top_evidence":
            convert_label(pred_from_top_ev, inverse=True)
        })
    save_jsonl(results, abs_path(args.out_file))
