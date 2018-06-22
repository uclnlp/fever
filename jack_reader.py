import argparse
from util import abs_path
from converter import titles_to_jsonl_num, convert_label
from fever_io import load_doclines, read_jsonl, save_jsonl, get_evidence_sentence
from jack import readers
from jack.core import QASetting


def read_ir_result(path, cutoff):
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
        instance["evidence"] = get_evidence_sentence(
            instance["predicted_sentences"], t2l2s, cutoff)

    return instances


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

    pred = list()
    dam_reader = readers.reader_from_file(args.saved_reader)

    if args.cutoff:
        cutoff = int(args.cutoff)
    else:
        cutoff = None

    for instance in read_ir_result(args.in_file, cutoff):
        claim = instance["claim"]
        evidence = instance["evidence"]
        # question: hypothesis, support: [premise]
        nli_setting = QASetting(question=claim, support=[evidence])
        prediction = dam_reader([nli_setting])
        pred.append({
            "actual":
            instance["label"],
            "predicted":
            convert_label(prediction[0][0].text, inverse=True)
        })
    save_jsonl(pred, abs_path(args.out_file))
