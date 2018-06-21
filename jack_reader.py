import os
import argparse
import json
from util import abs_path, save_jsonl
from converter import get_evidence_sentence, titles_to_jsonl_num, convert_label
from jack import readers
from jack.core import QASetting
from jack.io.load import load_jack

def read_retrieval_output(path):
    t2jnum = titles_to_jsonl_num(wikipedia_dir=abs_path("data/wiki-pages/wiki-pages/"), doctitles=abs_path("data/doctitles"))
    out = list()
    with open(abs_path(path), "r") as f:
        for line in f:
            instance = json.loads(line.strip())
            out.append({
                "cid": instance["id"],
                "claim": instance["claim"],
                "evidence": get_evidence_sentence(instance["predicted_sentences"], t2jnum),
                "label": instance["label"]
            })

    return out


def predict(instances, t2jnum):
    """
    instances: each lines of dev.sentences.p5.s5.jsonl
    """
    pred = list()
    for instance in instances:
        evidence_sentences = get_evidence_sentence(instance["evidence"], t2jnum)
        nli_setting = QASetting(question=instance["claim"], support=[evidence_sentences])
        prediction = dam_reader([nli_setting])
        pred.append({"actual": instance["label"], "predicted": prediction[0][0].text})
    return pred

if __name__ == "__main__":
    parser = argparse.ArgumentParser("read claim/evidence and output verdict")
    parser.add_argument("in_file", help="input file path for rte (e.g., dev.sentences.p5.s5.jsonl)")
    parser.add_argument("out_file", help="output prediction file")
    parser.add_argument("--saved_reader", help="path to saved reader directory")
    args = parser.parse_args()

    pred = list()
    dam_reader = readers.reader_from_file(args.saved_reader)
    for instance in read_retrieval_output(args.in_file):
        cid = instance["cid"]
        claim = instance["claim"]
        evidence = instance["evidence"]
        # question: hypothesis, support: [premise]
        nli_setting = QASetting(question=claim, support=[evidence])
        prediction = dam_reader([nli_setting])
        pred.append({"actual": instance["label"], "predicted": convert_label(prediction[0][0].text)})
    save_jsonl(pred, abs_path(args.out_file))
