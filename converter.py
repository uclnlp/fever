"""
convert FEVER dataset format to SNLI format for makeing it work on jack
"""
import os
import argparse
import json
from fever_io import titles_to_jsonl_num, load_doc_lines
current_dir = os.path.dirname(os.path.abspath(__file__))

def get_abs_path(file):
    return os.path.join(current_dir, file)

def get_evidence_sentence(pointers, t2jnum):
    """
    pointers: [(title, linum), ...]
    """
    titles = [title for title, _ in pointers]
    linums = [linum for _, linum in pointers]
    evidence_instances = {(title, "dummy_linum") for title in titles}

    doclines = load_doc_lines(docs={"dummy_id": evidence_instances}, t2jnum=t2jnum, wikipedia_dir=get_abs_path("data/wiki-pages/wiki-pages/"))
    evidences = list()
    for title, linum in zip(titles, linums):
        # if linum < 0 (which means NEI) then just use linum 0
        if linum < 0:
            linum = 0
        evidences.append(doclines[title][linum])

    return " ".join(evidences)



def load_evidence(evidences, t2jnum):
    """
    Args:
    evidences: (evidence part of wiki-data)
    [
        [
            [<annotation_id>, <evidence_id>, "wiki-title", <line number>]
        ],
        [
            [<annotation_id>, <evidence_id>, "wiki-title", <line number>]
        ]
    ]
    Return:
    evidence sentences jointed with space
    """
    titles = list()
    linums = list()
    for evidence in evidences[0]:
        _, _, title, linum = evidence
        titles.append(title)
        linums.append(linum)
    evidence_instances = {(title, "dummy_linum") for title in titles}

    doclines = load_doc_lines(docs={"dummy_id": evidence_instances}, t2jnum=t2jnum)
    evidences = list()
    for title, linum in zip(titles, linums):
        # if linum < 0 (which means NEI) then just use linum 0
        if linum < 0:
            linum = 0
        evidences.append(doclines[title][linum])

    return " ".join(evidences)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("src")
    parser.add_argument("tar")
    # parser.add_argument("--testset", help="turn on when you convert test data", action="store_true")
    args = parser.parse_args()

    keyerr_count = 0
    t2jnum = titles_to_jsonl_num(wikipedia_dir=get_abs_path("data/wiki-pages/wiki-pages/"), doctitles=get_abs_path("data/doctitles"))
    with open(args.src) as f:
        for line in f:
            instance = json.loads(line.strip())

            id = instance["id"]
            pair_id = id
            original_label = instance["label"]
            if original_label == "SUPPORTS":
                label = "entailment"
            elif original_label == "REFUTES":
                label = "contradiction"
            elif original_label == "NOT ENOUGH INFO":
                label = "neutral"
            claim = instance["claim"]
            try:
                evidence = load_evidence(instance["evidence"], t2jnum)
            except KeyError:
                keyerr_count += 1
                continue

            with open(args.tar, "a") as outfile:
                snli_format = {"captionID": id, "pairID": pair_id, "gold_label": label, "sentence1": evidence, "sentence2": claim}
                outfile.write(json.dumps(snli_format) + "\n")

    print("keyerror count:", keyerr_count)
