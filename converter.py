"""
convert FEVER dataset format to SNLI format for makeing it work on jack
"""
import os
import argparse
import json
from tqdm import tqdm
from util import abs_path
from fever_io import titles_to_jsonl_num, load_doc_lines, load_doclines, read_jsonl, save_jsonl, get_evidence_sentence
current_dir = os.path.dirname(os.path.abspath(__file__))


def get_abs_path(file):
    return os.path.join(current_dir, file)


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

    doclines = load_doc_lines(
        docs={"dummy_id": evidence_instances}, t2jnum=t2jnum)
    evidences = list()
    for title, linum in zip(titles, linums):
        # if linum < 0 (which means NEI) then just use linum 0
        if linum < 0:
            linum = 0
        evidences.append(doclines[title][linum])

    return " ".join(evidences)


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


def convert_to_snli_format(instances, wikipedia_dir, doctitles):
    t2jnum = titles_to_jsonl_num(wikipedia_dir, doctitles)
    keyerr_count = 0
    out = list()

    for instance in instances:
        cid = instance["id"]
        pair_id = cid
        label = convert_label(instance["label"])
        claim = instance["claim"]
        try:
            evidence = load_evidence(instance["evidence"], t2jnum)
        except KeyError:
            keyerr_count += 1
            continue
        snli_instance = {
            "captionID": id,
            "pairID": pair_id,
            "gold_label": label,
            "sentence1": evidence,
            "sentence2": claim
        }
        out.append(snli_instance)
    return out


def snli_format(id, pair_id, label, evidence, claim):
    return {
        "captionID": id,
        "pairID": pair_id,
        "gold_label": label,
        "sentence1": evidence,
        "sentence2": claim
    }


def convert(instances, nei_sampling=True):
    """convert FEVER format to jack SNLI format
    Arg
    instances: list of dictionary of FEVER format

    Returns
    instances: list of dictionary of jack SNLI format
    """
    # get all titles and load t2l2s
    all_titles = list()

    # use "predicted_sentences" for NEI
    for instance in tqdm(instances, desc="process for NEI"):
        if nei_sampling and instance["label"] == "NOT ENOUGH INFO":
            evidences = instance["predicted_sentences"]
            # assert evidences == [(title, linum), (title, linum), ...]

            # change its shape to the normal evidence format
            evidences = [[["dummy", "dummy", title, linum]]
                         for title, linum in evidences]
            instance["evidence"] = evidences

        titles = [
            title for evidence_set in instance["evidence"]
            for _, _, title, _ in evidence_set
        ]
        all_titles.extend(titles)

    print("loading wiki data...")
    t2jnum = titles_to_jsonl_num(
        wikipedia_dir=abs_path("data/wiki-pages/wiki-pages/"),
        doctitles=abs_path("data/doctitles"))
    t2l2s = load_doclines(all_titles, t2jnum)

    converted_instances = list()
    for instance in tqdm(instances, desc="conversion"):
        # assert instance["evidence"] == [[[hoge, hoge, title, linum], [hoge, hoge, title, linum]], [[..],[..],..], ...]
        for eidx, evidence_set in enumerate(instance["evidence"]):
            evidence_linum = [(title, linum)
                              for _, _, title, linum in evidence_set if title in t2l2s]

            # continue if evidence_linum is empty
            if not evidence_linum:
                continue
            converted_instances.append(
                snli_format(
                    id="{}-{}".format(instance["id"], str(eidx)),
                    pair_id="{}-{}".format(instance["id"], str(eidx)),
                    label=convert_label(instance["label"]),
                    evidence=get_evidence_sentence(evidence_linum, t2l2s),
                    claim=instance["claim"]))

    return converted_instances


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("src")
    parser.add_argument("tar")
    parser.add_argument("--convert_test", action="store_true")
    # parser.add_argument("--testset", help="turn on when you convert test data", action="store_true")
    args = parser.parse_args()

    if args.convert_test:
        fever_format = json.loads('''
        [{"id": 15812, "verifiable": "VERIFIABLE", "label": "REFUTES", "claim": "Peggy Sue Got Married is a Egyptian film released in 1986.", "evidence": [[[31205, 37902, "Peggy_Sue_Got_Married", 0], [31205, 37902, "Francis_Ford_Coppola", 0]], [[31211, 37908, "Peggy_Sue_Got_Married", 0]]], "predicted_pages": ["Peggy_Sue_Got_Married_-LRB-musical-RRB-", "Peggy_Sue_Got_Married_-LRB-song-RRB-", "Peggy_Sue_Got_Married", "Peggy_Sue", "Peggy_Sue_-LRB-band-RRB-"], "predicted_sentences": [["Peggy_Sue_Got_Married", 0], ["Peggy_Sue_Got_Married_-LRB-musical-RRB-", 0], ["Peggy_Sue_Got_Married_-LRB-song-RRB-", 0], ["Peggy_Sue", 0], ["Peggy_Sue_Got_Married_-LRB-musical-RRB-", 2]]}, {"id": 229289, "verifiable": "NOT VERIFIABLE", "label": "NOT ENOUGH INFO", "claim": "Neal Schon was named in 1954.", "evidence": [[[273626, null, null, null]]], "predicted_pages": ["Neal_Schon", "Neal", "Named", "Was_-LRB-Not_Was-RRB-", "Was"], "predicted_sentences": [["Neal_Schon", 0], ["Neal_Schon", 6], ["Neal_Schon", 5], ["Neal_Schon", 1], ["Neal_Schon", 2]]}]
        ''')
        snli_format = convert(fever_format)
        print(snli_format)

    else:
        assert not os.path.exists(args.tar), "file {} alreadly exists".format(args.tar)
        keyerr_count = 0

        instances = read_jsonl(args.src)
        snli_format_instances = convert(instances)
        save_jsonl(snli_format_instances, args.tar)

    # with open(args.src) as f:
    #     for line in f:
    #         instance = json.loads(line.strip())

    #         id = instance["id"]
    #         pair_id = id
    #         original_label = instance["label"]
    #         if original_label == "SUPPORTS":
    #             label = "entailment"
    #         elif original_label == "REFUTES":
    #             label = "contradiction"
    #         elif original_label == "NOT ENOUGH INFO":
    #             label = "neutral"
    #         claim = instance["claim"]
    #         try:
    #             evidence = load_evidence(instance["evidence"], t2jnum)
    #         except KeyError:
    #             keyerr_count += 1
    #             continue

    #         with open(args.tar, "a") as outfile:
    #             snli_format = {"captionID": id, "pairID": pair_id, "gold_label": label, "sentence1": evidence, "sentence2": claim}
    #             outfile.write(json.dumps(snli_format) + "\n")

    # print("keyerror count:", keyerr_count)
