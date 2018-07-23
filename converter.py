"""
convert FEVER dataset format to SNLI format for makeing it work on jack
"""
import os
import re
import argparse
import json
from collections import Counter
from tqdm import tqdm
from util import abs_path
from fever_io import titles_to_jsonl_num, load_doclines, read_jsonl, save_jsonl, get_evidence_sentence_list
from analyse import compare_evidences
current_dir = os.path.dirname(os.path.abspath(__file__))


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


def snli_format(id, pair_id, label, evidence, claim):
    return {
        "captionID": id,
        "pairID": pair_id,
        "gold_label": label,
        "sentence1": evidence,
        "sentence2": claim
    }

def sampling(converted_instances):
    """to avoid imbalance of data, sample nei labels according to the # of non_nei labels
    ["NEI", "NEI", "SUP", "SUP", "NEI"]
    ->
    ["NEI", "NEI", "SUP", "SUP"]
    should be refactored
    """
    filtered_labels = list()
    for instance in converted_instances:
        filtered_labels.append(instance["gold_label"])

    n_neis = Counter(filtered_labels)[convert_label("NOT ENOUGH INFO")]
    n_instances = len(converted_instances)
    n_samples = (n_instances - n_neis)

    if n_samples == 0:
        return converted_instances

    sampled_instances = list()
    for filtered_label, instance in zip(filtered_labels, converted_instances):
        if filtered_label != convert_label("NOT ENOUGH INFO"):
            sampled_instances.append(instance)

    sample_count = 0
    for filtered_label, instance in zip(filtered_labels, converted_instances):
        if filtered_label != convert_label("NOT ENOUGH INFO"):
            continue
        sampled_instances.append(instance)
        sample_count += 1
        if sample_count == n_samples:
            break

    return sampled_instances


def _convert_instance(instance, t2l2s, prependlinum, prependtitle, use_ir_prediction, n_sentences):
    """convert single instance to either one or multiple instances
    Args
    instance: instance of FEVER dataset.
    t2l2s: output of titles_to_jsonl_num

    Returns
    list of converted instances
    """

    def _evidence_format(evidences):
        """return evidence sentence from (possibly) multiple evidence sentences"""
        return " ".join(evidences)

    converted_instances = list()
    # assert instance["evidence"] == [[[hoge, hoge, title, linum], [hoge, hoge, title, linum]], [[..],[..],..], ...]
    if use_ir_prediction:
        evidence_linum = [[title, linum] for title, linum in instance["predicted_sentences"][:n_sentences]
                          if title in t2l2s]
        contained_flags = compare_evidences(instance["evidence"], evidence_linum)

        for eidx, ((title, linum), contained) in enumerate(zip(evidence_linum, contained_flags)):
            label = instance["label"] if ( instance["label"] != "NOT ENOUGH INFO" and contained ) else "NOT ENOUGH INFO"

            converted_instances.append(
                snli_format(
                    id="{}-{}".format(instance["id"], str(eidx)),
                    pair_id="{}-{}".format(instance["id"], str(eidx)),
                    label=convert_label(label),
                    evidence=_evidence_format(
                        get_evidence_sentence_list(
                            [(title, linum)], t2l2s, prependlinum=prependlinum, prependtitle=prependtitle)),
                    claim=instance["claim"]))

        converted_instances = sampling(converted_instances)

    else:
        for eidx, evidence_set in enumerate(instance["evidence"]):
            evidence_linum = [(title, linum)
                              for _, _, title, linum in evidence_set
                              if title in t2l2s]

            # continue if evidence_linum is empty
            if not evidence_linum:
                continue
            converted_instances.append(
                snli_format(
                    id="{}-{}".format(instance["id"], str(eidx)),
                    pair_id="{}-{}".format(instance["id"], str(eidx)),
                    label=convert_label(instance["label"]),
                    evidence=_evidence_format(
                        get_evidence_sentence_list(
                            evidence_linum, t2l2s, prependlinum=prependlinum, prependtitle=prependtitle)),
                    claim=instance["claim"]))
    return converted_instances


def convert(instances, prependlinum=False, prependtitle=False, use_ir_prediction=False, n_sentences=5):
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
        if instance["label"] == "NOT ENOUGH INFO":
            evidences = instance["predicted_sentences"][:n_sentences]
            # assert evidences == [(title, linum), (title, linum), ...]

            # change its shape to the normal evidence format
            evidences = [[["dummy", "dummy", title, linum]]
                         for title, linum in evidences]
            instance["evidence"] = evidences

        if use_ir_prediction:
            titles = [title for title, _ in instance["predicted_sentences"][:n_sentences]]
        else:
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
        converted_instances.extend(
            _convert_instance(
                instance, t2l2s, prependlinum=prependlinum, prependtitle=prependtitle, use_ir_prediction=use_ir_prediction, n_sentences=n_sentences))

    return converted_instances


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("src")
    parser.add_argument("tar")
    parser.add_argument("--use_ir_pred", action="store_true")
    parser.add_argument("--n_sentences", default=5, type=int)
    parser.add_argument("--prependlinum", action="store_true")
    parser.add_argument("--prependtitle", action="store_true")
    parser.add_argument("--convert_test", action="store_true")
    # parser.add_argument("--testset", help="turn on when you convert test data", action="store_true")
    args = parser.parse_args()
    print(args)

    if args.convert_test:
        test_in = '''[{"id": 15812, "verifiable": "VERIFIABLE", "label": "REFUTES", "claim": "Peggy Sue Got Married is a Egyptian film released in 1986.", "evidence": [[[31205, 37902, "Peggy_Sue_Got_Married", 0], [31205, 37902, "Francis_Ford_Coppola", 0]], [[31211, 37908, "Peggy_Sue_Got_Married", 0]]], "predicted_pages": ["Peggy_Sue_Got_Married_-LRB-musical-RRB-", "Peggy_Sue_Got_Married_-LRB-song-RRB-", "Peggy_Sue_Got_Married", "Peggy_Sue", "Peggy_Sue_-LRB-band-RRB-"], "predicted_sentences": [["Peggy_Sue_Got_Married", 0], ["Peggy_Sue_Got_Married_-LRB-musical-RRB-", 0], ["Peggy_Sue_Got_Married_-LRB-song-RRB-", 0], ["Peggy_Sue", 0], ["Peggy_Sue_Got_Married_-LRB-musical-RRB-", 2]]}, {"id": 229289, "verifiable": "NOT VERIFIABLE", "label": "NOT ENOUGH INFO", "claim": "Neal Schon was named in 1954.", "evidence": [[[273626, null, null, null]]], "predicted_pages": ["Neal_Schon", "Neal", "Named", "Was_-LRB-Not_Was-RRB-", "Was"], "predicted_sentences": [["Neal_Schon", 0], ["Neal_Schon", 6], ["Neal_Schon", 5], ["Neal_Schon", 1], ["Neal_Schon", 2]]}, {"id": 15711, "verifiable": "VERIFIABLE", "label": "SUPPORTS", "claim": "Liverpool F.C. was valued at $1.55 billion at one point.", "evidence": [[[31112, 37788, "Liverpool_F.C.", 11]]], "predicted_pages": ["Liverpool_F.C.", "Liverpool_F.C._-LRB-Montevideo-RRB-", "Liverpool_F.C._-LRB-Superleague_Formula_team-RRB-", "Liverpool_F.C._-LRB-disambiguation-RRB-", "Liverpool"], "predicted_sentences": [["Liverpool_F.C.", 11], ["Liverpool", 0], ["Liverpool", 9], ["Liverpool", 10], ["Liverpool", 8]]}]'''

        print("input:\n", test_in)
        fever_format = json.loads(test_in)
        snli_format_instances = convert(fever_format, prependlinum=args.prependlinum, prependtitle=args.prependtitle, use_ir_prediction=args.use_ir_pred, n_sentences=args.n_sentences)
        print("\noutput:\n", json.dumps(snli_format_instances, indent=4))

    else:
        if os.path.exists(args.tar):
            print("WARNING: file {} alreadly exists".format(args.tar))
        keyerr_count = 0

        instances = read_jsonl(args.src)
        snli_format_instances = convert(instances, prependlinum=args.prependlinum, prependtitle=args.prependtitle, use_ir_prediction=args.use_ir_pred, n_sentences=args.n_sentences)
        save_jsonl(snli_format_instances, args.tar, skip_if_exists=True)
