import json
import random
import re
import os
import sys
from util import abs_path
from tqdm import tqdm

def save_jsonl(dictionaries, path, print_message=True, skip_if_exists=False):
    """save jsonl file from list of dictionaries
    """
    if os.path.exists(path):
        if not skip_if_exists:
            raise OSError("file {} already exists".format(path))
        else:
            print("CAUTION: skip saving (file {} already exists)".format(path))
            return

    if print_message:
        print("saving at {}".format(path))
    with open(path, "a") as out_file:
        for instance in dictionaries:
            out_file.write(json.dumps(instance) + "\n")

def read_jsonl(path):
    with open(path, "r") as in_file:
        out = [json.loads(line) for line in in_file]

    return out

def load_doc_lines(docs=dict(),t2jnum=dict(),wikipedia_dir="data/wiki-pages/wiki-pages/"):
    """Returns a dictionary from titles to line numbers to line text.
    Args
    docs: {cid: [(title, score),  ...], ...}

    Input is a dictionary from claim ids to titles and line numbers, 
    and a lookup from titles to filenumbers.
    """
    doclines=dict()
    jnums=dict()
    titles=set()
    for cid in docs:
        for title, score in docs[cid]:
            doclines[title]=dict()
            titles.add(title)
            jnum,point=t2jnum[title]
            if jnum not in jnums:
                jnums[jnum]=set()
            jnums[jnum].add(point)
    for jnum in tqdm(jnums):
        points=sorted(list(jnums[jnum]))
        fname=wikipedia_dir+"wiki-"+jnum+".jsonl"
        with open(fname, "r") as f:
            for point in points:
                f.seek(point,0)
                line=f.readline()
                data=json.loads(line.rstrip("\n"))
                title=data["id"]
                lines=data["lines"]
                assert title in titles
                if title in titles and lines != "":
                    for l in lines.split("\n"):
                        fields=l.split("\t")
                        if fields[0].isnumeric():
                            l_id=int(fields[0])
                            l_txt=fields[1]
                            doclines[title][l_id]=l_txt
    return doclines
        
            
def load_doclines(titles, t2jnum, filtering=True):
    """load all lines for provided titles
    Args
    titles: list of titles
    """
    if filtering:
        filtered_titles = [title for title in titles if title in t2jnum]
        print("mismatch: {} / {}".format(len(titles) - len(filtered_titles), len(titles)))
        titles = filtered_titles

    return load_doc_lines({"dummy_id" : [(title, "dummy_linum") for title in titles]}, t2jnum, wikipedia_dir=abs_path("data/wiki-pages/wiki-pages/"))

def titles_to_jsonl_num(wikipedia_dir="data/wiki-pages/wiki-pages/", doctitles="data/doctitles"):
    """
    Returns a dictionary lookup from document titles to jsonl filenumbers and pointers.
    Saves the lookup in data/doctitles to speed up subsequent passes.
    """
    t2jnum=dict()
    try:
        with open(doctitles, "r") as f:
            for line in f:
                fields=line.rstrip("\n").split("\t")
                title=fields[0]
                jnum=fields[1]
                point=int(fields[2])
                t2jnum[title]=(jnum,point)
            if len(t2jnum) == 0:
                raise RuntimeError("doctitles file ({}) might be empty.".format(doctitles))
    except:
        with open(doctitles,"w") as w:
            for i in tqdm(range(1,110)):
                jnum="{:03d}".format(i)
                fname=wikipedia_dir+"wiki-"+jnum+".jsonl"
                with open(fname) as f:
                    point=f.tell()
                    line=f.readline()
                    while line:
                        data=json.loads(line.rstrip("\n"))
                        title=data["id"]
                        lines=data["lines"]
                        w.write(title+"\t"+jnum+"\t"+str(point)+"\n")
                        t2jnum[title]=(jnum,point)
                        point=f.tell()
                        line=f.readline()
    return t2jnum


def get_evidence_sentence_list(evidences, t2l2s, prependlinum=False, prependtitle=False):
    """lookup corresponding sentences and return list of sentences
    Args
    evidences: [(title, linum), ...]
    t2l2s: title2line2sentence <- output of load_doc_lines

    Returns
    list of evidence sentences
    """
    SEP = "#"
    def process_title(title):
        """ 'hoge_fuga_hoo' -> 'hoge fuga hoo' """
        return re.sub("_", " ", title)

    def maybe_prepend(title, linum):
        prep = list()
        if prependtitle:
            prep.append(title)
        if prependlinum:
            prep.append(str(linum))

        content = " {} ".format(SEP).join(prep)
        if prep:
            return "{0} {1} {0}".format(SEP, content)
        else:
            return content

    titles = [title for title, _ in evidences]
    linums = [linum for _, linum in evidences]

    return [ (maybe_prepend(process_title(title), linum) + " " + t2l2s[title][linum]).strip() for title, linum in zip(titles, linums)]


def load_wikipedia(wikipedia_dir="data/wiki-pages/wiki-pages/", howmany=99999):
    """
    Returns a list with in total 5,416,537 wikipedia article texts as elements.
    If one doesn't want to load all articles, one can use "howmany" to specify howmany files should be
    read (each containing 50000 articles). For example, to read only 100K articles, pick howmany=2.
    """
    all_texts = []
    print("loading wikipedia...")
    for filename in tqdm(sorted(os.listdir(wikipedia_dir))[:howmany]):
        with open(wikipedia_dir+filename, 'r') as openfile:
            some_texts = [json.loads(line)['text'] for line in openfile.readlines()]
        all_texts.extend(some_texts)
    print("Loaded", len(all_texts), "articles. Size (MB):", round(sys.getsizeof(all_texts)/1024/1024, 3))
    return all_texts

def get_label_set():
    label_set = {"SUPPORTS","REFUTES","NOT ENOUGH INFO"}
    return label_set


def load_split_trainset(dev_size:int):
    """
    Loads the full training set, splits it into preliminary train and dev set.
    This preliminary dev set is balanced.
    dev_size: size of dev set.
    """

    # load fever training data
    full_train = load_fever_train()

    positives = []
    negatives = []
    neutrals = []

    # sort dataset according to label.
    for example in full_train:
        example['id']
        label = example['label']
        if label == "SUPPORTS":
            positives.append(example)
        elif label == "REFUTES":
            negatives.append(example)
        elif label == "NOT ENOUGH INFO":
            neutrals.append(example)
        else:
            raise AssertionError("Bad label!", label)

    # shuffle examples for each label.
    random.seed(42)
    random.shuffle(positives)
    random.shuffle(negatives)
    random.shuffle(neutrals)

    # split off a preliminary dev set, balanced across each of the three classes
    size = int(dev_size/3)
    preliminary_dev_set = positives[:size] + negatives[:size] + neutrals[:size]

    # the remaining data will be the new training data
    train_set = positives[size:] + negatives[size:] + neutrals[size:]

    # shuffle order of examples
    random.shuffle(preliminary_dev_set)
    random.shuffle(train_set)

    return train_set, preliminary_dev_set


def load_fever_train(path="data/train.jsonl", howmany=999999):
    """
    Reads the Fever Training set, returns list of examples.
    howmany: how many examples to load. Useful for debugging.
    """
    data = []
    with open(path, 'r') as openfile:
        for iline, line in enumerate(openfile.readlines()):
            data.append(json.loads(line))
            if iline+1 >= howmany:
                break
    return data

def load_paper_dataset(train=abs_path("data/train.jsonl"), dev=abs_path("data/dev.jsonl")):
    """Reads the Fever train/dev set used on the paper.
    """
    train_ds = load_fever_train(path=train, howmany=9999999999)
    dev_ds = load_fever_train(path=dev, howmany=9999999999)
    return train_ds, dev_ds



if __name__ == "__main__":
    # load fever training data
    fever_data = load_fever_train(howmany=20)
    print(len(fever_data))

    # load train and split-off dev set of size 9999.
    train, dev = load_split_trainset(9999)
    print(len(train))
    print(len(dev))


    for sample in train[:3]:
        print(sample)

    # s = Sample(train[0])
    # print(s.__dict__)
