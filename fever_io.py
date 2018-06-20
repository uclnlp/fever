import json
import random
import os
import sys
from util import abs_path
from tqdm import tqdm

def load_doc_lines(docs=dict(),t2jnum=dict(),wikipedia_dir="data/wiki-pages/wiki-pages/"):
    """
    Returns a dictionary from titles to line numbers to line text.
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
        with open(fname) as f:
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
        
            

def titles_to_jsonl_num(wikipedia_dir="data/wiki-pages/wiki-pages/", doctitles="data/doctitles"):
    """
    Returns a dictionary lookup from document titles to jsonl filenumbers and pointers.
    Saves the lookup in data/doctitles to speed up subsequent passes.
    """
    t2jnum=dict()
    try:
        with open(doctitles) as f:
            for line in f:
                fields=line.rstrip("\n").split("\t")
                title=fields[0]
                jnum=fields[1]
                point=int(fields[2])
                t2jnum[title]=(jnum,point)
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
                        if lines != "":
                            w.write(title+"\t"+jnum+"\t"+str(point)+"\n")
                        t2jnum[title]=(jnum,point)
                        point=f.tell()
                        line=f.readline()
    return t2jnum


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

def load_paper_dataset():
    """Reads the Fever train/dev set used on the paper.
    """
    train_ds = load_fever_train(path=abs_path("data/train.jsonl"), howmany=9999999999)
    dev_ds = load_fever_train(path=abs_path("data/dev.jsonl"), howmany=9999999999)
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
