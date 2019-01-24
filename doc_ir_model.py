import os
import argparse
import numpy as np
from sklearn.linear_model import LogisticRegression
from util import edict, pdict, normalize_title, load_stoplist
from doc_ir import *
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import gazetteers, names
from collections import Counter
from fever_io import titles_to_jsonl_num, load_split_trainset, load_paper_dataset
import pickle
from tqdm import tqdm
from random import random, shuffle


class doc_ir_model:
    def __init__(self,phrase_features=phrase_features):
        self.model=LogisticRegression(C=100000000,solver="sag",max_iter=100000)
        featurelist=sorted(list(phrase_features("dummy",0,"dummy","dummy").keys()))
        self.f2v={f:i for i,f in enumerate(featurelist)}
    def fit(self,X,y):
        self.model.fit(X,y)
    def prob(self,x):
        return self.model.predict_proba(x)[0,1]
    def score_instance(self,phrase="dummy",start=0,title="dummy",claim="dummy"):
        x=np.zeros(shape=(1,len(self.f2v)),dtype=np.float32)
        self.process_instance(phrase,start,title,claim,0,x)
        return self.prob(x)
    def process_instance(self,phrase="dummy",start=0,title="dummy",claim="dummy",obsnum=0,array=np.zeros(shape=(1,1)),dtype=np.float32):
        features=phrase_features(phrase,start,title,claim)
        for f in features:
            array[obsnum,self.f2v[f]]=float(features[f])        
    def process_train(self,selected,train):
        obs=len(selected)*2
        nvars=len(self.f2v)
        X=np.zeros(shape=(obs,nvars),dtype=np.float32)
        y=np.zeros(shape=(obs),dtype=np.float32)
        obsnum=0
        for example in tqdm(train):
            cid=example["id"]
            if cid in selected:
                claim=example["claim"]
                for yn in selected[cid]:
                    [title,phrase,start]=selected[cid][yn]
                    self.process_instance(phrase,start,title,claim,obsnum,X)
                    y[obsnum]=float(yn)
                    obsnum+=1
        assert obsnum==obs
        return X,y 
        
    
        

def count_labels(train):
    """count labels for select_docs"""
    supports = 0
    refutes = 0
    print("counting labels...")
    for instance in tqdm(train):
        if instance["label"] == "NOT ENOUGH INFO":
            continue
        if instance["label"] == "SUPPORTS":
            supports += 1
        else:
            refutes += 1

    counts = {"SUPPORTS": supports, "REFUTES": refutes}
    print("result:", counts)
    return counts

def select_docs(train):
    samp_size=25000
    tots={"SUPPORTS": 0, "REFUTES": 0}
    sofar={"SUPPORTS": 0, "REFUTES": 0}
    try:
        with open("data/edocs.bin","rb") as rb:
            if os.path.getsize("data/edocs.bin") < 100:
                raise RuntimeError("Size of edocs.bin is too small. It may be an empty file.")
            edocs=pickle.load(rb)
    except:
        t2jnum=titles_to_jsonl_num()
        edocs=title_edict(t2jnum)
        with open("data/edocs.bin","wb") as wb:
            pickle.dump(edocs,wb)
    examples=Counter()
    id2titles=dict()
    for example in train:
        cid=example["id"]
        claim=example["claim"]
        l=example["label"]
        if l=='NOT ENOUGH INFO':
            continue
        all_evidence=[e for eset in example["evidence"] for e in eset]
        docs=set()
        for ev in all_evidence:
            evid  =ev[2]
            if evid != None:
                docs.add(evid)
        t2phrases=find_titles_in_claim(example["claim"],edocs)
        id2titles[cid]=t2phrases
        flag=False
        for title in t2phrases:
            if title in docs:
                flag=True
        if flag:
            tots[l]+=1
    selected=dict()
    for example in tqdm(train):
        yn=0
        cid=example["id"]
        l=example["label"]
        if l=='NOT ENOUGH INFO':
            continue
        all_evidence=[e for eset in example["evidence"] for e in eset]
        docs=set()
        for ev in all_evidence:
            evid  =ev[2]
            if evid != None:
                docs.add(evid)
        #t2phrases=find_titles_in_claim(example["claim"],edocs)
        t2phrases=id2titles[cid]
        for title in t2phrases:
            if title in docs:
                yn=1
        prob=(samp_size-sofar[l])/(tots[l])
        if yn==1 and random()<prob:
            titles=list(t2phrases.keys())
            shuffle(titles)
            flagy=False
            flagn=False
            for t in titles:
                if not flagy and t in docs:
                    ty=t
                    flagy=True
                if not flagn and t not in docs:
                    tn=t
                    flagn=True
                if flagy and flagn:
                    selected[cid]=dict()
                    for t,y_n in [(ty,1),(tn,0)]:
                        ps=t2phrases[t]
                        shuffle(ps)
                        p,s=ps[0]
                        selected[cid][y_n]=[t,p,s]
                    sofar[l]+=1
                    break
        if yn==1:
            tots[l]-=1
    with open("data/doc_ir_docs","w") as w:
        for cid in selected:
            for yn in selected[cid]:
                [t,p,s]=selected[cid][yn]
                w.write(str(cid)+"\t"+str(yn)+"\t"+t+"\t"+p+"\t"+str(s)+"\n")
    for l in sofar:
        print(l,sofar[l])
    return selected


def load_selected(fname="data/doc_ir_docs"):
    selected=dict()
    with open(fname) as f:
        for line in tqdm(f):
            fields=line.rstrip("\n").split("\t")
            cid=int(fields[0])
            yn=int(fields[1])
            t=fields[2]
            p=fields[3]
            s=int(fields[4])
            if cid not in selected:
                selected[cid]=dict()
            selected[cid][yn]=[t,p,s]
    return selected
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser("perform ir for document")
    parser.add_argument("--best", type=int, default=5, help="how many documents to retrieve")
    args = parser.parse_args()
    print(args)

    train, dev = load_paper_dataset()
    # train, dev = load_split_trainset(9999)
    try:
        with open("data/doc_ir_model.bin","rb") as rb:
            model=pickle.load(rb)
    except:
        try:
            selected=load_selected() 
        except:
            selected=select_docs(train)
        model=doc_ir_model()
        X,y=model.process_train(selected,train)
        model.fit(X,y)
        with open("data/doc_ir_model.bin","wb") as wb:
            if os.path.getsize("data/edocs.bin") < 100:
                raise RuntimeError("Size of edocs.bin is too small. It may be an empty file.")
            pickle.dump(model,wb)
    try:
        with open("data/edocs.bin","rb") as rb:
            edocs=pickle.load(rb)
    except:
        t2jnum=titles_to_jsonl_num()
        edocs=title_edict(t2jnum)
        with open("data/edocs.bin","wb") as wb:
            pickle.dump(edocs,wb)
    print(len(model.f2v))
    docs=doc_ir(dev,edocs,best=args.best,model=model)
    title_hits(dev,docs)
