import argparse
import numpy as np
from sklearn.linear_model import LogisticRegression
from util import edict, pdict, normalize_title, load_stoplist
from doc_ir import doc_ir
from doc_ir_model import doc_ir_model
from line_ir import *
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import gazetteers, names
from collections import Counter
from fever_io import load_doc_lines, titles_to_jsonl_num, load_split_trainset, load_paper_dataset
import pickle
from tqdm import tqdm
from random import random, shuffle

class line_ir_model:
    def __init__(self,line_features=line_features):
        self.model=LogisticRegression(C=100000000,solver="sag",max_iter=100000)
        featurelist=sorted(list(line_features({"dummy"},"dummy",{"dummy"},"dummy",{"dummy"},0,0).keys()))
        self.f2v={f:i for i,f in enumerate(featurelist)}
    def fit(self,X,y):
        self.model.fit(X,y)
    def prob(self,x):
        return self.model.predict_proba(x)[0,1]
    def score_instance(self,c_toks={"dummy"},t="dummy",t_toks={"dummy"},line="dummy",l_toks={"dummy"},lid=0,tscore=0):
        x=np.zeros(shape=(1,len(self.f2v)),dtype=np.float32)
        self.process_instance(c_toks,t,t_toks,line,l_toks,lid,tscore,0,x)
        return self.prob(x)
    def process_instance(self,c_toks={"dummy"},t="dummy",t_toks={"dummy"},line="dummy",l_toks={"dummy"},lid=0,tscore=0,obsnum=0,array=np.zeros(shape=(1,1)),dtype=np.float32):
        features=line_features(c_toks,t,t_toks,line,l_toks,lid,tscore)
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
                c_toks=set(word_tokenize(claim.lower()))
                for yn in selected[cid]:
                    [title,lid,line,tscore]=selected[cid][yn]
                    t_toks=normalize_title(title)
                    t=" ".join(t_toks)
                    t_toks=set(t_toks)
                    l_toks=set(word_tokenize(line.lower()))
                    self.process_instance(c_toks,t,t_toks,line,l_toks,lid,tscore,obsnum,X)
                    y[obsnum]=float(yn)
                    obsnum+=1
        assert obsnum==obs
        return X,y


def select_lines(docs,t2jnum,train):
    selected=dict()
    rlines=load_doc_lines(docs,t2jnum)
    samp_size=20000
    tots={"SUPPORTS": 0, "REFUTES": 0}
    sofar={"SUPPORTS": 0, "REFUTES": 0}
    examples=Counter()
    for example in train:
        cid=example["id"]
        claim=example["claim"]
        l=example["label"]
        if l=='NOT ENOUGH INFO':
            continue
        all_evidence=[e for eset in example["evidence"] for e in eset]
        evset=set()
        for ev in all_evidence:
            evid  =ev[2]
            if evid != None:
                evset.add(evid)
        flag=False
        for doc,score in docs[cid]:
            if doc in evset:
                flag=True
        if flag:
            tots[l]+=1
            examples[l]+=1
    for l,c in examples.most_common():
        print(l,c)
    for example in train:
        cid=example["id"]
        claim=example["claim"]
        l=example["label"]
        if l=='NOT ENOUGH INFO':
            continue
        all_evidence=[e for eset in example["evidence"] for e in eset]
        lines=dict()
        for ev in all_evidence:
            evid  =ev[2]
            evline=ev[3]
            if evid != None:
                if evid not in lines:
                    lines[evid]=set()
                lines[evid].add(evline)
        flag=False
        for doc,score in docs[cid]:
            if doc in lines:
                flag=True
        if flag:
            prob=(samp_size-sofar[l])/(tots[l])
            if random()<prob:
                ylines=list()
                nlines=list()
                for title,score in docs[cid]:
                    for l_id in rlines[title]:
                        l_txt=rlines[title][l_id]
                        if title in lines and l_id in lines[title]:
                            ylines.append([title,l_id,l_txt,score])
                        elif l_txt != "":
                            nlines.append([title,l_id,l_txt,score])
                selected[cid]=dict()
                for yn, ls in [(1,ylines),(0,nlines)]:
                    shuffle(ls)
                    selected[cid][yn]=ls[0]
                sofar[l]+=1 
            tots[l]-=1
    with open("data/line_ir_lines","w") as w:
        for cid in selected:
            for yn in selected[cid]:
                [t,i,l,s]=selected[cid][yn]
                w.write(str(cid)+"\t"+str(yn)+"\t"+t+"\t"+str(i)+"\t"+str(l)+"\t"+str(s)+"\n")
    for l in sofar:
        print(l,sofar[l])
    return selected

def load_selected(fname="data/line_ir_lines"):
    selected=dict()
    with open(fname) as f:
        for line in tqdm(f):
            fields=line.rstrip("\n").split("\t")
            cid=int(fields[0])
            yn=int(fields[1])
            t=fields[2]
            i=int(fields[3])
            l=fields[4]
            s=float(fields[5])
            if cid not in selected:
                selected[cid]=dict()
            selected[cid][yn]=[t,i,l,s]
    return selected
                    
            
        



if __name__ == "__main__":
    parser = argparse.ArgumentParser("perform ir for sentences")
    parser.add_argument("--best", type=int, default=5, help="how many setences to retrieve")
    args = parser.parse_args()
    print(args)

    train, dev = load_paper_dataset()
    # train, dev = load_split_trainset(9999)
    with open("data/edocs.bin","rb") as rb:
        edocs=pickle.load(rb)
    with open("data/doc_ir_model.bin","rb") as rb:
        dmodel=pickle.load(rb)
    t2jnum=titles_to_jsonl_num()
    try:
        with open("data/line_ir_model.bin","rb") as rb:
            model=pickle.load(rb)
    except:
        try:
            selected=load_selected() 
        except:
            docs=doc_ir(train,edocs,model=dmodel)
            selected=select_lines(docs,t2jnum,train)
        model=line_ir_model()
        X,y=model.process_train(selected,train)
        model.fit(X,y)
        with open("data/line_ir_model.bin","wb") as wb:
            pickle.dump(model,wb)
    docs=doc_ir(dev,edocs,model=dmodel)
    lines=load_doc_lines(docs,t2jnum)
    evidence=line_ir(dev,docs,lines,best=args.best,model=model)
    line_hits(dev,evidence)
    
