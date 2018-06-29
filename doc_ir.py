from util import edict, pdict, normalize_title, load_stoplist
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import gazetteers, names
from collections import Counter
from fever_io import titles_to_jsonl_num, load_split_trainset
import pickle
from tqdm import tqdm
import numpy as np


places=set(gazetteers.words())
people=set(names.words())
stop=load_stoplist()



def title_edict(t2jnum={}):
    edocs=edict()
    for title in t2jnum:
        l_txt=normalize_title(title)
        if len(l_txt) > 0:
            if edocs[l_txt][0] is None:
                edocs[l_txt]=[]
            edocs[l_txt][0].append(title)
    return edocs

def find_titles_in_claim(claim="",edocs=edict()):
    find=pdict(edocs)
    docset={}
    ctoks=word_tokenize(claim)
    for word in ctoks:
        for dlist,phrase,start in find[word]:
            for d in dlist:
                if d not in docset:
                    docset[d]=[]
                docset[d].append((phrase,start))
    return docset

def phrase_features(phrase="",start=0,title="",claim=""):
    features=dict()
    stoks=phrase.split()
    t_toks,rmndr = normalize_title(title,rflag=True)
    features["rmndr"]=(rmndr=="")
    features["rinc"]=((rmndr!="") and (rmndr in claim))
    features["start"]=start
    features["start0"]=(start==0)
    features["lend"]=len(stoks)
    features["lend1"]=(features["lend"]==1)
    features["cap1"]=stoks[0][0].isupper()
    features["stop1"]=(stoks[0].lower() in stop)
    features["people1"]=(stoks[0] in people)
    features["places1"]=(stoks[0] in places)
    features["capany"]=False
    features["capall"]=True
    features["stopany"]=False
    features["stopall"]=True
    features["peopleany"]=False
    features["peopleall"]=True
    features["placesany"]=False
    features["placesall"]=True
    for tok in stoks:
        features["capany"]=(features["capany"] or tok[0].isupper())
        features["capall"]=(features["capall"] and tok[0].isupper())
        features["stopany"]=(features["stopany"] or tok.lower() in stop)
        features["stopall"]=(features["stopall"] and tok.lower() in stop)
        features["peopleany"]=(features["peopleany"] or tok in people)
        features["peopleall"]=(features["peopleall"] and tok in people)
        features["placesany"]=(features["placesany"] or tok in places)
        features["placesall"]=(features["placesall"] and tok in places)
    return features
    


def score_phrase(features=dict()):
    vlist={"lend":0.928, "lend1":-2.619, "cap1":0.585, "capany":0.408, "capall":0.685, "stop1":-1.029, "stopany":-1.419, "stopall":-1.061, "places1":0.305, "placesany":-0.179, "placesall":0.763, "people1":0.172, "peopleany":-0.278, "peopleall":-1.554, "start":-0.071, "start0":2.103}
    score=0
    for v in vlist:
        score=score+features[v]*vlist[v]
    return score
        

def score_title(ps_list=[],title="dummy",claim="dummy",model=None):
    maxscore=-1000000
    for phrase,start in ps_list:
        if model is None:
            score=score_phrase(phrase_features(phrase,start,title,claim))
        else:
            score=model.score_instance(phrase,start,title,claim)
        maxscore=max(maxscore,score)
    return maxscore


def best_titles(claim="",edocs=edict(),best=5,model=None):
    t2phrases=find_titles_in_claim(claim,edocs)
    tscores=list()
    for title in t2phrases:
        tscores.append((title,score_title(t2phrases[title],title,claim,model)))
    tscores=sorted(tscores,key=lambda x:-1*x[1])[:best]
    return tscores

def title_hits(data=list(),tscores=dict()):
    hits=Counter()
    returned=Counter()
    full=Counter()
    for example in data:
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
        e2s=dict()
        evsets=dict()
        sid=0
        for s in example["evidence"]:
            evsets[sid]=set()
            for e in s:
                evsets[sid].add(e[2])
                if e[2] not in e2s:
                    e2s[e[2]]=set()
                e2s[e[2]].add(sid)
            sid=sid+1
        for i,(d,s) in enumerate(tscores[cid]):
            hits[i]=hits[i]+1*(d in docs)
            returned[i]=returned[i]+1
            flag=0
            if d in e2s:
                for sid in e2s[d]:
                    s=evsets[sid]
                    if d in s:
                        if len(s)==1:
                            flag=1
                        s.remove(d)
            full[i]+=flag
            if flag==1:
                break
    print()
    denom=returned[0]
    for i in range(0,len(hits)):
        print(i,hits[i],returned[i],full[i]/denom)
        full[i+1]+=full[i]



def doc_ir(data=list(),edocs=edict(),best=5,model=None):
    """
    Returns a dictionary of n best document titles for each claim.
    """
    docs=dict()
    for example in tqdm(data):
        tscores=best_titles(example["claim"],edocs,best,model)
        docs[example["id"]]=tscores
    return docs
    


if __name__ == "__main__":
    try:
        with open("data/edocs.bin","rb") as rb:
            edocs=pickle.load(rb)
    except:
        t2jnum=titles_to_jsonl_num()
        edocs=title_edict(t2jnum)
        with open("data/edocs.bin","wb") as wb:
            pickle.dump(edocs,wb)
    train, dev = load_split_trainset(9999)
    docs=doc_ir(dev,edocs)
    title_hits(dev,docs)
    docs=doc_ir(train,edocs)
    title_hits(train,docs)
    

        
        
