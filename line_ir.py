from doc_ir import doc_ir, title_edict
from util import normalize_title, load_stoplist
from fever_io import load_doc_lines, titles_to_jsonl_num, load_split_trainset
from collections import Counter
from nltk import word_tokenize, sent_tokenize
import pickle
from tqdm import tqdm

stop=load_stoplist()

def div(x,y):
    if y==0:
        return 1.0
    else:
        return x/y
#def div(x,y):
#    return x

def line_features(c_toks=set(), title="", t_toks=set(), line="", l_toks=set(), lid=0, score=0):
    features=dict()
    features["lenl"]=len(l_toks)
    features["tinl"]=(title in line)
    features["lid"]=lid
    features["lid0"]=(lid==0)
    features["score"]=score
    cns_toks=c_toks-stop
    cnt_toks=c_toks - t_toks
    cntns_toks=cns_toks - t_toks    
    lnt_toks=l_toks - t_toks
    lns_toks=l_toks - stop
    lntns_toks=lns_toks - t_toks
    cl_toks=c_toks & l_toks
    clnt_toks=cnt_toks & lnt_toks
    clns_toks=cns_toks & lns_toks
    clntns_toks=cntns_toks & lntns_toks
    features["pc"]=div(len(cl_toks),len(c_toks))
    features["pl"]=div(len(cl_toks),len(l_toks))
    features["pcns"]=div(len(clns_toks),len(cns_toks))
    features["plns"]=div(len(clns_toks),len(lns_toks))
    features["pcnt"]=div(len(clnt_toks),len(cnt_toks))
    features["plnt"]=div(len(clnt_toks),len(lnt_toks))
    features["pcntns"]=div(len(clntns_toks),len(cntns_toks))
    features["plntns"]=div(len(clntns_toks),len(lntns_toks))
    return features

def score_line(features=dict()):
    vlist={"lenl":0.032, "tinl":-0.597, "lid":-0.054, "lid0":1.826, "pc":-3.620, "pl":3.774, "pcns":3.145, "plns":-6.423, "pcnt":4.195, "pcntns":2.795, "plntns":5.133}
    score=0
    for v in vlist:
        score=score+features[v]*vlist[v]
    return score


def best_lines(claim="",tscores=list(),lines=dict(),best=5,model=None):
    lscores=list()
    c_toks=set(word_tokenize(claim.lower()))
    for title,tscore in tscores:
        t_toks=normalize_title(title)
        t=" ".join(t_toks)
        t_toks=set(t_toks)
        for lid in lines[title]:
            line=lines[title][lid]
            l_toks=set(word_tokenize(line.lower()))
            if len(l_toks) > 0:
                if model==None:
                    lscores.append((title,lid,score_line(line_features(c_toks,t,t_toks,line,l_toks,lid,tscore))))
                else:
                    lscores.append((title,lid,model.score_instance(c_toks,t,t_toks,line,l_toks,lid,tscore)))
    lscores=sorted(lscores,key=lambda x:-1*x[2])[:best]
    return lscores


def line_hits(data=list(),evidence=dict()):
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
        lines=dict()
        for ev in all_evidence:
            evid  =ev[2]
            evline=ev[3]
            if evid != None:
                if evid not in lines:
                    lines[evid]=set()
                lines[evid].add(evline)
        e2s=dict()
        evsets=dict()
        sid=0
        for s in example["evidence"]:
            evsets[sid]=set()
            for e in s:
                evsets[sid].add((e[2],e[3]))
                if (e[2],e[3]) not in e2s:
                    e2s[(e[2],e[3])]=set()
                e2s[(e[2],e[3])].add(sid)
            sid=sid+1
        for i,(d,l,s) in enumerate(evidence[cid]):
            hits[i]=hits[i]+1*(d in lines and l in lines[d])
            returned[i]=returned[i]+1
            flag=0
            if (d,l) in e2s:
                for sid in e2s[(d,l)]:
                    s=evsets[sid]
                    if (d,l) in s:
                        if len(s)==1:
                            flag=1
                        s.remove((d,l))
            full[i]+=flag
            if flag==1:
                break
    print()
    denom=returned[0]
    for i in range(0,len(hits)):
        print(i,hits[i],returned[i],full[i]/denom)
        full[i+1]+=full[i]




def line_ir(data=list(),docs=dict(),lines=dict(),best=5,model=None):
    """
    Returns a dictionary of n best lines for each claim.
    """
    evidence=dict()
    for example in tqdm(data):
        cid=example["id"]
        evidence[cid]=list()
        tscores=docs[cid]
        claim=example["claim"]
        evidence[cid]=best_lines(claim,tscores,lines,best,model)
    return evidence



if __name__ == "__main__":
    t2jnum=titles_to_jsonl_num()
    try:
        with open("data/edocs.bin","rb") as rb:
            edocs=pickle.load(rb)
    except:
        edocs=title_edict(t2jnum)
        with open("data/edocs.bin","wb") as wb:
            pickle.dump(edocs,wb)
    train, dev = load_split_trainset(9999)
    docs=doc_ir(dev,edocs)
    print(len(docs))
    lines=load_doc_lines(docs,t2jnum)
    print(len(lines))
    evidence=line_ir(dev,docs,lines)
    line_hits(dev,evidence)
    docs=doc_ir(train,edocs)
    print(len(docs))
    lines=load_doc_lines(docs,t2jnum)
    print(len(lines))
    evidence=line_ir(train,docs,lines)
    line_hits(train,evidence)
