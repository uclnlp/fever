import os
import json
from nltk import word_tokenize, sent_tokenize




def load_stoplist(stopfile="stoplist"):
    stop=set()
    with open(stopfile) as f:
        for line in f:
            word=line.rstrip("\n")
            stop.add(word)
    return stop





def normalize_title(title,rflag=False):
    rmndr=""
    l_txt=title.replace("_"," ").replace("-COLON-",":")
    if l_txt.find("-LRB-") > -1:
        rmndr=l_txt[l_txt.find("-LRB-"):]
        rmndr=rmndr.replace("-LRB-","(").replace("-RRB-",")")
        l_txt=l_txt[:l_txt.find("-LRB-")].rstrip(" ")
    l_txt=word_tokenize(l_txt.lower())
    if rflag:
        return l_txt, rmndr
    else:
        return l_txt



def abs_path(relative_path_to_file):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_dir, relative_path_to_file)


class edict():
    def __init__(self):
        self.d=dict()
    
    def __getitem__(self,key):
        if key[0] in self.d:
            if len(key)==1:
                return self.d[key[0]]
            else:
                return self.d[key[0]][1][key[1:]]
        else:
            return (None,None)
    
    def __setitem__(self,key,value):
        if len(key)==1:
            self.d[key[0]]=(value,self.d.get(key[0],(None,edict()))[1])
        else:
            val,sube=self.d.get(key[0],(None,edict()))
            sube[key[1:]]=value
            self.d[key[0]]=(val,sube)
    
    def __contains__(self,key):
        if len(key)==1:
            return key[0] in self.d
        else:
            return key[0] in self.d and key[1:] in self.d[key[0]][1]
    
    def __len__(self):
        return len(self.d)
        
class pdict():
    def __init__(self,ed):
        self.ed=ed
        self.pos=0
        self.d={"":(self.ed,self.pos)}
    def __getitem__(self,key):
        self.pos+=1
        newd={"":(self.ed,self.pos)}
        rlist=[]
        for prefix in self.d:
            start=self.d[prefix][1]
            if [key.lower()] in self.d[prefix][0]:
                tf,ped=self.d[prefix][0][[key.lower()]]
                newprefix=prefix+" "+key
                if len(ped)>0:
                    newd[newprefix]=(ped,start)
                if tf is not None:
                    rlist.append((tf,newprefix,start))
        self.d=newd
        return rlist
