from flask import Flask, jsonify, abort, request
from line_ir import line_ir
from doc_ir import best_titles
from doc_ir_model import doc_ir_model
from line_ir_model import line_ir_model
from util import edict, pdict, normalize_title, load_stoplist
from fever_io import load_doc_lines, titles_to_jsonl_num, load_split_trainset, load_paper_dataset
import pickle
import json
import requests

def get_retrieval_method(dmodel, lmodel, edocs, t2jnum):
    def get_evidence(data):
        claim=data["claim"]
        tscores=best_titles(claim,edocs,best=30,model=dmodel)
        docs={0:tscores}
        lines=load_doc_lines(docs,t2jnum)
        evidence=line_ir([{"claim":claim, "id":0}],docs,lines,model=lmodel,best=15)
        evs=[]
        for i, (title,lid,score) in enumerate(evidence[0]):
            print(title)
            evs.append({
                    "id":i,
                    "document":{"header":{"sourceItemOriginFeedName":title,
                                            "sourceItemIdAtOrigin":"https://en.wikipedia.org/wiki/"+title}},
                    "element":{"text":lines[title][lid]},
                    "group_fact_check": "SUPPORTS"})

        return {"claim":claim, "evidences":evs, "global_fc": "SUPPORTS"}

    return get_evidence
