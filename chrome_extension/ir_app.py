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

app = Flask(__name__)

fc_endpoint = 'http://0.0.0.0:4000/api/factcheck'
def fact_check(data):
    r = requests.post(fc_endpoint, data=json.dumps(data),headers={'content-type': 'application/json'})
    returned_json = json.loads(r.text)
    return returned_json


with open("data/edocs.bin","rb") as rb:
    edocs=pickle.load(rb)
with open("data/doc_ir_model.bin","rb") as rb:
    dmodel=pickle.load(rb)
t2jnum=titles_to_jsonl_num()
with open("data/line_ir_model.bin","rb") as rb:
    lmodel=pickle.load(rb)


@app.route('/api/evidence', methods=['GET'])
def get_evidence(): 
    claim=request.args["claim"]
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
    results=fact_check({"claim":claim, "evidences":evs, "global_fc": "SUPPORTS"})
    return jsonify(results)
        
        
        


if __name__ == "__main__":
    context=("certs/ssl.cert","certs/ssl.key")
    app.run(host='0.0.0.0', port=5000, ssl_context=context)
