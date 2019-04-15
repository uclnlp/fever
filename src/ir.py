from line_ir import line_ir
from doc_ir import best_titles
from fever_io import load_doc_lines


def get_retrieval_method(dmodel, lmodel, edocs, t2jnum):
    def get_evidence(claim):
        tscores=best_titles(claim,edocs,best=30,model=dmodel)
        docs={0:tscores}
        lines=load_doc_lines(docs,t2jnum)
        evidence=line_ir([{"claim":claim, "id":0}],docs,lines,model=lmodel,best=15)
        evs=[]
        for i, (title,lid,score) in enumerate(evidence[0]):
            print(title)
            evs.append({
                "id":i,
                "title": title,
                "linum": lid,
                "document":{"header":{"sourceItemOriginFeedName":title,
                                        "sourceItemIdAtOrigin":"https://en.wikipedia.org/wiki/"+title}},
                "element":{"text":lines[title][lid]},
                "group_fact_check": "SUPPORTS"})

        return {"claim":claim, "evidences":evs, "global_fc": "SUPPORTS"}

    return get_evidence
