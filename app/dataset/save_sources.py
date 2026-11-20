import random
import argparse
from utils import DataUtils,GRAPH_TYPE

"""
label_loader_list 저장 dir 위치: github_repo/data/TR-NE/label_loader_list/~
label_loader_list 저장 dir 계층 구조:
purpose (val, test)
    graph_type
        label_loader_list
"""
def main(**kwargs):
    ### unpack kwargs
    n_node=kwargs["n_node"]
    purpose=kwargs["purpose"]
    
    for graph_type in GRAPH_TYPE:
        # load graph_list
        graph_list=DataUtils.load_graph_list(
            purpose=purpose,
            graph_type=graph_type,
            n_node=n_node
        )

        # random sources (각 graph마다 10개 source 랜덤 샘플링)
        sources=[]
        for _ in range(len(graph_list)):
            graph_sources=random.sample(range(1,n_node+1),10)
            sources.append(graph_sources)

        # save sources
        DataUtils.save_sources(
            sources=sources,
            purpose=purpose,
            graph_type=graph_type,
            n_node=n_node
        )

if __name__=="__main__":
    """
    Execute app
    """
    parser=argparse.ArgumentParser()
    parser.add_argument("--n_node",type=int,choices=[20,50,100,500,1000],default=20)
    parser.add_argument("--purpose",type=str,choices=["val","test"],default="val")
    args=parser.parse_args()
    app_config={
        "n_node":args.n_node,
        "purpose":args.purpose
    }
    main(**app_config)