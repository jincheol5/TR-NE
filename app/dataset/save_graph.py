import argparse
from utils import DataUtils,GraphGenerator

"""
graph_list 저장 dir 위치: github_repo/data/TR-NE/graph/~
graph_list 저장 dir 계층 구조:
purpose
    graph_type
        graph_list
"""
def main(**kwargs):
    ### unpack kwargs
    n_node=kwargs["n_node"]
    n_graph=kwargs["n_graph"]
    purpose=kwargs["purpose"]

    ### save graph_list
    graph_list_dict=GraphGenerator.generate_7_type_graphs(n_node=n_node,n_graph=n_graph)
    for graph_type,graph_list in graph_list_dict.items():
        DataUtils.save_graph_list_to_pkl(
            graph_list=graph_list,
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
    parser.add_argument("--n_graph",type=int,choices=[5,100],default=5)
    parser.add_argument("--purpose",type=str,choices=["train","val","test"],default="CollegeMsg")
    args=parser.parse_args()
    app_config={
        "n_node":args.n_node,
        "n_graph":args.n_graph,
        "purpose":args.purpose
    }
    main(**app_config)