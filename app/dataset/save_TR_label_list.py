import argparse
from tqdm import tqdm
from utils import DataUtils,GraphGenerator,GraphUtils,GRAPH_TYPE

"""
TR_label_list 저장 dir 위치: github_repo/data/TR-NE/TR_label/~
graph_list 저장 dir 계층 구조:
purpose
    graph_type
        TR_label_list

Batch size:
N20 -> B10
N50 -> B20
N100 -> B30
N500 -> B100
N1000 -> B200
"""
def main(**kwargs):
    ### unpack kwargs
    purpose=kwargs["purpose"]
    n_node=kwargs["n_node"]
    batch_size=kwargs["batch_size"]

    ### compute and save TR_label_list
    for graph_type in tqdm(GRAPH_TYPE,desc=f"Compute TR_label_list..."):
        graph_list=DataUtils.load_graph_list(purpose=purpose,graph_type=graph_type,n_node=n_node)
        TR_label_list=[]
        for graph in tqdm(graph_list,desc=f"Compute {graph_type} TR_label_list..."):
            graph_df=GraphUtils.convert_nx_graph_to_df(graph=graph)
            TR_label=GraphUtils.get_TR_label_using_Time_Centric(graph_df=graph_df,batch_size=batch_size)
            TR_label_list.append(TR_label)
        DataUtils.save_TR_label_list_to_pt(
            TR_label_list=TR_label_list,
            purpose=purpose,
            graph_type=graph_type,
            n_node=n_node,
            batch_size=batch_size
        )

if __name__=="__main__":
    """
    Execute app
    """
    parser=argparse.ArgumentParser()
    parser.add_argument("--purpose",type=str,choices=["train","val","test"],default="train")
    parser.add_argument("--n_node",type=int,choices=[20,50,100,500,1000],default=20)
    parser.add_argument("--batch_size",type=int,choices=[10,20,30,100,200],default=10)
    args=parser.parse_args()
    app_config={
        "purpose":args.purpose,
        "n_node":args.n_node,
        "batch_size":args.batch_size
    }
    main(**app_config)