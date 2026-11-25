import argparse
import mlflow
from torch.utils.data import DataLoader
from model import *
from model_train import *
from utils import *
from .config_info import *

def main(**kwargs):
    """
    0. Init Setting
    """
    ### unpack kwargs
    model_name=kwargs["model_name"]
    seed=kwargs["seed"]
    lr=kwargs["lr"]
    n_node=kwargs["n_node"]
    batch_size=kwargs["batch_size"]
    experiment_name=kwargs["experiment_name"]
    save_metric=kwargs["save_metric"]

    ### set seed
    TrainUtils.set_seed(seed=seed)

    """
    1. Data
    """
    test_graph_list=[]
    test_loader_list=[]
    test_TR_label_list=[]
    test_sources=[]
    for graph_type in GRAPH_TYPE:
        ### train data
        # get graph_list
        graph_list=DataUtils.load_graph_list(
            purpose="test",
            graph_type=graph_type,
            n_node=n_node
        )
        test_graph_list+=graph_list

        # get graph_loader
        graph_loader_list=[]
        for graph in graph_list:
            graph_df=GraphUtils.convert_nx_graph_to_df(graph=graph)
            graph_dataset=TemporalGraphDataset(graph_df=graph_df)
            graph_loader=DataLoader(dataset=graph_dataset,batch_size=batch_size,shuffle=False)
            graph_loader_list.append(graph_loader)
        test_loader_list+=graph_loader_list

        # get TR_label_list
        TR_label_list=DataUtils.load_TR_label_list(
            purpose="test",
            graph_type=graph_type,
            n_node=n_node,
            batch_size=batch_size
        )
        test_TR_label_list+=TR_label_list

        # get sources
        each_test_sources=DataUtils.load_sources(
            purpose="test",
            graph_type=graph_type,
            n_node=n_node
        )
        test_sources+=each_test_sources

    """
    2. Model
    """
    ### load model from mlflow
    # 모델 전체를 저장해두었기 때문에 set_graph 단계 필요 없음.
    MLFlowUtils.setup(experiment_name=experiment_name)
    run_ids=MLFlowUtils.search_run_ids(
        experiment_name=experiment_name,
        model_name=model_name,
        seed=seed,
        lr=lr
    )
    run_id=run_ids[0] # 가장 최근 run_id
    model=MLFlowUtils.load_model(run_id=run_id,artifact_name=model_name)

    """
    3. Test Model
    """
    model_trainer=TRAINER_REGISTRY[model_name]
    ### test model
    test_result=model_trainer.test(
        model=model,
        test_graph_list=test_graph_list,
        test_loader_list=test_loader_list,
        test_TR_label_list=test_TR_label_list,
        test_sources=test_sources
    )
    print(f"Evaluation Result:")
    print(f"model: {model_name}")
    print(f"n_node: {n_node}, batch_size: {batch_size}")
    print(f"hypher parameter: seed={seed}, lr={lr}")
    print(f"ACC: {test_result['acc']}")
    print(f"AP: {test_result['ap']}")
    print(f"Positive Recall: {test_result['pos_recall']}")
    print(f"Negative Recall: {test_result['neg_recall']}")
    print(f"Balanced ACC: {test_result['balanced_acc']}")

    if save_metric:
        with mlflow.start_run(run_id=run_id) as run:
            test_metrics={
                "test_acc":test_result["acc"],
                "test_ap":test_result["ap"],
                "test_pos_recall":test_result["pos_recall"],
                "test_neg_recall":test_result["neg_recall"],
                "test_balanced_acc":test_result["balanced_acc"]
            }
            mlflow.log_metrics(test_metrics)

if __name__=="__main__":
    """
    Execute app
    """
    parser=argparse.ArgumentParser()
    parser.add_argument("--model_name",
        type=str,
        choices=list(MODEL_REGISTRY),
        default=f"TGAT"
    )
    parser.add_argument("--seed",type=int,choices=[1,2,3],default=1)
    parser.add_argument("--lr",type=float,choices=[0.0001,0.0005],default=0.0001)
    parser.add_argument("--n_node",type=int,choices=[20,50,100,500,1000],default=20)
    parser.add_argument("--batch_size",type=int,choices=[10,20,30,100,200],default=10)
    parser.add_argument("--experiment_name",type=str,default=f"TR-NE")
    parser.add_argument("--save_metric",type=int,default=0)
    args=parser.parse_args()
    main(**vars(args))