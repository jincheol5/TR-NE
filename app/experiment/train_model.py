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
    use_mlflow=kwargs["use_mlflow"]
    experiment_name=kwargs["experiment_name"]
    save_model=kwargs["save_model"]

    ### set seed
    TrainUtils.set_seed(seed=seed)

    """
    1. Data
    """
    train_graph_list=[]
    train_loader_list=[]
    train_TR_label_list=[]

    val_graph_list=[]
    val_loader_list=[]
    val_TR_label_list=[]
    val_sources=[]
    for graph_type in GRAPH_TYPE:
        ### train data
        # get graph_list
        graph_list=DataUtils.load_graph_list(
            purpose="train",
            graph_type=graph_type,
            n_node=20
        )
        train_graph_list+=graph_list

        # get graph_loader
        graph_loader_list=[]
        for graph in graph_list:
            graph_df=GraphUtils.convert_nx_graph_to_df(graph=graph)
            graph_dataset=TemporalGraphDataset(graph_df=graph_df)
            graph_loader=DataLoader(dataset=graph_dataset,batch_size=10,shuffle=False)
            graph_loader_list.append(graph_loader)
        train_loader_list+=graph_loader_list

        # get TR_label_list
        TR_label_list=DataUtils.load_TR_label_list(
            purpose="train",
            graph_type=graph_type,
            n_node=20,
            batch_size=10
        )
        train_TR_label_list+=TR_label_list

        ### validate data
        # get graph_list
        graph_list=DataUtils.load_graph_list(
            purpose="val",
            graph_type=graph_type,
            n_node=20
        )
        val_graph_list+=graph_list

        # get graph_loader
        graph_loader_list=[]
        for graph in graph_list:
            graph_df=GraphUtils.convert_nx_graph_to_df(graph=graph)
            graph_dataset=TemporalGraphDataset(graph_df=graph_df)
            graph_loader=DataLoader(dataset=graph_dataset,batch_size=10,shuffle=False)
            graph_loader_list.append(graph_loader)
        val_loader_list+=graph_loader_list

        # get TR_label_list
        TR_label_list=DataUtils.load_TR_label_list(
            purpose="val",
            graph_type=graph_type,
            n_node=20,
            batch_size=10
        )
        val_TR_label_list+=TR_label_list

        # get sources
        each_val_sources=DataUtils.load_sources(
            purpose="val",
            graph_type=graph_type,
            n_node=20
        )
        val_sources+=each_val_sources

    """
    2. Model
    """
    ### set model
    node_dim=172
    edge_dim=172
    common_model_config={
        "model_name":model_name,
        "node_dim":node_dim,
        "edge_dim":edge_dim,
        **COMMON_MODEL_CONFIG,
    }
    specific_model_config=SPECIFIC_MODEL_CONFIG[model_name]
    model_config=common_model_config|specific_model_config

    # 모델 생성자에 필요한 파라미터만 추출
    init_param_config={
        key:model_config[key]
        for key in MODEL_PARAM[model_name]
    }
    model=MODEL_REGISTRY[model_name](**init_param_config)

    """
    3. Train
    """
    train_config={
        **TRAIN_CONFIG,
        "seed":seed,
        "lr":lr,
        "n_node":20,
        "batch_size":10
    }
    model_trainer=TRAINER_REGISTRY[model_name]

    if use_mlflow:
        ### set mlflow 
        MLFlowUtils.setup(experiment_name=experiment_name)
        run_name=f"{model_name}_S{seed}_LR{lr}"
        with mlflow.start_run(run_name=run_name) as run:
            # log hyperparameters
            mlflow.log_params(model_config|train_config)

            ### train model
            train_config["use_mlflow"]=use_mlflow
            model=model_trainer.train(
                model=model,
                train_graph_list=train_graph_list,
                train_loader_list=train_loader_list,
                train_TR_label_list=train_TR_label_list,
                val_graph_list=val_graph_list,
                val_loader_list=val_loader_list,
                val_TR_label_list=val_TR_label_list,
                val_sources=val_sources,
                **train_config,
            )

            ### save model
            if save_model:
                MLFlowUtils.save_model(model=model,artifact_name=model_name)

            # mlflow run_id
            run_id=run.info.run_id
        print(f"Training completed: {run_name}")
        print(f"Run ID: {run_id}")
    else:
        train_config["use_mlflow"]=use_mlflow
        ### train model
        model=model_trainer.train(
            model=model,
            train_graph_list=train_graph_list,
            train_loader_list=train_loader_list,
            train_TR_label_list=train_TR_label_list,
            val_graph_list=val_graph_list,
            val_loader_list=val_loader_list,
            val_TR_label_list=val_TR_label_list,
            val_sources=val_sources,
            **train_config
        )
        print(f"Training completed!")

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
    parser.add_argument("--use_mlflow",type=int,default=0)
    parser.add_argument("--experiment_name",type=str,default=f"TR-NE")
    parser.add_argument("--save_model",type=int,default=0)
    args=parser.parse_args()
    main(**vars(args))