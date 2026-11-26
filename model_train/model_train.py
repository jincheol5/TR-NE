import random
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import mlflow
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import TGAT,GraphMixer
from graph import GNN_Graph
from utils import GraphUtils,TrainUtils,EarlyStopper,MetricUtils,MLFlowUtils

class Model_Trainer:
    @staticmethod
    def train(
            model:TGAT|GraphMixer,
            train_graph_list:list[nx.Graph],
            train_loader_list:list[DataLoader],
            train_TR_label_list:list[torch.Tensor],
            val_graph_list:list[nx.Graph],
            val_loader_list:list[DataLoader],
            val_TR_label_list:list[torch.Tensor],
            val_sources:list[list[int]],
            **kwargs
        ):
        """
        Set GPU, Optimizer
        """
        if torch.cuda.is_available():
            device=torch.device("cuda")
        elif torch.backends.mps.is_available():
            device=torch.device("mps")
        else:
            device=torch.device("cpu")
        model=model.to(device)

        if kwargs["optimizer"]=="adam":
            optimizer=torch.optim.Adam(
                model.parameters(),
                lr=kwargs["lr"]
            )
        else:
            optimizer=torch.optim.SGD(
                model.parameters(),
                lr=kwargs["lr"]
            )

        """
        Set Early Stopper
        """
        if kwargs["early_stop"]:
            early_stop=EarlyStopper(patience=kwargs["patience"])

        """
        Train model
        """
        n_node=kwargs["n_node"]
        for epoch in tqdm(range(kwargs["epoch"]),desc=f"Model Training..."):
            for train_graph,train_loader,train_TR_label in tqdm(
                    zip(train_graph_list,train_loader_list,train_TR_label_list),
                    total=len(train_graph_list),
                    desc=f"Epoch {epoch+1} Training..."
                ):
                ### set model.graph
                graph_df=GraphUtils.convert_nx_graph_to_df(graph=train_graph)
                model.set_graph(graph=GNN_Graph(graph_df=graph_df))
                
                ### set source info
                source=random.randint(1,n_node) # random source node
                model.graph.set_source_info(source=source)
                model.graph.to_device(device=device)

                ### get label_loader
                label_loader=TrainUtils.get_label_loader(
                    source=source,
                    graph_loader=train_loader,
                    TR_label=train_TR_label
                )

                model.train()
                for batch_event,batch_label in zip(train_loader,label_loader):
                    ### update in_adj of batch_eventstream
                    event_src,event_dst,event_t,event_edge=batch_event
                    event_src=event_src.to(device)
                    event_dst=event_dst.to(device)
                    event_t=event_t.to(device)
                    event_edge=event_edge.to(device)
                    model.graph.update_in_adj(
                        src=event_src,
                        dst=event_dst,
                        event_t=event_t,
                        edge=event_edge
                    )

                    ### query_t
                    query_t=event_t.max().expand(event_dst.size(0))  # [B]
                    query_t=query_t.to(device)

                    ### TR Label
                    label=batch_label.to(device)

                    ### Forward
                    pred_logit=model(
                        dst=event_dst,
                        event_t=query_t
                    ) # [B,]

                    ### Loss
                    label_weight=TrainUtils.get_label_weight(label=label)
                    criterion=nn.BCEWithLogitsLoss(weight=label_weight)
                    loss=criterion(pred_logit,label)

                    ### Backward
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            """
            Validate Model
            """
            val_result=Model_Trainer.validate(
                model=model,
                val_graph_list=val_graph_list,
                val_loader_list=val_loader_list,
                val_TR_label_list=val_TR_label_list,
                val_sources=val_sources,
                **kwargs
            )
            print(f"{epoch+1} epoch  Validate ACC: {val_result['acc']}")
            print(f"{epoch+1} epoch  Validate AP: {val_result['ap']}")
            print(f"{epoch+1} epoch  Validate Positive Recall: {val_result['pos_recall']}")
            print(f"{epoch+1} epoch  Validate Negative Recall: {val_result['neg_recall']}")
            print(f"{epoch+1} epoch  Validate Balanced ACC: {val_result['balanced_acc']}")

            """
            MLFlow tracking
            """
            if kwargs["use_mlflow"]:
                mlflow.log_metrics({
                    "val_loss":val_result["loss"],
                    "val_acc":val_result["acc"],
                    "val_ap":val_result["ap"],
                    "val_pos_recall":val_result["pos_recall"],
                    "val_neg_recall":val_result["neg_recall"],
                    "val_balanced_acc":val_result["balanced_acc"]
                },step=epoch+1)

            """
            Check Early Stop
            """
            print(f"{epoch+1} epoch Validate Loss: {val_result['loss']}")
            if kwargs["early_stop"]:
                pre_model=early_stop(
                    val_loss=val_result["loss"],
                    model=model
                )
                if early_stop.early_stop:
                    model=pre_model
                    print(f"Early Stop in epoch {epoch+1}")
                    break

        # early_stop이 max_epoch 내에 안 끝난 경우 최적 모델 가중치 설정
        if kwargs["early_stop"] and early_stop.best_state is not None:
            model.load_state_dict(early_stop.best_state)

        return model

    @staticmethod
    def validate(
            model:TGAT|GraphMixer,
            val_graph_list:list[nx.Graph],
            val_loader_list:list[DataLoader],
            val_TR_label_list:list[torch.Tensor],
            val_sources:list[list[int]],
            **kwargs
        ):
        """
        각 graph마다 10개의 sources 에 대해 각각 추론 후,
        각 graph의 각 source마다 성능 평가하여 집계 후 최종 metric 계산하여 반환. 
        """

        """
        Set GPU, Optimizer
        """
        if torch.cuda.is_available():
            device=torch.device("cuda")
        elif torch.backends.mps.is_available():
            device=torch.device("mps")
        else:
            device=torch.device("cpu")
        model=model.to(device)
        model.eval()

        """
        Validate model
        """
        loss_list=[]
        acc_list=[]
        ap_list=[]
        pos_recall_list=[]
        neg_recall_list=[]
        balanced_acc_list=[]
        with torch.no_grad():
            for val_graph,val_loader,val_TR_label,sources in tqdm(
                    zip(val_graph_list,val_loader_list,val_TR_label_list,val_sources),
                    total=len(val_graph_list),
                    desc=f"Validate..."
                ):
                graph_loss_list=[]
                graph_acc_list=[]
                graph_ap_list=[]
                for source in sources:
                    ### set model.graph
                    graph_df=GraphUtils.convert_nx_graph_to_df(graph=val_graph)
                    model.set_graph(graph=GNN_Graph(graph_df=graph_df))

                    ### set source info
                    model.graph.set_source_info(source=source)
                    model.graph.to_device(device=device)
                    
                    ### get label_loader
                    label_loader=TrainUtils.get_label_loader(
                        source=source,
                        graph_loader=val_loader,
                        TR_label=val_TR_label
                    )

                    src_loss_list=[]
                    src_pred_logit_list=[]
                    src_label_list=[]
                    for batch_event,batch_label in zip(val_loader,label_loader):
                        ### update in_adj of batch_eventstream
                        event_src,event_dst,event_t,event_edge=batch_event
                        event_src=event_src.to(device)
                        event_dst=event_dst.to(device)
                        event_t=event_t.to(device)
                        event_edge=event_edge.to(device)
                        model.graph.update_in_adj(
                            src=event_src,
                            dst=event_dst,
                            event_t=event_t,
                            edge=event_edge
                        )

                        ### query_t
                        query_t=event_t.max().expand(event_dst.size(0))  # [B]
                        query_t=query_t.to(device)

                        ### TR Label
                        label=batch_label.to(device)

                        ### Forward
                        pred_logit=model(
                            dst=event_dst,
                            event_t=query_t
                        ) # [B,]

                        ### Loss
                        criterion=nn.BCEWithLogitsLoss()
                        batch_loss=criterion(pred_logit,label)
                        src_loss_list.append(batch_loss)

                        # Collect all samples on CPU for dataset-level metrics.
                        src_pred_logit_list.append(pred_logit.detach().cpu())
                        src_label_list.append(label.detach().cpu())

                    # each graph metric
                    src_pred_logit=torch.cat(src_pred_logit_list)
                    src_label=torch.cat(src_label_list)
                    graph_loss_list.append(torch.stack(src_loss_list).mean().item())
                    graph_acc_list.append(MetricUtils.compute_ACC(pred_logit=src_pred_logit,label=src_label))
                    graph_ap_list.append(MetricUtils.compute_AP(pred_logit=src_pred_logit,label=src_label))
                    pos_recall_list.append(
                        MetricUtils.compute_pos_recall(pred_logit=src_pred_logit,label=src_label)
                    )
                    neg_recall_list.append(
                        MetricUtils.compute_neg_recall(pred_logit=src_pred_logit,label=src_label)
                    )
                    balanced_acc_list.append(
                        MetricUtils.compute_balanced_ACC(pred_logit=src_pred_logit,label=src_label)
                    )

                # total val metric
                loss_list+=graph_loss_list
                acc_list+=graph_acc_list
                ap_list+=graph_ap_list

        return {
            "loss":float(np.mean(loss_list)),
            "acc":float(np.mean(acc_list)),
            "ap":float(np.mean(ap_list)),
            "pos_recall":float(np.mean(pos_recall_list)),
            "neg_recall":float(np.mean(neg_recall_list)),
            "balanced_acc":float(np.mean(balanced_acc_list)),
        }

    @staticmethod
    def test(
            model:TGAT|GraphMixer,
            test_graph_list:list[nx.Graph],
            test_loader_list:list[DataLoader],
            test_TR_label_list:list[torch.Tensor],
            test_sources:list[list[int]],
            **kwargs
        ):
        """
        각 graph마다 10개의 sources 에 대해 각각 추론 후,
        각 graph의 각 source마다 성능 평가하여 집계 후 최종 metric 계산하여 반환. 
        """

        """
        Set GPU, Optimizer
        """
        if torch.cuda.is_available():
            device=torch.device("cuda")
        elif torch.backends.mps.is_available():
            device=torch.device("mps")
        else:
            device=torch.device("cpu")
        model=model.to(device)
        model.eval()

        """
        Test model
        """
        acc_list=[]
        ap_list=[]
        pos_recall_list=[]
        neg_recall_list=[]
        balanced_acc_list=[]
        with torch.no_grad():
            for test_graph,test_loader,test_TR_label,sources in tqdm(
                    zip(test_graph_list,test_loader_list,test_TR_label_list,test_sources),
                    total=len(test_graph_list),
                    desc=f"Test..."
                ):
                graph_acc_list=[]
                graph_ap_list=[]
                for source in sources:
                    ### set model.graph
                    graph_df=GraphUtils.convert_nx_graph_to_df(graph=test_graph)
                    model.set_graph(graph=GNN_Graph(graph_df=graph_df))

                    ### set source info
                    model.graph.set_source_info(source=source)
                    model.graph.to_device(device=device)
                    
                    ### get label_loader
                    label_loader=TrainUtils.get_label_loader(
                        source=source,
                        graph_loader=test_loader,
                        TR_label=test_TR_label
                    )

                    src_pred_logit_list=[]
                    src_label_list=[]
                    for batch_event,batch_label in zip(test_loader,label_loader):
                        ### update in_adj of batch_eventstream
                        event_src,event_dst,event_t,event_edge=batch_event
                        event_src=event_src.to(device)
                        event_dst=event_dst.to(device)
                        event_t=event_t.to(device)
                        event_edge=event_edge.to(device)
                        model.graph.update_in_adj(
                            src=event_src,
                            dst=event_dst,
                            event_t=event_t,
                            edge=event_edge
                        )

                        ### query_t
                        query_t=event_t.max().expand(event_dst.size(0))  # [B]
                        query_t=query_t.to(device)

                        ### TR Label
                        label=batch_label.to(device)

                        ### Forward
                        pred_logit=model(
                            dst=event_dst,
                            event_t=query_t
                        ) # [B,]

                        # Collect all samples on CPU for dataset-level metrics.
                        src_pred_logit_list.append(pred_logit.detach().cpu())
                        src_label_list.append(label.detach().cpu())

                    # each graph metric
                    src_pred_logit=torch.cat(src_pred_logit_list)
                    src_label=torch.cat(src_label_list)
                    graph_acc_list.append(MetricUtils.compute_ACC(pred_logit=src_pred_logit,label=src_label))
                    graph_ap_list.append(MetricUtils.compute_AP(pred_logit=src_pred_logit,label=src_label))
                    pos_recall_list.append(
                        MetricUtils.compute_pos_recall(pred_logit=src_pred_logit,label=src_label)
                    )
                    neg_recall_list.append(
                        MetricUtils.compute_neg_recall(pred_logit=src_pred_logit,label=src_label)
                    )
                    balanced_acc_list.append(
                        MetricUtils.compute_balanced_ACC(pred_logit=src_pred_logit,label=src_label)
                    )

                # total val metric
                acc_list+=graph_acc_list
                ap_list+=graph_ap_list

        return {
            "acc":float(np.mean(acc_list)),
            "ap":float(np.mean(ap_list)),
            "pos_recall":float(np.mean(pos_recall_list)),
            "neg_recall":float(np.mean(neg_recall_list)),
            "balanced_acc":float(np.mean(balanced_acc_list)),
        }
