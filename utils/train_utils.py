import os
import random
import pandas as pd
import numpy as np
import torch
from typing import Literal
from torch.utils.data import Dataset,DataLoader

class TemporalGraphDataset(Dataset):
    def __init__(self,graph_df:pd.DataFrame):
        self.src=torch.tensor(graph_df["u"].values,dtype=torch.long)
        self.dst=torch.tensor(graph_df["i"].values,dtype=torch.long)
        self.t=torch.tensor(graph_df["t"].values,dtype=torch.int64)
        self.edge=torch.tensor(graph_df["idx"].values,dtype=torch.long)

    def __len__(self):
        return len(self.src)

    def __getitem__(self,idx):
        return self.src[idx],self.dst[idx],self.t[idx],self.edge[idx]

class EarlyStopper:
    def __init__(self,
            patience:int=1
        ):
        self.patience=patience
        self.patience_count=0
        self.best_loss=np.inf
        self.best_state=None
        self.early_stop=False
    def __call__(self,
            val_loss:float,
            model:torch.nn.Module
        ):
        # val_loss가 NaN, Inf이면 즉시 early stop
        if not np.isfinite(val_loss): 
            print("Loss is NaN or Inf!")
            self.early_stop=True
            if self.best_state is not None:
                model.load_state_dict(self.best_state)
            return model

        # 첫 번째 validation에서는 비교할 이전 best가 없으므로 현재 loss와 모델을 그대로 best로 저장
        if self.best_state is None: 
            self.best_loss=val_loss
            self.best_state={
                key: value.detach().clone()
                for key,value in model.state_dict().items()
            }
            return model

        # val_loss가 개선 되지 않은 경우
        if self.best_loss<=val_loss: 
            self.patience_count+=1
            if self.patience<=self.patience_count:
                self.early_stop=True
                model.load_state_dict(self.best_state)
            return model

        # val_loss가 개선 된 경우
        self.patience_count=0
        self.best_loss=val_loss
        self.best_state={
            key:value.detach().clone()
            for key,value in model.state_dict().items()
        }
        return model

class TrainUtils:
    @staticmethod
    def set_seed(seed:int):
        os.environ["PYTHONHASHSEED"]=str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.benchmark=False

    @staticmethod
    def get_label_loader(
            source:int,
            graph_loader:DataLoader,
            TR_label:torch.Tensor
        ):
        """
        Input:
            source: int
            graph_loader: DataLoader, len = batch_seq_len
            TR_label: [batch_seq_len,N+1,N+1] bool tensor
        Return:
            label_loader: list of [B,] float32 tensor (source -> dst TR_label, 1.0 or 0.0)
        """
        label_loader=[]
        for batch_event,batch_TR_label in zip(graph_loader,TR_label):
            _,dst,_,_=batch_event
            batch_dst_label=batch_TR_label[source,dst].to(torch.float32) # [B,]
            label_loader.append(batch_dst_label)
        return label_loader # list of [B,] float32 tensor

    @staticmethod
    def get_label_weight(label:torch.Tensor):
        """
        class가 모두 1 or 0이면 동일 가중치.
        """
        n_pos=(label==1).sum()
        n_neg=(label==0).sum()
        batch_size=label.numel()
        if n_pos>0 and n_neg>0:
            pos_weight=batch_size/(2*n_pos)
            neg_weight=batch_size/(2*n_neg)
            label_weight=torch.where(label==1,pos_weight,neg_weight)
        else:
            label_weight=torch.ones_like(label)
        return label_weight