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