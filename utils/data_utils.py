import os
import pickle
import networkx as nx
import torch
from typing import Literal

BASE_PATH=os.path.join("..","data")
os.makedirs(os.path.dirname(BASE_PATH),exist_ok=True)
class DataUtils:
    @staticmethod
    def save_graph_list_to_pkl(
            graph_list:list[nx.Graph],
            purpose:Literal["train","val","test"],
            graph_type:Literal[
                "ladder","grid","tree",
                "erdos","barabasi","community","caveman"
            ],
            n_node:int
        ):
        file_name=f"{purpose}_{graph_type}_{n_node}_graph_list.pkl"
        file_path=os.path.join(BASE_PATH,"TR-NE","graph_list",purpose,graph_type,file_name)
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        with open(file_path,'wb') as f:
            pickle.dump(graph_list,f)
        print(f"Complete to save {file_name}!")

    @staticmethod
    def load_graph_list(
            purpose:Literal["train","val","test"],
            graph_type:Literal[
                "ladder","grid","tree",
                "erdos","barabasi","community","caveman"
            ],
            n_node:int
        ):
        file_name=f"{purpose}_{graph_type}_{n_node}_graph_list.pkl"
        file_path=os.path.join(BASE_PATH,"TR-NE","graph_list",purpose,graph_type,file_name)
        with open(file_path,'rb') as f:
            graph_list=pickle.load(f)
        return graph_list

    @staticmethod
    def save_TR_label_list_to_pt(
            TR_label_list:list[torch.Tensor],
            purpose:Literal["train","val","test"],
            graph_type:Literal[
                "ladder","grid","tree",
                "erdos","barabasi","community","caveman"
            ],
            n_node:int,
            batch_size:int
        ):
        """
        file_name example: train_ladder_20_B200.pt
        """
        file_name=f"{purpose}_{graph_type}_{n_node}_B{batch_size}.pt"
        file_path=os.path.join(BASE_PATH,"TR-NE","TR_label_list",purpose,graph_type,file_name)
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        torch.save(TR_label_list,file_path)
        print(f"Save {file_name}!")

    @staticmethod
    def load_TR_label_list(
            purpose:Literal["train","val","test"],
            graph_type:Literal[
                "ladder","grid","tree",
                "erdos","barabasi","community","caveman"
            ],
            n_node:int,
            batch_size:int
        )->dict[str,torch.Tensor]:
        file_name=f"{purpose}_{graph_type}_{n_node}_B{batch_size}.pt"
        file_path=os.path.join(BASE_PATH,"TR-NE","TR_label_list",purpose,graph_type,file_name)
        TR_label_list=torch.load(file_path)
        return TR_label_list

    @staticmethod
    def save_sources(
            sources:list[int],
            purpose:Literal["val","test"],
            graph_type:Literal[
                "ladder","grid","tree",
                "erdos","barabasi","community","caveman"
            ],
            n_node:int
        ):
        """
        sources: [[graph_1_sources],[graph_2_sources],...]
        file_name example: train_ladder_20_B200_sources.pt
        """
        file_name=f"{purpose}_{graph_type}_{n_node}_sources.pkl"
        file_path=os.path.join(BASE_PATH,"TR-NE","sources",purpose,graph_type,file_name)
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        with open(file_path,'wb') as f:
            pickle.dump(sources,f)
        print(f"Complete to save {file_name}!")

    @staticmethod
    def load_sources(
            purpose:Literal["val","test"],
            graph_type:Literal[
                "ladder","grid","tree",
                "erdos","barabasi","community","caveman"
            ],
            n_node:int
        )->dict[str,torch.Tensor]:
        file_name=f"{purpose}_{graph_type}_{n_node}_sources.pkl"
        file_path=os.path.join(BASE_PATH,"TR-NE","sources",purpose,graph_type,file_name)
        with open(file_path,'rb') as f:
            sources=pickle.load(f)
        return sources