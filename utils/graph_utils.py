import random
import pandas as pd
import numpy as np
import networkx as nx
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from .train_utils import *

GRAPH_TYPE=[
    "ladder","grid","tree",
    "erdos","barabasi","community","caveman"
]

class GraphGenerator:
    @staticmethod
    def remove_self_loop(graph:nx.Graph):
        graph.remove_edges_from(nx.selfloop_edges(graph))

    @staticmethod
    def connect_isolated_nodes(graph:nx.Graph):
        """
        Connect isolated nodes to another node without changing the node set.
        """
        nodes=list(graph.nodes())
        for node in list(nx.isolates(graph)):
            if graph.degree(node)==0:
                neighbor=random.choice([other for other in nodes if other!=node])
                graph.add_edge(node,neighbor)

    @staticmethod
    def set_edge_timestamp_uniformly(graph:nx.Graph):
        """
        nx.Graph의 각 edge에 1~5개의 timestamp 랜덤하게 배정
        timestamp 범위: 0 ~ 1000 (unix timestamp day)
        timestamp 범위를 10등분 하여 각 세부 범위 시간 값들이 균등하게 배정되도록 설정
        """
        # 각 edge가 가질 timestamp 개수 결정
        edge_num_timestamps={
            edge:random.randint(1,5)
            for edge in graph.edges()
        }

        # 그래프 전체 timestamp 개수
        total_timestamps=sum(edge_num_timestamps.values())

        # 100 단위 구간
        ranges=[
            (0,100),
            (101,200),
            (201,300),
            (301,400),
            (401,500),
            (501,600),
            (601,700),
            (701,800),
            (801,900),
            (901,1000),
        ]

        # 각 구간에 들어갈 기본 timestamp 개수
        base_count=total_timestamps//len(ranges)

        # 균등 분배 후 남는 timestamp 개수
        remainder=total_timestamps%len(ranges)

        # 남는 timestamp를 받을 구간을 랜덤하게 선택
        extra_ranges=set(
            random.sample(
                range(len(ranges)),
                remainder
            )
        )

        timestamps=[]
        for i,(start,end) in enumerate(ranges):
            count=base_count

            # 나머지가 배정된 구간은 1개 추가
            if i in extra_ranges:
                count+=1

            timestamps.extend([
                random.randint(start,end)
                for _ in range(count)
            ])

        # 전체 timestamp 랜덤하게 섞기
        random.shuffle(timestamps)

        # 각 edge에 timestamp 배분
        idx=0
        for edge,num_timestamps in edge_num_timestamps.items():
            graph.edges[edge]["t"]=timestamps[
                idx:idx+num_timestamps
            ]
            idx+=num_timestamps

    @staticmethod
    def generate_7_type_graphs(
            n_node:int,
            n_graph:int
        )->dict[str,list[nx.Graph]]:
        """
        << Generate 7-type graphs >>
        1. ladder graph
        2. 2D grid graph
        3. tree graph
        4. Erdos-Renyi graph
        5. Barabasi-Albert graph
        6. 4-community graph
        7. 4-caveman graph
        """
        ladder_graph_list=[]
        grid_graph_list=[]
        tree_graph_list=[]
        erdos_renyi_graph_list=[]
        barabasi_albert_graph_list=[]
        community_graph_list=[]
        caveman_graph_list=[]

        ### generate graph
        for _ in tqdm(range(n_graph),desc=f"Generate Graph..."):
            # 1. generate ladder graph
            if n_node%2!=0:
                raise ValueError(f"ladder graph requires an even number of nodes.")
            ladder_graph=nx.ladder_graph(n_node//2)
            GraphGenerator.remove_self_loop(graph=ladder_graph)
            GraphGenerator.set_edge_timestamp_uniformly(graph=ladder_graph)
            ladder_graph_list.append(ladder_graph)

            # 2. generate 2D grid graph
            side_length=int(np.ceil(np.sqrt(n_node)))
            grid_graph=nx.grid_2d_graph(side_length,side_length)
            grid_graph=nx.convert_node_labels_to_integers(grid_graph)
            grid_graph=grid_graph.subgraph(range(n_node)).copy()
            GraphGenerator.remove_self_loop(graph=grid_graph)
            GraphGenerator.set_edge_timestamp_uniformly(graph=grid_graph)
            grid_graph_list.append(grid_graph)

            # 3. generate tree graph
            tree_graph=nx.random_labeled_tree(n_node)
            GraphGenerator.remove_self_loop(graph=tree_graph)
            GraphGenerator.set_edge_timestamp_uniformly(graph=tree_graph)
            tree_graph_list.append(tree_graph)

            # 4. generate Erdos-Renyi graph
            p=min(np.log2(n_node)/n_node,0.5)
            erdos_renyi_graph=nx.erdos_renyi_graph(n_node,p)
            GraphGenerator.remove_self_loop(graph=erdos_renyi_graph)
            GraphGenerator.connect_isolated_nodes(graph=erdos_renyi_graph)
            GraphGenerator.set_edge_timestamp_uniformly(graph=erdos_renyi_graph)
            erdos_renyi_graph_list.append(erdos_renyi_graph)

            # 5. generate Barabasi-Albert graph
            if n_node<=4:
                raise ValueError("Barabasi-Albert graph requires more than 4 number of nodes.")
            m=random.choice([4,5])
            barabasi_albert_graph=nx.barabasi_albert_graph(n_node,m)
            GraphGenerator.remove_self_loop(graph=barabasi_albert_graph)
            GraphGenerator.set_edge_timestamp_uniformly(graph=barabasi_albert_graph)
            barabasi_albert_graph_list.append(barabasi_albert_graph)

            # 6. generate 4 community graph
            if n_node<4:
                raise ValueError("4-Community graph requires at least 4 nodes.")
            community_size=n_node//4
            remaining_nodes=n_node%4
            communities=[nx.erdos_renyi_graph(community_size,0.1) for _ in range(4)]
            community_graph=nx.disjoint_union_all(communities)
            for i in range(remaining_nodes):
                community_graph.add_node(community_graph.number_of_nodes())
            nodes=list(community_graph.nodes())
            for i in range(len(nodes)):
                for j in range(i+1,len(nodes)):
                    if (i//community_size)!=(j//community_size):
                        if random.random()<0.01:
                            community_graph.add_edge(i,j)
            GraphGenerator.remove_self_loop(graph=community_graph)
            GraphGenerator.connect_isolated_nodes(graph=community_graph)
            GraphGenerator.set_edge_timestamp_uniformly(graph=community_graph)
            community_graph_list.append(community_graph)

            # 7. generate 4-caveman graph
            if n_node<4:
                raise ValueError("4-Caveman graph requires at least 4 nodes.")
            clique_size=n_node//4
            remaining_nodes=n_node%4
            caveman_graph=nx.caveman_graph(4,clique_size)
            for i in range(remaining_nodes):
                caveman_graph.add_node(caveman_graph.number_of_nodes())
            edges_to_remove=[edge for edge in caveman_graph.edges() if random.random()<0.8]
            caveman_graph.remove_edges_from(edges_to_remove)
            num_shortcuts=int(0.025*n_node)
            for _ in range(num_shortcuts):
                u,v=random.sample(list(caveman_graph.nodes()),2)
                if not caveman_graph.has_edge(u,v):
                    caveman_graph.add_edge(u,v)
            GraphGenerator.remove_self_loop(graph=caveman_graph)
            GraphGenerator.connect_isolated_nodes(graph=caveman_graph)
            GraphGenerator.set_edge_timestamp_uniformly(graph=caveman_graph)
            caveman_graph_list.append(caveman_graph)
        return {
            "ladder":ladder_graph_list,
            "grid":grid_graph_list,
            "tree":tree_graph_list,
            "erdos":erdos_renyi_graph_list,
            "barabasi":barabasi_albert_graph_list,
            "community":community_graph_list,
            "caveman":caveman_graph_list
        }

class GraphAlgorithm:
    @staticmethod
    def compute_TR_using_Time_Centric(
            eventstream:list[tuple],
            n_node:int,
            gamma_table:np.ndarray|None=None,
        )->np.ndarray:
        """
        Compute All Pairs Temporal Reachability using Time-Centric Algorithm
        - Strictly increasing temporal path
        - Reachability only 1 or 0

        Return:

        """
        INF=np.iinfo(np.int64).max # 정수 최대값

        ### Gamma Table
        if gamma_table is None:
            gamma_table=np.zeros((n_node+1,n_node+1,2),dtype=np.int64) # [N+1,N+1,2], 마지막 차원=(TR,last_t)
            gamma_table[:,:,1]=INF

            # Self Reachability
            idx=np.arange(1,n_node+1)
            gamma_table[idx,idx,0]=1
            gamma_table[idx,idx,1]=-1

        ### Compute Temporal Reachability
        for src,dst,t in eventstream:
            # 각 Source에서 src까지 도달한 시간
            src_arrival=gamma_table[1:,src,1]

            # Strictly increasing temporal path
            valid=src_arrival<t

            # dst에 더 빠르게 도달하는 경우만 갱신
            dst_arrival=gamma_table[1:,dst,1]
            update=valid&(t<dst_arrival)

            gamma_table[1:,dst,0][update]=1
            gamma_table[1:,dst,1][update]=t
        return gamma_table

class GraphUtils:
    @staticmethod
    def convert_nx_graph_to_df(
            graph:nx.Graph,
            add_bidirect:bool=False # True면 edge의 역방향도 추가
        )->pd.DataFrame:
        """
        Return:
            DataFrame columns:
                u   : source node id (1~N)
                i   : target node id (1~N)
                t   : timestamp, int, unix_timestamp (day)
                idx : 시간순 edge index (1~E)
        """
        edge_event_list=[]
        for src,dst,data in graph.edges(data=True):
            for t in data["t"]:
                edge_event_list.append({
                    "u":src+1,
                    "i":dst+1,
                    "t":t,
                })
                if add_bidirect:
                    edge_event_list.append({
                        "u":dst+1,
                        "i":src+1,
                        "t":t,
                    })

        # timestamp 오름차순 정렬
        graph_df=pd.DataFrame(edge_event_list)
        graph_df=graph_df.sort_values(
            by="t",
            ascending=True
        ).reset_index(drop=True)

        # edge index: 1~E
        graph_df["idx"]=range(1,len(graph_df)+1)
        return graph_df

    @staticmethod
    def convert_batch_to_eventstream(
            src:torch.Tensor,
            dst:torch.Tensor,
            event_t:torch.Tensor
        )->list[tuple[int,int,int]]:
        """
        Convert a batch to directed events, preserving input order.
        """
        eventstream=[]
        for source,destination,timestamp in zip(
                src.tolist(),dst.tolist(),event_t.tolist()
            ):
                source=int(source)
                destination=int(destination)
                timestamp=int(timestamp)
                eventstream.append((source,destination,timestamp))
        return eventstream

    @staticmethod
    def get_TR_label_using_Time_Centric(
            graph_df:pd.DataFrame,
            batch_size:int
        ):
        """
        Input:
            graph_df: graph_df (bidirect)
            batch_size: int
        Return:
            TR_label: [batch_seq_len,N+1,N+1], boolean tensor
        """
        ### get graph_loader
        graph_dataset=TemporalGraphDataset(graph_df=graph_df)
        graph_loader=DataLoader(dataset=graph_dataset,batch_size=batch_size,shuffle=False)

        ### init TR_label
        n_node=int(max(graph_df["u"].max(),graph_df["i"].max()))
        TR_label=torch.zeros(
            (len(graph_loader),n_node+1,n_node+1),
            dtype=torch.bool
        )

        ### compute TR_label
        for seq_idx,(src,dst,event_t,_) in enumerate(graph_loader):
            eventstream=GraphUtils.convert_batch_to_eventstream(src=src,dst=dst,event_t=event_t)
            gamma_table=GraphAlgorithm.compute_TR_using_Time_Centric(
                eventstream=eventstream,
                n_node=n_node,
                gamma_table=None if seq_idx==0 else gamma_table
            ) # [N+1,N+1,2] int64 ndarray
            TR_label[seq_idx]=torch.from_numpy(gamma_table[:,:,0]).bool()
        return TR_label