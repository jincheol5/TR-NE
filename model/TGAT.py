import torch
import torch.nn as nn
from graph import GNN_Graph
from module import TimeEncoder,GraphAttnEmbedding,MLP

class TGAT(nn.Module):
    def __init__(self,
            node_dim:int,
            edge_dim:int,
            time_dim:int,
            latent_dim:int,
            embed_dim:int,
            n_layer:int,
            n_neighbor:int,
            n_head:int
        ):
        super().__init__()
        self.node_dim=node_dim
        self.edge_dim=edge_dim
        self.time_dim=time_dim
        self.latent_dim=latent_dim
        self.embed_dim=embed_dim
        self.n_layer=n_layer
        self.n_neighbor=n_neighbor
        self.n_head=n_head

        # time encoder
        self.time_encoder=TimeEncoder(time_dim=time_dim)

        # encoder
        self.encoder=GraphAttnEmbedding(
            node_dim=node_dim,
            edge_dim=edge_dim,
            time_dim=time_dim,
            latent_dim=latent_dim,
            embed_dim=embed_dim,
            n_layer=n_layer,
            n_neighbor=n_neighbor,
            n_head=n_head,
            time_encoder=self.time_encoder,
            use_memory=False
        )

        # decoder
        self.decoder=MLP(
            input_dim=embed_dim,
            latent_dim=latent_dim,
            output_dim=1
        )

    def set_graph(self,graph:GNN_Graph):
        self.graph=graph
        self.n_node=graph.n_node
        self.encoder.set_graph(graph=graph)

    def embedding(self,
            tar:torch.Tensor,
            tar_t:torch.Tensor
        ):
        embedded_tar_vec=self.encoder.compute_embedding(
            tar=tar,
            tar_t=tar_t,
            n_layer=self.n_layer
        )
        return embedded_tar_vec

    def forward(self,
            dst:torch.Tensor,
            event_t:torch.Tensor
        ):
        ### embedding
        embedded_vec=self.embedding(
            tar=dst,
            tar_t=event_t
        )

        ### decode 
        pred_logit=self.decoder(embedded_vec) # [B,1]
        return pred_logit.squeeze(-1) # [B,]