# P-GNN and I-GNN are borrowed from https://github.com/LivXue/GNN4CMR/ and have been modified
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import sparse
from backbone_pyg import GCN, MultiHopGNN
from torch_geometric.utils import shuffle_node
# from backbone import GCN, GAT, SpGAT
# from utils import gen_negative_samples
from layers import AvgReadout, Discriminator
from utils import count_occurence, construct_graph, rwr, sparse_adj


class BIODGI(nn.Module):
    def __init__(self, attrs_dim, hiddens_dim, out_dim, dropout_rate):
        super().__init__()
        self.num_views = len(attrs_dim)
        self.models = nn.ModuleList(
            [GCN(attrs_dim[i], hiddens_dim[i], out_dim, dropout_rate) for i in range(self.num_views)])
        self.read = AvgReadout()
        self.dropout = nn.Dropout(dropout_rate)
        self.sigm = nn.Sigmoid()

        self.disc = Discriminator(out_dim)

    def forward(self, datas):
        pos_embeddings = [self.models[i](
            datas[i].x, datas[i].edge_index, datas[i].edge_weight)for i in range(self.num_views)]
        shuffled_x, _ = shuffle_node(datas[i].x)
        neg_embddings = [self.models[i](
            shuffled_x, datas[i].edge_index, datas[i].edge_weight) for i in range(self.num_views)]

        view_embeddings = [self.sigm(self.read(self.dropout(
            pos_embedding))) for pos_embedding in pos_embeddings]

        ret = self.disc(view_embeddings, pos_embeddings, neg_embddings)

        return ret

    # Detach the return variables
    def embed(self, datas):
        embeddings = [self.models[i](datas[i].x, datas[i].edge_index, datas[i].edge_weight)
                      for i in range(self.num_views)]
        return torch.cat(embeddings, dim=-1).detach()

    # def get_attention_weight(self):
    #     attention_weights = []
    #     for model in self.models:
    #         attention_weights.append(model.get_attention_weight())
    #     return attention_weights


class P_GNN(nn.Module):
    def __init__(self, labels, num_layers=2, hidden_ft=17, GNN='GAT', threshold=0.1, p=0.5, occurence_count_file=None):
        super().__init__()
        file_path = f'datasets/occurence_count/{occurence_count_file}'
        if os.path.exists(file_path):
            A, count = torch.load(file_path)
        else:
            A, count = count_occurence(labels)
        adj = self.construct_graph(A, count, threshold=threshold, p=p)
        self.attrs = rwr(adj).T
        adj = sparse.coo_matrix(adj)
        self.edge_index, self.edge_weight = sparse_adj(adj)
        assert GNN in ['GAT', 'GCN']
        self.GNN = GNN
        self.gnn = MultiHopGNN(self.attrs.shape[1], hidden_ft, num_layers, GNN)

    def forward(self):
        if self.GNN == "GCN":
            ret = self.gnn(self.attrs, self.edge_index, self.weight)
        else:
            ret = self.gnn(self.attrs, self.edge_index)


class SimpleFC(nn.Module):
    def __init__(self):
        pass

    def forward(self):
        pass
