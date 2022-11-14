# P-GNN and I-GNN are borrowed from https://github.com/LivXue/GNN4CMR/ and have been modified
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import sparse
from backbone_pyg import GCN, MultiHopGNN
# from torch_geometric.utils import shuffle_node #2.0.4 don't support this function
# from backbone import GCN, GAT, SpGAT
from layers import AvgReadout, Discriminator
from utils import count_occurence, construct_graph, rwr, sparse_adj, gen_negative_samples


class BIODGI(nn.Module):
    def __init__(self, attrs_dim, hiddens_dim, out_dim, dropout_rate):
        super().__init__()
        self.num_views = len(attrs_dim)
        self.models = nn.ModuleList(
            [GCN(attrs_dim[i], hiddens_dim[i], out_dim, dropout_rate) for i in range(self.num_views)])
        self.read = AvgReadout()
        self.dropout = nn.Dropout(dropout_rate)
        self.sigm = nn.Sigmoid()

        self.W = nn.Linear(self.num_views * out_dim, out_dim)
        # self.view_weight=nn.Linear(num_views)
        # self.W=nn.Linear(num_views*out_dim, out_dim)

        self.disc = Discriminator(out_dim)

    def forward(self, datas):
        pos_embeddings = []
        neg_embddings = []
        for i in range(self.num_viewyis):
            pos_embedding = self.models[i](
                datas[i].x, datas[i].edge_index, datas[i].edge_weight)
            pos_embeddings.append(pos_embedding)

            shuffled_x = gen_negative_samples(datas[i].x)
            neg_embdding = self.models[i](
                shuffled_x, datas[i].edge_index, datas[i].edge_weight)
            neg_embddings.append(neg_embdding)

        view_embeddings = []
        num_nodes = datas[0].num_nodes
        for i in range(self.num_views):
            view_embedding = self.sigm(self.read(self.dropout(pos_embedding)))
            # bilinear needs the input1 and 2 has the same size except for the last dim
            view_embeddings.append(view_embedding.repeat(num_nodes, 1))

        ret = self.disc(view_embeddings, pos_embeddings, neg_embddings)

        return ret

    # Detach the return variables
    def embed(self, datas):
        embeddings = [self.models[i](datas[i].x, datas[i].edge_index, datas[i].edge_weight)
                      for i in range(self.num_views)]
        embedding = self.W(torch.cat(embeddings, dim=-1))
        embedding = F.softmax(embedding)
        return embedding

    # def get_attention_weight(self):
    #     attention_weights = []
    #     for model in self.models:
    #         attention_weights.append(model.get_attention_weight())
    #     return attention_weights


class P_GNN(nn.Module):
    def __init__(self, labels, num_layers=2, hidden_ft=17, embedding_ft=17, GNN='GAT', dropout_rate=0.1, neg_slop=0.2, threshold=0.1, p=0.5, occurence_count_file=None):
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
        self.gnn = MultiHopGNN(
            self.attrs.shape[1], hidden_ft, embedding_ft, num_layers, GNN, dropout_rate, neg_slop)

    def forward(self):
        if self.GNN == "GCN":
            self.class_embeddings = self.gnn(
                self.attrs, self.edge_index, self.weight)
        else:
            self.class_embeddings = self.gnn(self.attrs, self.edge_index)
        return self.class_embeddings

    def predict(self, sample_embeddings):
        norm_val = torch.norm(sample_embeddings, dim=1)[
            :, None] * torch.norm(self.class_embeddings, dim=1)[None, :] + 1e-6
        return torch.matmul(sample_embeddings, self.class_embeddings.T) / norm_val


class SimpleFC(nn.Module):
    def __init__(self, in_dim, out_dim):
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        x = self.fc(x)
        return F.log_softmax(x, dim=1)
