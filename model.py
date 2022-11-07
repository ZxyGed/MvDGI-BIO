import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from backbone_pyg import GCN
from torch_geometric.utils import shuffle_node
# from backbone import GCN, GAT, SpGAT
# from utils import gen_negative_samples
from layers import AvgReadout, Discriminator


class BIODGI(nn.Module):
    def __init__(self, attrs_dim, hiddens_dim, out_dim, dropout_rate):
        super(BIODGI, self).__init__()
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
