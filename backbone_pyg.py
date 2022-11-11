import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv


class GCN(torch.nn.Module):
        # default activation function is relu
    def __init__(self, in_ft, hidden_ft, out_ft, dropout_rate=0.1, neg_slop=0.2):
        super().__init__()
        self.conv1 = GCNConv(in_ft, hidden_ft)
        self.conv2 = GCNConv(hidden_ft, out_ft)
        self.relu = nn.LeakyReLU(neg_slop)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, edge_index, edge_weight):
        x = self.conv1(x, edge_index, edge_weight)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index, edge_weight)

        return F.log_softmax(x, dim=1)


class MultiHopGNN(torch.nn.Module):
    def __init__(self, in_ft, hidden_ft, embedding_ft, num_layers, GNN, dropout_rate=0.1, neg_slop=0.2):
        super().__init__()
        assert GNN in ["GCN", "GAT"]
        self.GNN = GNN
        if GNN == "GCN":
            self.layers = [
                GCNConv(in_ft, hidden_ft, add_self_loops=False, normalize=False)]
            for i in range(num_layers - 1):
                self.layers.append(GCNConv(hidden_ft, hidden_ft))
        elif CNN == "GAT":
            self.layers = [GATConv(
                in_ft, hidden_ft, add_self_loops=False, negative_slop=neg_slop, dropout=dropout_rate)]
            for i in range(num_layers - 1):
                self.layers.append(GATConv(in_ft, hidden_ft))
        else:
            raise ValueError(
                f'GNN {GNN} found, but expected either GCN or GAT'
            )

        self.num_layers = num_layers
        self.neg_slop = neg_slop
        self.dropout_rate = dropout_rate
        self.hypo = nn.Linear(self.num_layers * hidden_ft, embedding_ft)

    def forward(self, x, edge_index, edge_weight=None):
        layers_ret = []
        if self.GNN == "GCN":
            for i in range(self.num_layers):
                x = self.layers[i](x, edge_index, edge_weight)
                x = F.leaky_relu(x, self.neg_slop)
                x = F.dropout(x, self.dropout_rate)
                layers.append(x)
        elif self.GNN == "GAT":
            for i in range(self.num_layers):
                # leakyrelu and drop are in GATConv settings
                x = self.layers[i](x, edge_index)
                layers.append(x)
        else:
            pass
        x = torch.cat(layers, -1)
        return self.hypo(x)
