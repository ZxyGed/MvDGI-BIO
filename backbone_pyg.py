import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
        # default activation function is relu
    def __init__(self, in_ft, hidden_ft, out_ft, dropout_rate):
        super().__init__()
        self.conv1 = GCNConv(in_ft, hidden_ft)
        self.conv2 = GCNConv(hidden_ft, out_ft)
        self.dropout_rate = dropout_rate

    def forward(self, x, edge_index, edge_weight):
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)

        return F.log_softmax(x, dim=1)
