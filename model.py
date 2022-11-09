# P-GNN and I-GNN are borrowed from https://github.com/LivXue/GNN4CMR/ and have been modified
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
    def __init__(self, labels, num_layers, GNN='GAT'):
        super().__init__()
        self.labels

    def __init__(self, img_input_dim=4096, text_input_dim=1024, minus_one_dim=1024, num_classes=10, in_channel=300, t=0,
                 adj_file=None, inp=None, GNN='GAT', n_layers=4):
        super().__init__()
        self.img_net = ImgNN(img_input_dim, minus_one_dim)
        self.text_net = TextNN(text_input_dim, minus_one_dim)
        self.img2text_net = TextDec(minus_one_dim, text_input_dim)
        self.text2img_net = ImgDec(minus_one_dim, img_input_dim)
        self.img_md_net = ModalClassifier(img_input_dim)
        self.text_md_net = ModalClassifier(text_input_dim)
        self.num_classes = num_classes
        if GNN == 'GAT':
            self.gnn = GraphAttentionLayer
        elif GNN == 'GCN':
            self.gnn = GraphConvolution
        else:
            raise NameError("Invalid GNN name!")
        self.n_layers = n_layers

        self.relu = nn.LeakyReLU(0.2)
        self.lrn = [self.gnn(in_channel, minus_one_dim)]
        for i in range(1, self.n_layers):
            self.lrn.append(self.gnn(minus_one_dim, minus_one_dim))
        for i, layer in enumerate(self.lrn):
            self.add_module('lrn_{}'.format(i), layer)
        self.hypo = nn.Linear(self.n_layers * minus_one_dim, minus_one_dim)

        _adj = torch.FloatTensor(gen_A(num_classes, t, adj_file))
        if GNN == 'GAT':
            self.adj = Parameter(_adj, requires_grad=False)
        else:
            self.adj = Parameter(gen_adj(_adj), requires_grad=False)

        if inp is not None:
            self.inp = Parameter(inp, requires_grad=False)
        else:
            self.inp = Parameter(torch.rand(num_classes, in_channel))
        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

    def forward(self, feature_img, feature_text):
        view1_feature = self.img_net(feature_img)
        view2_feature = self.text_net(feature_text)

        layers = []
        x = self.inp
        for i in range(self.n_layers):
            x = self.lrn[i](x, self.adj)
            if self.gnn == GraphConvolution:
                x = self.relu(x)
            layers.append(x)
        x = torch.cat(layers, -1)
        x = self.hypo(x)

        norm_img = torch.norm(view1_feature, dim=1)[:, None] * torch.norm(x, dim=1)[None, :] + 1e-6
        norm_txt = torch.norm(view2_feature, dim=1)[:, None] * torch.norm(x, dim=1)[None, :] + 1e-6
        x = x.transpose(0, 1)
        y_img = torch.matmul(view1_feature, x)
        y_text = torch.matmul(view2_feature, x)
        y_img = y_img / norm_img
        y_text = y_text / norm_txt

        view1_feature_view2 = self.img2text_net(view1_feature)
        view2_feature_view1 = self.text2img_net(view2_feature)
        view1_modal_view1 = self.img_md_net(feature_img)
        view2_modal_view1 = self.img_md_net(view2_feature_view1)
        view1_modal_view2 = self.text_md_net(view1_feature_view2)
        view2_modal_view2 = self.text_md_net(feature_text)

        return view1_feature, view2_feature, y_img, y_text, x.transpose(0, 1), \
            view1_modal_view1, view2_modal_view1, view1_modal_view2, view2_modal_view2


class SimpleFC(nn.Module):
    def __init__(self):
        pass

    def forward(self):
        pass
