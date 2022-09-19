import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from backbone import GCN,GAT,SpGAT
from utils import gen_negative_samples
from layers import AvgReadout, Discriminator


class BIODGI(nn.Module):
	def __init__(self, attrs_dim, hiddens_dim, out_dim, dropout, alpha, num_heads):
        super(BIODGI, self).__init__()
        self.num_views=len(num_views)
        self.models=nn.ModuleList([GAT(attrs_dim[i], hiddens_dim[i], outs_dim[i], dropout, alpha, num_heads) for i in range(self.num_views)])
        self.read = AvgReadout()
        self.dropout = nn.Dropout(dropout)
        self.sigm = nn.Sigmoid()

        self.disc = Discriminator(out_dim)

    def forward(self, attrs, neg_embddings, adjs, sparse, msk, samp_bias1, samp_bias2):
        pos_embeddings=[self.models[i](attrs[i],adjs[i]) for i in range(self.num_views)]
        neg_embddings=[self.models[i](gen_negative_samples(attrs[i]),adjs[i]) for i in range(self.num_views)]
        
        view_embeddings=[self.sigm(self.read(self.dropout(pos_embedding))) for pos_embedding in pos_embeddings]
        
        ret = self.disc(view_embeddings,pos_embeddings,neg_embddings)

        return ret

    # Detach the return variables
    def embed(self, attrs, adjs):
        embeddings=[self.models[i](attrs[i],adjs[i]) for i in range(self.num_views)]
        return torch.cat(embeddings, dim=-1).detach()