import os
import numpy as np
from scipy import sparse
import torch
import torch.nn as nn
from model import BIODGI


dataset = 'yeast'

if dataset == 'yeast':
    args = {'name': 'yeast', 'input_dim': 6400, 'hidden_dim': 512, 'embedding_dim': 32,
                    'num_epoch': 1200, 'learning_rate': 0.005, 'dropout_rate': 0.1, 'alpha': 0.2, 'num_heads': 8, 'patience': 35}
elif dataset == 'human':
    args = {'name': 'human', 'input_dim': 18362, 'hidden_dim': 512, 'embedding_dim': 32,
                    'num_epoch': 1200, 'learning_rate': 0.005, 'dropout_rate': 0.1, 'alpha': 0.2, 'num_heads': 8, 'patience': 35}
else:
    print('wrong data type')

views_list = ['coexpression', 'cooccurence', 'database',
              'experimental', 'fusion', 'neighborhood']


def load_data(file_name): return torch.FloatTensor(
    np.array(sparse.load_npz(file_name).todense())).to(device)


attrs = [load_data('data/attrs/%s_%s.npz' %
                   (args['name'], view)) for view in views_list]
adjs = [load_data('data/adjs/%s_%s.npz' %
                  (args['name'], view)) for view in views_list]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

attrs_dim = [args['input_dim']] * 6
hiddens_dim = [args['hidden_dim']] * 6
out_dim = args['embedding_dim']
model = BIODGI(attrs_dim, hiddens_dim, out_dim,
               args['dropout_rate'], alpha=args['alpha'], num_heads=args['num_heads']).to(device)
optimiser = torch.optim.Adam(model.parameters(), lr=args['learning_rate'])
