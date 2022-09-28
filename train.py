import os
import numpy as np
from scipy import sparse
import torch
import torch.nn as nn
from model import BIODGI
from utils import build_consistency_loss


dataset = 'yeast'
num_views = 6

# folder to preserve model parameters
if not os.path.exists('model'):
    os.mkdir('model')

if dataset == 'yeast':
    args = {'name': 'yeast', 'num_nodes': 6400, 'input_dim': 6400, 'hidden_dim': 512, 'embedding_dim': 32,
                    'num_epoch': 1200, 'learning_rate': 0.005, 'dropout_rate': 0.1, 'alpha': 0.2, 'num_heads': 8, 'patience': 35, 'seed': 42}
elif dataset == 'human':
    args = {'name': 'human', 'num_nodes': 6400, 'input_dim': 18362, 'hidden_dim': 512, 'embedding_dim': 32,
                    'num_epoch': 1200, 'learning_rate': 0.005, 'dropout_rate': 0.1, 'alpha': 0.2, 'num_heads': 8, 'patience': 35, 'seed': 42}
else:
    print('wrong data type')

views_list = ['coexpression', 'cooccurence', 'database',
              'experimental', 'fusion', 'neighborhood']


def load_data(filename): return torch.FloatTensor(
    np.array(sparse.load_npz(file_name).todense())).to(device)


attrs = [load_data('data/attrs/%s_%s.npz' %
                   (args['name'], view)) for view in views_list]
adjs = [load_data('data/adjs/%s_%s.npz' %
                  (args['name'], view)) for view in views_list]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

random.seed(args['seed'])
np.random.seed(args['seed'])
torch.manual_seed(args['seed'])
torch.cuda.manual_seed(args['seed'])

torch.cuda.empty_cache()

attrs_dim = [args['input_dim']] * num_views
hiddens_dim = [args['hidden_dim']] * num_views
out_dim = args['embedding_dim']
model = BIODGI(attrs_dim, hiddens_dim, out_dim,
               args['dropout_rate'], alpha=args['alpha'], num_heads=args['num_heads']).to(device)
optimiser = torch.optim.Adam(model.parameters(), lr=args['learning_rate'])

b_xent = nn.BCEWithLogitsLoss()
xent = nn.CrossEntropyLoss()
bad_counter = 0
best_performance = float('inf')
best_epoch = args['num_epoch'] + 1

target_label_1 = torch.ones(num_views * args['num_nodes'], 1)
target_label_0 = torch.zeros(num_views * args['num_nodes'], 1)
target_label = torch.cat((target_label_1, target_label_0), 0).to(self.device)

for epoch in range(args['num_epoch']):
    model.train()
    optimiser.zero_grad()
    logits = model(attrs, adjs)
    loss = b_xent(logits, target_label)
    loss += build_consistency_loss(model.get_attention_weight())
    if loss < best_performance:
        best_performance = loss
        best_epoch = epoch
        bad_counter = 0
        torch.save(model.state_dict(), 'model/%s.pkl' % args['name'])
    else:
        bad_counter += 1
    if bad_counter == args['patience']:
        print('Early Stoping at epoch %d' % epoch)
        break

    loss.backward()
    optimiser.step()
