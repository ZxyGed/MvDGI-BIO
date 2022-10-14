import os
import gc
import time
import yaml
import random
import numpy as np
from scipy import sparse
import torch
import torch.nn as nn
from model import BIODGI
from utils import build_consistency_loss, free_gpu_cache


dataset = 'yeast'
num_views = 6
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# folder to preserve model parameters and embeddings
if not os.path.exists('model'):
    os.mkdir('model')
if not os.path.exists('embeddings'):
    os.mkdir('embeddings')

with open(f"hyper_parameters/train/{dataset}.yaml", 'r', encoding='utf-8') as f:
    args = yaml.safe_load(f)


views_list = ['coexpression', 'cooccurence', 'database',
              'experimental', 'fusion', 'neighborhood']


def load_data(file_name): return torch.FloatTensor(
    np.array(sparse.load_npz(file_name).todense())).to(device)


attrs = [load_data('data/attrs/%s_%s.npz' %
                   (args['name'], view)) for view in views_list]
adjs = [load_data('data/adjs/%s_%s.npz' %
                  (args['name'], view)) for view in views_list]


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
target_label = torch.cat((target_label_1, target_label_0), 0).to(device)

t = time.time()
for epoch in range(args['num_epoch']):
    # gc.collect()
    # torch.cuda.empty_cache()
    free_gpu_cache()
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

print("Optimization Finished!")
print("Total time elapsed: %.4fs" % (time.time() - t))

print("Loading %dth epoch" % best_epoch)
model.load_state_dict(torch.load("model/%s.pkl" % args['name']))
embeddings = model.embed(attrs, adjs).detach().numpy()
np.save("embeddings/%s.npy" % args['name'], embeddings)
