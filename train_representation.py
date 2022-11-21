# coding=utf-8
import os
import sys
import time
import random
import argparse
import logging
import hashlib

import yaml
import numpy as np
from scipy import sparse
import torch
import torch.nn as nn

from model import BIODGI, P_GNN, SimpleFC
from dataset import Yeast, Human
from metrics import evaluate_performance


random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

parser = argparse.ArgumentParser(description='Training Config', add_help=False)

parser.add_argument('-c', '--config_yaml', default="./hyper_parameters/train/yeast.yaml",
                    type=str, metavar='FILE', help='YAML config file specifying default arguments')
parser.add_argument('-l', '--level', default='level1', type=str)
parser.add_argument('-d', '--domain', default='bp', type=str)
parser.add_argument('-s', '--size', default='1130', type=str)
parser.add_argument('-rg', '--re_generate', default=False, type=bool)

parser.add_argument('-ep', '--num_epoch', default=1200, type=int)
parser.add_argument('-lr', '--learning_rate', default=0.005, type=float)
parser.add_argument('-dr', '--dropout_rate', default=0.1, type=float)

parser.add_argument('-hd', '--hidden_dim', default=0.1, type=int)
parser.add_argument('-ed', '--embedding_dim', default=0.1, type=int)
parser.add_argument('-pa', '--patience', default=0.1, type=int)


args_temp = parser.parse_args()
with open(args_temp.config_yaml, 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)
    parser.set_defaults(**cfg)
args = parser.parse_args()

# folder to preserve model parameters and embeddings
if not os.path.exists('model'):
    os.mkdir('model')
if not os.path.exists('embeddings'):
    os.mkdir('embeddings')
if not os.path.exists('results'):
    os.mkdir('results')

args_dict = vars(args)
selcted_params = ['hidden_dim', 'embedding_dim',
                  'dropout_rate', 'learning_rate']
selcted_params_dict = {key: args_dict[key] for key in selcted_params}
tale = hashlib.md5(str(sorted(selcted_params_dict.items())
                       ).encode('utf-8')).hexdigest()

if args.name == 'yeast':
    head = f"{args.name}_{args.level}_{args.type}"
else:
    head = f"{args.name}_{args.domain}_{args.size}_{args.type}"

identity_name = f"{head}_{tale}"


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if args.name == 'yeast':
    datas = Yeast(args.root, level=args.level,
                  re_generate=args.re_generate).data
elif args.name == 'human':
    datas = Human(args.root, domain=args.domain, size=args.size,
                  re_generate=args.re_generate).data
else:
    raise ValueError(
        f'Dataset({args.name}) found, but expected either yeast or human')

for d in datas:
    d = d.to(device)
num_views = 6
num_nodes = datas[0].num_nodes

attrs_dim = [d.num_features for d in datas]
hiddens_dim = [args.hidden_dim] * num_views
out_dim = args.embedding_dim

# print(f'embdding_dim:{out_dim}')
# print(f'attrs_dim:{attrs_dim}')

model = BIODGI(attrs_dim, hiddens_dim, out_dim, args.dropout_rate).to(device)
optimiser = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

b_xent = nn.BCEWithLogitsLoss()
xent = nn.CrossEntropyLoss()
bad_counter = 0
best_performance = float('inf')
best_epoch = args.num_epoch + 1

target_label_1 = torch.ones(num_views * num_nodes, 1)
target_label_0 = torch.zeros(num_views * num_nodes, 1)
target_label = torch.cat((target_label_1, target_label_0), 0).to(device)
# target_label = torch.cat((target_label_0, target_label_1), 0).to(device)

t = time.time()
losses = []
for epoch in range(args.num_epoch):
    # gc.collect()
    # torch.cuda.empty_cache()
    # free_gpu_cache()
    model.train()
    optimiser.zero_grad()
    logits = model(datas)
    loss = b_xent(logits, target_label)
    if not epoch % 10:
        print(f'epoch:{epoch}, loss:{loss.cpu().detach().numpy()}')
    losses.append(loss.cpu().detach().numpy())
    # loss += build_consistency_loss(model.get_attention_weight())
    if loss < best_performance:
        best_performance = loss
        best_epoch = epoch
        bad_counter = 0
        torch.save(model.state_dict(), f'model/{identity_name}.pkl')
    else:
        bad_counter += 1
    if bad_counter == args.patience:
        print('Early Stoping at epoch %d' % epoch)
        break

    loss.backward()
    optimiser.step()

print("Optimization Finished!")
print("Total time elapsed: %.4fs" % (time.time() - t))

print("Loading %dth epoch" % best_epoch)
model.load_state_dict(torch.load("model/%s.pkl" % args.name))
embeddings = model.embed(datas).cpu().detach().numpy()
# print(f'embeddings.shape:{embeddings.shape}')

np.savez(f"embeddings/{identity_name}.npz",
         X=embeddings, params=selcted_params_dict)
