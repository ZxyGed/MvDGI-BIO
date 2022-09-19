import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from model import BIODGI


dataset = 'yeast'

if dataset == 'yeast':
    args = {'name': 'yeast', 'input_dim': 6400, 'hidden_dim': 512, 'embedding_dim': 32,
                    'num_epoch': 1200, 'learn_rate': 0.005, 'dropout_rate': 0.1, 'beta': 1.0, 'patience': 35}
elif dataset == 'human':
    args = {'name': 'human', 'input_dim': 18362, 'hidden_dim': 512, 'embedding_dim': 32,
                    'num_epoch': 1200, 'learn_rate': 0.005, 'dropout_rate': 0.1, 'beta': 1.0, 'patience': 35}
else:
    print('wrong data type')

attrs = 
