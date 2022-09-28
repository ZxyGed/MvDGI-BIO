import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def gen_negative_samples(X, is_batch=False):
    # seed is set as 42
    np.random.seed(42)
    num_samples = X.shape[0]
    idx = np.random.permutation(num_samples)
    if is_batch:
        return X[:, idx, :]
    else:
        return X[idx, :]


def build_consistency_loss(attention_weights):
    num_views = len(attention_weights)
    num_heads = attention_weights.shape[0]
    loss = 0
    loss_func = torch.nn.MSELoss(reduction='sum')
    for i in range(num_views - 1):
        for j in range(i + 1: num_views):
            loss += loss_func(attention_weights[i],
                              attention_weights[j]) / num_heads
    return loss
