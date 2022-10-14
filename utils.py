import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from GPUtil import showUtilization as gpu_usage
from numba import cuda


def gen_negative_samples(X, is_batch=False):
    # seed is set as 42
    np.random.seed(42)
    num_samples = X.shape[0]
    idx = np.random.permutation(num_samples)
    if is_batch:
        return X[:, idx, :]
    else:
        return X[idx, :]


def get_useful_sample_index(mat):
    # filter the columns filled with zeros and return the bool idx
    return mat.sum(axis=0) != 0


def build_consistency_loss(attention_weights):
    num_views = len(attention_weights)
    num_heads = attention_weights.shape[0]
    loss = 0
    loss_func = torch.nn.MSELoss(reduction='sum')
    for i in range(num_views - 1):
        for j in range(i + 1, num_views):
            loss += loss_func(attention_weights[i],
                              attention_weights[j]) / num_heads
    return loss


def free_gpu_cache():
    print("Initial GPU Usage")
    gpu_usage()

    torch.cuda.empty_cache()

    cuda.select_device(0)
    cuda.close()
    cuda.select_device(0)

    print("GPU Usage after emptying the cache")
    gpu_usage()
