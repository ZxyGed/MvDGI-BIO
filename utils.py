import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import sparse
from GPUtil import showUtilization as gpu_usage
from numba import cuda


def gen_negative_samples(X):
    # seed is set as 42
    np.random.seed(42)
    num_samples = X.shape[0]
    idx = np.random.permutation(num_samples)
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


def rwr(graph, restart_prob=0.5):
    n = graph.shape[0]
    graph = graph - np.diag(np.diag(graph))
    graph = graph + np.diag(np.sum(graph, axis=0) == 0)
    norm_graph = graph / np.sum(graph, axis=0)
    ret = np.matmul(np.linalg.inv(np.eye(n) - (1 - restart_prob)
                                  * norm_graph), restart_prob * np.eye(n))
    return ret


def count_occurence(labels, is_save=False, file_name=None):
    num_class = labels.shape[1]
    A = np.zeros((num_class, num_class))
    for i in range(num_class):
        for j in range(i, num_class):
            if i == j:
                continue
            else:
                A[j, i] = A[i, j] = np.sum(labels[:, i] & labels[:, j])
    count = np.sum(labels, axis=0)
    if is_save:
        torch.save((A, count), f'datasets/occurence_count/{file_name}')
    return A, count


def construct_graph(A, count, threshold=0.5, p=0.5):
    # A, count = torch.load(f'datasets/occurence_count/{file_name}')
    A = A / count
    A[A < threshold] = 0
    A[A >= threshold] = p
    A = A + np.eye(num_class)
    return A


def sparse_adj(adj):
    adj = sparse.coo_matrix(adj)
    edge_index = torch.tensor(np.vstack((adj.row, adj.col)), dtype=torch.long)
    edge_weight = torch.tensor(adj.data.reshape(-1, 1), dtype=torch.float)
    return edge_index, edge_weight


def free_gpu_cache():
    print("Initial GPU Usage")
    gpu_usage()

    torch.cuda.empty_cache()

    cuda.select_device(0)
    cuda.close()
    cuda.select_device(0)

    print("GPU Usage after emptying the cache")
    gpu_usage()
