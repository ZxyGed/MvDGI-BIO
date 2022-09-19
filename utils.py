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
        return X[:,idx,:]
    else:
        return X[idx,:]