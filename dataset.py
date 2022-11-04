import os

import numpy as np
from scipy import sparse
from typing import Callable, List, Optional, Union
import torch
from torch_geometric.data import Data, InMemoryDataset


class Yeast:
    node_num = 6400
    attr_dir = 'data/attrs'
    adj_dir = 'data/adjs'
    label_dir = 'data/labels'
    levels = ['level1', 'level2', 'level3', 'all']
    views_list = [
        'coexpression', 'cooccurence', 'database', 'experimental', 'fusion',
        'neighborhood'
    ]

    def __init__(self,
                 root: str,
                 level: str = 'all',
                 re_generate: bool = False):
        assert level in self.levels
        self.root = root
        self.level = level
        super().__init__()
        if not os.path.exists(root):
            os.makedirs(root)
        path = f'{root}/yeast_{level}.pt'
        if re_generate or not os.path.exists(path):
            self.process(level)
        self.data, self.y, self.selected_idx = torch.load(path)

    @property
    def num_classes(self) -> int:
        return self.y.shape[1] if self.level != 'all' else -1

    def filter_level(self, level):
        label = np.load(f'{self.label_dir}/yeast_{level}_label.npy')
        temp = np.sum(label, axis=0)
        return label.T, temp != 0

    def process_filenames(self, selected_idx):
        data_list = []
        for view in self.views_list:
            adj_sparse = sparse.load_npz(f'{self.adj_dir}/yeast_{view}.npz')
            # after rwr, each col represents a node's attr, so needs to be transposed to fit for Data
            attr_dense = np.array(
                sparse.load_npz(
                    f'{self.attr_dir}/yeast_{view}.npz').todense().T)
            adj = np.array(adj_sparse.todense())
            sub_adj = sparse.coo_matrix(adj[selected_idx, selected_idx])
            edge_index = torch.tensor(np.vstack((sub_adj.row, sub_adj.col)),
                                      dtype=torch.long)
            edge_weight = torch.tensor(sub_adj.data.reshape(-1, 1),
                                       dtype=torch.float)
            sub_attr = torch.tensor(attr_dense[selected_idx],
                                    dtype=torch.float)
            data = Data(x=sub_attr,
                        edge_index=edge_index,
                        edge_weight=edge_weight)
            data_list.append(data)
        return data_list

    def process(self, level):
        if level != 'all':
            y, selected_idx = self.filter_level(level)
            data_list = self.process_filenames(selected_idx)
            torch.save((data_list, y[selected_idx, :], selected_idx),
                       f'{self.root}/yeast_{level}.pt')
        elif level == 'all':
            data_list = self.process_filenames(
                np.array([True] * self.node_num))
            torch.save((data_list, None, None), f'{self.root}/yeast_all.pt')
        else:
            raise ValueError(
                f'level {level} found, but expected either level1, level2, level3 or all'
            )


class Human:
    node_num = 18362
    attr_dir = 'data/attrs'
    adj_dir = 'data/adjs'
    label_dir = 'data/labels'
    domains = ['bp', 'cc', 'mf', 'all']
    sizes = ['1130', '31100', '101300']
    views_list = [
        'coexpression', 'cooccurence', 'database', 'experimental', 'fusion',
        'neighborhood'
    ]

    def __init__(self,
                 root: str,
                 domain: str = 'all',
                 size: str='1130',
                 re_generate: bool = False):
        assert domain in self.domains
        assert size in self.sizes
        self.root = root
        self.domain = domain
        self.size = size
        super().__init__()
        if not os.path.exists(root):
            os.makedirs(root)
        if domain != 'all':
            path = f'{root}/human_{domain}_{size}.pt'
        else:
            path = f'{root}/human_all.pt'
        if re_generate or not os.path.exists(path):
            self.process(domain, size)
        self.data, self.y, self.selected_idx = torch.load(path)

    @property
    def num_classes(self) -> int:
        return self.y.shape[1] if self.domain != 'all' else -1

    def filter_level(self, domain, size):
        label = np.load(f'{self.label_dir}/human_{domain}_{size}_label.npy')
        temp = np.sum(label, axis=0)
        return label.T, temp != 0

    def process_filenames(self, selected_idx):
        data_list = []
        for view in self.views_list:
            adj_sparse = sparse.load_npz(f'{self.adj_dir}/human_{view}.npz')
            # after rwr, each col represents a node's attr, so needs to be transposed to fit for Data
            attr_dense = np.array(
                sparse.load_npz(
                    f'{self.attr_dir}/human_{view}.npz').todense().T)
            adj = np.array(adj_sparse.todense())
            sub_adj = sparse.coo_matrix(adj[selected_idx, selected_idx])
            edge_index = torch.tensor(np.vstack((sub_adj.row, sub_adj.col)),
                                      dtype=torch.long)
            edge_weight = torch.tensor(sub_adj.data.reshape(-1, 1),
                                       dtype=torch.float)
            sub_attr = torch.tensor(attr_dense[selected_idx],
                                    dtype=torch.float)
            data = Data(x=sub_attr,
                        edge_index=edge_index,
                        edge_weight=edge_weight)
            data_list.append(data)
        return data_list

    def process(self, domain, size):
        if domain != 'all':
            y, selected_idx = self.filter_level(domain, size)
            data_list = self.process_filenames(selected_idx)
            torch.save((data_list, y[selected_idx, :], selected_idx),
                       f'{self.root}/human_{domain}_{size}.pt')
        elif domain == 'all':
            data_list = self.process_filenames(
                np.array([True] * self.node_num))
            torch.save((data_list, None, None), f'{self.root}/human_all.pt')
        else:
            raise ValueError(
                f'level {level} found, but expected either level1, level2, level3 or all'
            )
