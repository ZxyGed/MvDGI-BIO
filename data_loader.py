import numpy as np
from scipy import sparse


def load_attrs(attrs_folder, dataset, subset):
    '''
    dataset: human/yeast
    subset:  neighborhood/fusion/experimental/database/cooccurence/coexpression
    '''
    X = sparse.load_npz('%s/%s_%s.npz' % (attrs_folder, dataset, subset))
    return X


def load_labels(labels_folder, dataset, level):
    '''
    dataset: human/yeast
    level:   if the dataset is human, then combine bp/mf/cc with 1130/31100/101300, such as bp_1130
             if the dataset is yeast, then level1/level2/level3
    '''
    Y = np.load('%s/%s_%s_label.npy' %
                (labels_folder, dataset, level))
    return Y


if __name__ == '__main__':
    attrs_folder = r'data/attrs'
    labels_folder = r'data/labels'
    dataset = 'human'
    subset = 'neighborhood'
    level = 'bp_1130'
    X = load_attrs(attrs_folder, dataset, subset)
    Y = load_labels(labels_folder, dataset, level)
