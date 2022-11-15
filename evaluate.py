import sqlite3
import argparse

import torch
import yaml
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import zero_one_loss
# from sklearn.preprocessing import StandardScaler, minmax_scale

from metrics import evaluate_performance


parser = argparse.ArgumentParser(description='Testing Config', add_help=False)

parser.add_argument('-c', '--config_yaml', default="./hyper_parameters/test/test_params.yaml",
                    type=str, metavar='FILE', help='YAML config file specifying default arguments')
parser.add_argument('-na', '--name', default='yeast', type=str)
parser.add_argument('-l', '--level', default='level1', type=str)
parser.add_argument('-d', '--domain', default='bp', type=str)
parser.add_argument('-s', '--size', default='1130', type=str)

args_temp = parser.parse_args()
assert args_temp.name in ['yeast', 'human']
with open(args_temp.config_yaml, 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)
    parser.set_defaults(**cfg['data_params'])
    parser.set_defaults(lgb_params=cfg['lgb_params'])
args = parser.parse_args()

if args.name == 'yeast':
    path = f'datasets/labels/{args.name}_{args.level}.npz'
else:
    path = f'datasets/labels/{args.name}_{args.domain}_{args.size}.npz'

content = np.load(path)
Y = content['y']
X = np.load(f'embeddings/{args.name}.npy')

maupr_all = []
Maupr_all = []
acc_all = []
f1_all = []
zero_one_loss_all = []


for j in range(10):

    shuffle_indices = np.random.permutation(np.arange(len(Y)))
    emb_shuffled = X[shuffle_indices, :]
    anno_shuffled = Y[shuffle_indices]

    test_sample_percentage = 0.1
    test_sample_index = int(test_sample_percentage * float(len(Y)))
    print(test_sample_index)
    X_test, X_dev, X_train = emb_shuffled[:test_sample_index, :], emb_shuffled[test_sample_index:2 *
                                                                               test_sample_index, :], emb_shuffled[2 * test_sample_index:, :]  # (6,3555,500),(6,888,500)
    y_test, y_dev, y_train = anno_shuffled[:test_sample_index, :], anno_shuffled[test_sample_index:2 *
                                                                                 test_sample_index, :], anno_shuffled[2 * test_sample_index:, :]

    y_score = np.zeros_like(y_test)
    y_pred = np.zeros_like(y_test)

    for i in range(y_train.shape[1]):
        train_data = lgb.Dataset(X_train, label=y_train[:, i])
        validation_data = lgb.Dataset(X_dev, label=y_dev[:, i])
        clf = lgb.train(args.lgb_params, train_data,
                        valid_sets=[validation_data])
        y_score_sub = clf.predict(X_test)

        y_pred_sub = (y_score_sub >= 0.5) * 1

        y_score[:, i] = y_score_sub
        y_pred[:, i] = y_pred_sub

    result = evaluate_performance(y_test, y_score, y_pred)
    zero = zero_one_loss(y_test, y_pred)

    maupr_all.append(result['m-aupr'])
    Maupr_all.append(result['M-aupr'])
    acc_all.append(result['acc'])
    f1_all.append(result['F1'])
    zero_one_loss_all.append(zero)


print('acc:', np.mean(acc_all), np.std(acc_all))
print('f1:', np.mean(f1_all), np.std(f1_all))
print('m-aupr:', np.mean(maupr_all), np.std(maupr_all))
print('M-aupr:', np.mean(Maupr_all), np.std(Maupr_all))
print('subset zero_one loss:', np.mean(
    zero_one_loss_all), np.std(zero_one_loss_all))
