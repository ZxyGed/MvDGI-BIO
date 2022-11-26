import os
import math
from itertools import product

# yhbatch -N 10 yeast_level1_semisupervised.sh in main folder, so py file don't need to be add ../->prefix
# yhrun command should add {save_dir} prefix
save_dir = 'bash_file'
dataset = 'yeast'
level = 'level1'
_type = 'semisupervised'
filename = f"{dataset}_{level}_{_type}"
num_piece = 10

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

hiddens_dim_list = [256, 512, 1024, 2048, 3200]
embedding_dim_list = [1024, 512, 256, 128, 64, 32]  # level1 17
dropout_rate = [0, 0.05, 0.1, 0.15, 0.2]
learning_rate = [0.001, 0.005, 0.01, 0.05, 0.1]
params_list = list(product(hiddens_dim_list, embedding_dim_list,
                           dropout_rate, learning_rate))

list_num = math.ceil(len(params_list) / num_piece)
for i in range(num_piece):
    with open(f"{save_dir}/{filename}_{i}.sh", 'w') as f:
        f.write("#!/bin/bash")
        for params in params_list[i * list_num:(i + 1) * list_num]:
            hd, ed, lr, dr = params
            f.write(
                f"\npython train_representation.py -hd {hd} -ed {ed} -lr {lr} -dr {dr}")

with open(f"{save_dir}/{filename}.sh", 'w') as f:
    f.write("#!/bin/bash")
    for i in range(num_piece):
        f.write(f"\nyhrun -N 1 -n 1 {save_dir}/{filename}_{i}.sh")
