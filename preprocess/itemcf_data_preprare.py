# %%
import numpy as np
import pandas as pd
import cudf
import sys
from scipy import sparse
import implicit
import collections
from tqdm import tqdm
import gc
import pickle

# %%
import argparse
parser = argparse.ArgumentParser(description='Test for argparse')
parser.add_argument('--is_train', '-v', type=bool, help='is_train',default=False)
args = parser.parse_args()
IS_TRAIN = args.is_train

# %%
if IS_TRAIN:
    train = cudf.read_parquet('/root/autodl-tmp/ottodata/valid/train.parquet')
    train = train[train['ts']>=train['ts'].max() - 7 * 24 * 3600]
    test = cudf.read_parquet('/root/autodl-tmp/ottodata/valid/test.parquet')
else:
    train = cudf.read_parquet('/root/autodl-tmp/ottodata/train.parquet')
    train = train[train['ts']>=train['ts'].max() - 7 * 24 * 3600]
    test = cudf.read_parquet('/root/autodl-tmp/ottodata/test.parquet')

# %%
data = cudf.concat([train,test])
data = data.groupby('session')['aid'].agg(list).reset_index()
data = data.to_pandas()

# %%
from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True,nb_workers=15)

# %%
def mat_func(items):
    tmp = np.zeros(1855603,dtype=bool)
    for item in items:
        tmp[item] += 1
    return sparse.csr_matrix(tmp,dtype=bool)

# %%
user_items_mat = data['aid'].parallel_apply(mat_func)
user_items_mat = sparse.vstack(user_items_mat.values)
if IS_TRAIN:
    sparse.save_npz('/root/autodl-tmp/ottodata/tmp/ui_train_data.npz',user_items_mat)
else:
    sparse.save_npz('/root/autodl-tmp/ottodata/tmp/ui_test_data.npz',user_items_mat)


# %%



