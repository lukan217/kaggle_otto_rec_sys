# %%
import numpy as np
import numba as nb
import pickle
import pandas as pd
import cudf
from tqdm import tqdm
from collections import defaultdict
import math
from operator import itemgetter
from joblib import Parallel, delayed
import heapq
import argparse
parser = argparse.ArgumentParser(description='Test for argparse')
parser.add_argument('--is_train', '-v', type=bool, help='is_train',default=False)
args = parser.parse_args()
IS_TRAIN = args.is_train

# %%
if IS_TRAIN:
    train = cudf.read_parquet('/root/autodl-tmp/ottodata/valid/train.parquet')
    last_week_time = train['ts'].max() - 7 * 24 * 3600
    train = train[train['ts']>last_week_time]
    test = cudf.read_parquet('/root/autodl-tmp/ottodata/valid/test.parquet')
else:
    train = cudf.read_parquet('/root/autodl-tmp/ottodata/train.parquet')
    last_week_time = train['ts'].max() - 7 * 24 * 3600
    train = train[train['ts']>last_week_time]
    test = cudf.read_parquet('/root/autodl-tmp/ottodata/test.parquet')

# %%
data = cudf.concat([train,test])

# %%
def itemCFTrain(df):
    # user_item_dict = df.groupby("session")[['aid','ts']].agg(["collect"])['collect'].to_pandas().to_dict()
    user_item_dict = df.groupby("session")[['aid']].collect().to_pandas()
    # user_item_dict.columns = ['aid','ts']
    # user_item_dict['collect'] = user_item_dict.apply(lambda x: list(zip(x['aid'],x['ts'])),axis=1)
    return user_item_dict['aid'].to_dict()

# %%
uidict = itemCFTrain(data)


# %%
@nb.jit(nopython = True, cache = True)
def ItemMatrix_fn(itemMatrix,N,items):
    for loc_1,i in enumerate(items):
        if i not in itemMatrix: itemMatrix[i] = {0: 0.0 for _ in range(0)}
        if i not in N: N[i] = 0
        N[i] += 1
        for loc_2,j in enumerate(items):
            if i==j:
                continue
            loc_alpha = 1.0 if loc_2 > loc_1 else 0.6
            loc_weight = loc_alpha * (0.5 **(np.abs(loc_2 - loc_1) - 1))
            # time_weight = np.exp(-15000 * np.abs(i_time - j_time))
            cnt = itemMatrix[i]
            if j not in cnt: cnt[j] = 0.0
            cnt[j] += loc_weight
            # itemMatrix[i].setdefault(j, 0)
            # itemMatrix[i][j] += loc_weight

# %%
itemMatrix = nb.typed.Dict.empty(
            key_type = nb.types.int64,
            value_type = nb.typeof(nb.typed.Dict.empty(key_type = nb.types.int64, value_type = nb.types.float64)))
N = nb.typed.Dict.empty(
            key_type = nb.types.int64,
            value_type = nb.types.int64)

# %%
for user, items in tqdm(uidict.items()):
    ItemMatrix_fn(itemMatrix,N,items)

# %%
@nb.jit(nopython = True, cache = True)
def heap_topk(cnt,  cap):
    q = [(0.0, 0, 0) for _ in range(0)]
    for i, (k, n) in enumerate(cnt.items()):
        heapq.heappush(q, (n, i, k))
        if len(q) > cap:
            heapq.heappop(q)
    res = [heapq.heappop(q) for _ in range(len(q))][::-1]
    res = [(r[2], r[0]) for r in res]
    return res

# %%
@nb.jit(nopython = True, cache = True)
def ItemSimilarityMatrix_fn(itemSimMatrix, i,N, related_items):
    if i not in itemSimMatrix: itemSimMatrix[i] = {0: 0.0 for _ in range(0)}
    res = heap_topk(related_items,500)
    for j, cij in res:
        cnt = itemSimMatrix[i]
        if j not in cnt: cnt[j] = 0.0
        cnt[j] =  cij / math.sqrt(N[i] * N[j])
        # itemMatrix[i][j] = cij / math.sqrt(N[i] * N[j])

# %%
itemSimMatrix = nb.typed.Dict.empty(
            key_type = nb.types.int64,
            value_type = nb.typeof(nb.typed.Dict.empty(key_type = nb.types.int64, value_type = nb.types.float64)))
for i, related_items in tqdm(itemMatrix.items()):
    ItemSimilarityMatrix_fn(itemSimMatrix, i,N, related_items)

# %%
item_sim = {}
for k,v in tqdm(itemSimMatrix.items()):
    item_sim[k] = dict(v)

# %%
del itemSimMatrix,itemMatrix

# %%
if IS_TRAIN:
    with open("/root/autodl-tmp/ottodata/tmp/item_sim_train.pkl", "wb") as f:
        pickle.dump(item_sim, f)
else:

    with open("/root/autodl-tmp/ottodata/tmp/item_sim_test.pkl", "wb") as f:
        pickle.dump(item_sim, f)

# %%



