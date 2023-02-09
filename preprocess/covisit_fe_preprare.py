# %%
import os
import gc
import heapq
import pickle
import numba as nb
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import argparse
parser = argparse.ArgumentParser(description='Test for argparse')
parser.add_argument('--is_train', '-v', type=bool, help='is_train',default=False)
parser.add_argument('--feat_name', '-f', type=str, help='feat_name, bm25, tdidf, cos',default='bm25')

args = parser.parse_args()
IS_TRAIN = args.is_train
HAS_TRAIN = False
tail = 30
parallel = 1024
topn = 100
ops_weights = np.array([1.0, 6.0, 3.0])
OP_WEIGHT = 0; TIME_WEIGHT = 1
parallel = 1024
test_ops_weights = np.array([1.0, 6.0, 3.0])

# %%
if IS_TRAIN:
    df = pd.read_csv("/root/autodl-tmp/ottodata/tmpdata/train_2.csv")
    df_test = pd.read_csv("/root/autodl-tmp/ottodata/tmpdata/test_2.csv")
    df = pd.concat([df, df_test]).reset_index(drop = True)
    npz = np.load("/root/autodl-tmp/ottodata/tmpdata/train_2.npz")
    npz_test = np.load("/root/autodl-tmp/ottodata/tmpdata/test_2.npz")
else:
    df = pd.read_csv("/root/autodl-tmp/ottodata/tmpdata/train.csv")
    df_test = pd.read_csv("/root/autodl-tmp/ottodata/tmpdata/test.csv")
    df = pd.concat([df, df_test]).reset_index(drop = True)
    npz = np.load("/root/autodl-tmp/ottodata/tmpdata/train.npz")
    npz_test = np.load("/root/autodl-tmp/ottodata/tmpdata/test.npz")

# %%
aids = np.concatenate([npz['aids'], npz_test['aids']])
ts = np.concatenate([npz['ts'], npz_test['ts']])
ops = np.concatenate([npz['type'], npz_test['type']])

df["idx"] = np.cumsum(df.length) - df.length
df["end_time"] = df.start_time + ts[df.idx + df.length - 1]

# %%
# get pair dict {(aid1, aid2): weight} for each session
# The maximum time span between two points is 1 day = 24 * 60 * 60 sec
@nb.jit(nopython = True, cache = True)
def get_single_pairs(pairs, aids, ts, ops, idx, length, start_time, ops_weights, mode):
    max_idx = idx + length
    min_idx = max(max_idx - tail, idx)
    for i in range(min_idx, max_idx):
        for j in range(i + 1, max_idx):
            if ts[j] - ts[i] >= 8 * 60 * 60: break
            if aids[i] == aids[j]: continue
            if mode == OP_WEIGHT:
                w1 = ops_weights[ops[j]]
                w2 = ops_weights[ops[i]] *0.3
            elif mode == TIME_WEIGHT:
                w1 = 1 + 3 * (ts[i] + start_time - 1659304800) / (1662328791 - 1659304800)
                w2 = (1 + 3 * (ts[j] + start_time - 1659304800) / (1662328791 - 1659304800))*0.3
            pairs[(aids[i], aids[j])] = w1
            pairs[(aids[j], aids[i])] = w2

# get pair dict of each session in parallel
# merge pairs into a nested dict format (cnt)
@nb.jit(nopython = True, parallel = True, cache = True)
def get_pairs(aids, ts, ops, row, cnts, ops_weights, mode):
    par_n = len(row)
    pairs = [{(0, 0): 0.0 for _ in range(0)} for _ in range(par_n)]
    for par_i in nb.prange(par_n):
        _, idx, length, start_time = row[par_i]
        get_single_pairs(pairs[par_i], aids, ts, ops, idx, length, start_time, ops_weights, mode)
    for par_i in range(par_n):
        for (aid1, aid2), w in pairs[par_i].items():
            if aid1 not in cnts: cnts[aid1] = {0: 0.0 for _ in range(0)}
            cnt = cnts[aid1]
            if aid2 not in cnt: cnt[aid2] = 0.0
            cnt[aid2] += w
    
# util function to get most common keys from a counter dict using min-heap
# overwrite == 1 means the later item with equal weight is more important
# otherwise, means the former item with equal weight is more important
# the result is ordered from higher weight to lower weight
@nb.jit(nopython = True, cache = True)
def heap_topk(cnt, overwrite, cap):
    q = [(0.0, 0, 0) for _ in range(0)]
    for i, (k, n) in enumerate(cnt.items()):
        if overwrite == 1:
            heapq.heappush(q, (n, i, k))
        else:
            heapq.heappush(q, (n, -i, k))
        if len(q) > cap:
            heapq.heappop(q)
    res = [heapq.heappop(q) for _ in range(len(q))][::-1]
    res = [(r[2], r[0]) for r in res]
    return res
   
# save top-k aid2 for each aid1's cnt
@nb.jit(nopython = True, cache = True)
def get_topk(cnts, topk, k):
    for aid1, cnt in cnts.items():
        for i in heap_topk(cnt, 1, k):
            # topk[aid1][i[0]] = i[1]
            if aid1 not in topk: topk[aid1] = {0: 0.0 for _ in range(0)}
            c = topk[aid1]
            if i[0] not in c: c[i[0]] = 0.0
            c[i[0]] += i[1]

# %%
if not HAS_TRAIN:
    topks = {}
    # for two modes
    for mode in [OP_WEIGHT,TIME_WEIGHT]:
        # get nested counter
        cnts = nb.typed.Dict.empty(
            key_type = nb.types.int64,
            value_type = nb.typeof(nb.typed.Dict.empty(key_type = nb.types.int64, value_type = nb.types.float64)))
        max_idx = len(df)
        for idx in tqdm(range(0, max_idx, parallel)):
            row = df.iloc[idx:min(idx + parallel, max_idx)][['session', 'idx', 'length', 'start_time']].values
            get_pairs(aids, ts, ops, row, cnts, ops_weights, mode)
        topk = nb.typed.Dict.empty(
                    key_type = nb.types.int64,
                    value_type = nb.typeof(nb.typed.Dict.empty(key_type = nb.types.int64, value_type = nb.types.float64)))
        get_topk(cnts, topk, topn)
        del cnts; gc.collect()
        topks[mode] = topk

# %%
topks_op_weight = {}
for k,v in topks[OP_WEIGHT].items():
    topks_op_weight[k] = dict(v)
topks_time_weight = {}
for k,v in topks[TIME_WEIGHT].items():
    topks_time_weight[k] = dict(v)

# %%
if IS_TRAIN:
    with open("/root/autodl-tmp/ottodata/tmp/topks_op_weight_with_w_train.pkl", "wb") as f:
        pickle.dump(topks_op_weight, f)
    with open("/root/autodl-tmp/ottodata/tmp/topks_time_weight_with_w_train.pkl", "wb") as f:
        pickle.dump(topks_time_weight, f)
else:
    with open("/root/autodl-tmp/ottodata/tmp/topks_op_weight_with_w_test.pkl", "wb") as f:
        pickle.dump(topks_op_weight, f)
    with open("/root/autodl-tmp/ottodata/tmp/topks_time_weight_with_w_test.pkl", "wb") as f:
        pickle.dump(topks_time_weight, f)

# %%



