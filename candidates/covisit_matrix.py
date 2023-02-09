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
args = parser.parse_args()
IS_TRAIN = args.is_train
HAS_TRAIN = True
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
    return [heapq.heappop(q)[2] for _ in range(len(q))][::-1]
   
# save top-k aid2 for each aid1's cnt
@nb.jit(nopython = True, cache = True)
def get_topk(cnts, topk, k):
    for aid1, cnt in cnts.items():
        topk[aid1] = np.array(heap_topk(cnt, 1, k))

# %%
if not HAS_TRAIN:
    topks = {}

    # for two modes
    for mode in [OP_WEIGHT, TIME_WEIGHT]:
        # get nested counter
        cnts = nb.typed.Dict.empty(
            key_type = nb.types.int64,
            value_type = nb.typeof(nb.typed.Dict.empty(key_type = nb.types.int64, value_type = nb.types.float64)))
        max_idx = len(df)
        for idx in tqdm(range(0, max_idx, parallel)):
            row = df.iloc[idx:min(idx + parallel, max_idx)][['session', 'idx', 'length', 'start_time']].values
            get_pairs(aids, ts, ops, row, cnts, ops_weights, mode)

        # get topk from counter
        topk = nb.typed.Dict.empty(
                key_type = nb.types.int64,
                value_type = nb.types.int64[:])
        get_topk(cnts, topk, topn)

        del cnts; gc.collect()
        topks[mode] = topk
    topks_op_weight = {}
    for k,v in topks[OP_WEIGHT].items():
        topks_op_weight[k] = v
    topks_time_weight = {}
    for k,v in topks[TIME_WEIGHT].items():
        topks_time_weight[k] = v
    if IS_TRAIN:
        with open("/root/autodl-tmp/ottodata/tmp/topks_op_weight_2_v2.pkl", "wb") as f:
            pickle.dump(topks_op_weight, f)
        with open("/root/autodl-tmp/ottodata/tmp/topks_time_weight_2_v2.pkl", "wb") as f:
            pickle.dump(topks_time_weight, f)
    else:
        with open("/root/autodl-tmp/ottodata/tmp/topks_op_weight.pkl", "wb") as f:
            pickle.dump(topks_op_weight, f)
        with open("/root/autodl-tmp/ottodata/tmp/topks_time_weight.pkl", "wb") as f:
            pickle.dump(topks_time_weight, f)
else:
    if IS_TRAIN:
        with open("/root/autodl-tmp/ottodata/tmp/topks_op_weight_2.pkl", "rb") as f:
            topks_op_weight = pickle.load(f)
        with open("/root/autodl-tmp/ottodata/tmp/topks_time_weight_2.pkl", "rb") as f:
            topks_time_weight = pickle.load(f)
    else:
        with open("/root/autodl-tmp/ottodata/tmp/topks_op_weight.pkl", "rb") as f:
            topks_op_weight = pickle.load(f)
        with open("/root/autodl-tmp/ottodata/tmp/topks_time_weight.pkl", "rb") as f:
            topks_time_weight = pickle.load(f)
    topks  = {}
    for mode in [OP_WEIGHT, TIME_WEIGHT]:
        topk = nb.typed.Dict.empty(
                    key_type = nb.types.int64,
                    value_type = nb.types.int64[:])
        if mode == OP_WEIGHT:
            for k,v in topks_op_weight.items():
                topk[k] = np.array(v)[:50]
        elif mode == TIME_WEIGHT:
            for k,v in topks_time_weight.items():
                topk[k] = np.array(v)[:50]
        topks[mode] = topk

# %%
topks  = {}
for mode in [OP_WEIGHT, TIME_WEIGHT]:
    topk = nb.typed.Dict.empty(
                key_type = nb.types.int64,
                value_type = nb.types.int64[:])
    if mode == OP_WEIGHT:
        for k,v in topks_op_weight.items():
            topk[k] = np.array(v)[:50]
    elif mode == TIME_WEIGHT:
        for k,v in topks_time_weight.items():
            topk[k] = np.array(v)[:50]
    topks[mode] = topk

# %%
@nb.jit(nopython = True, cache = True)
def inference_(aids, ops, row, result, topk, test_ops_weights, seq_weight):
    for session, idx, length in row:
        unique_aids = nb.typed.Dict.empty(key_type = nb.types.int64, value_type = nb.types.float64)
        cnt = nb.typed.Dict.empty(key_type = nb.types.int64, value_type = nb.types.float64)
        
        candidates = aids[idx:idx + length][::-1]
        candidates_ops = ops[idx:idx + length][::-1]
        for a in candidates:
            unique_aids[a] = 0
                
        # if len(unique_aids) >= 100:
        # sequence_weight = np.power(2, np.linspace(seq_weight, 1, len(candidates)))[::-1] - 1
        # for a, op, w in zip(candidates, candidates_ops, sequence_weight):
        #     if a not in cnt: cnt[a] = 0
        #     cnt[a] += w * test_ops_weights[op]
        # result_candidates = heap_topk(cnt, 0, 50)
        result_candidates = list(unique_aids)
        sequence_weight = np.power(2, np.linspace(0.1, 1, len(result_candidates)))[::-1] - 1

        for a,w in zip(result_candidates,sequence_weight):
            if a not in topk: continue
            for b in topk[a]:
                if b in unique_aids: continue
                if b not in cnt: cnt[b] = 0
                cnt[b] += w
        result_candidates = heap_topk(cnt, 0, 100)
        result[session] = np.array(result_candidates)
        
@nb.jit(nopython = True)
def inference(aids, ops, row, 
              result_clicks, result_buy,
              topk_clicks, topk_buy,
              test_ops_weights):
    inference_(aids, ops, row, result_clicks, topk_clicks, test_ops_weights, 0.1)
    inference_(aids, ops, row, result_buy, topk_buy, test_ops_weights, 0.5)

# %%
# result place holder
result_clicks = nb.typed.Dict.empty(
    key_type = nb.types.int64,
    value_type = nb.types.int64[:])
result_buy = nb.typed.Dict.empty(
    key_type = nb.types.int64,
    value_type = nb.types.int64[:])
for idx in tqdm(range(len(df) - len(df_test), len(df), parallel)):
    row = df.iloc[idx:min(idx + parallel, len(df))][['session', 'idx', 'length']].values
    inference(aids, ops, row, result_clicks, result_buy, topks[TIME_WEIGHT], topks[OP_WEIGHT], test_ops_weights)

# %%
pred_df = []
op_names = ["clicks", "carts", "orders"]
for result, op in zip([result_clicks, result_buy, result_buy], op_names):

    sub = pd.DataFrame({"session": result.keys(), "labels": result.values()})
    # sub.session_type = sub.session_type.astype(str) + f"_{op}"
    # sub.labels = sub.labels.apply(lambda x: " ".join(x.astype(str)))
    sub['type'] = op 
    pred_df.append(sub)
pred_df = pd.concat(pred_df).reset_index(drop = True)
# sub.to_csv('submission.csv', index = False)
pred_df.head()

# %%
if IS_TRAIN:
# COMPUTE METRIC
    test_labels = pd.read_parquet('/root/autodl-tmp/ottodata/valid/test_labels.parquet')
    score = 0
    weights = {'clicks': 0.10, 'carts': 0.30, 'orders': 0.60}
    for t in ['clicks','carts','orders']:
        sub = pred_df.loc[pred_df['type']==t].copy()
        test_labels_sub = test_labels.loc[test_labels['type']==t]
        test_labels_sub = test_labels_sub.merge(sub, how='left', on=['session'])
        test_labels_sub['hits'] = test_labels_sub.apply(lambda df: len(set(df.ground_truth).intersection(set(df.labels[:20]))), axis=1)
        test_labels_sub['gt_count'] = test_labels_sub.ground_truth.str.len().clip(0,20)
        recall = test_labels_sub['hits'].sum() / test_labels_sub['gt_count'].sum()
        score += weights[t]*recall
        print(f'{t} recall =',recall)
    print('=============')
    print('Overall Recall =',score)
    print('=============')

# %%
pred_df['labels'].apply(lambda x: len(x)).mean()

# %%
if IS_TRAIN:
    pred_df.to_parquet('/root/autodl-tmp/ottodata/tmp/co4train_v2.parquet')
else:
    pred_df.to_parquet('/root/autodl-tmp/ottodata/tmp/co4test_v2.parquet')

# %%



