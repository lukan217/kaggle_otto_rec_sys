# %%
import heapq
import numba as nb
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import argparse
parser = argparse.ArgumentParser(description='Test for argparse')
parser.add_argument('--is_train', '-v', type=bool, help='is_train',default=False)
args = parser.parse_args()
IS_TRAIN = args.is_train
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
@nb.jit(nopython = True, cache = True)
def inference_(aids, ops, row, result, test_ops_weights, seq_weight):
    for session, idx, length in row:
        unique_aids = nb.typed.Dict.empty(key_type = nb.types.int64, value_type = nb.types.float64)
        cnt = nb.typed.Dict.empty(key_type = nb.types.int64, value_type = nb.types.float64)
        
        candidates = aids[idx:idx + length][::-1]
        candidates_ops = ops[idx:idx + length][::-1]
        for a in candidates:
            unique_aids[a] = 0
                
        # if len(unique_aids) >= 100:
        sequence_weight = np.power(2, np.linspace(seq_weight, 1, len(candidates)))[::-1] - 1
        for a, op, w in zip(candidates, candidates_ops, sequence_weight):
            if a not in cnt: cnt[a] = 0
            cnt[a] += w * test_ops_weights[op]
        result_candidates = heap_topk(cnt, 0, 100)
        # else:
        #     result_candidates = list(unique_aids)
        #     for a in result_candidates:
        #         if a not in topk: continue
        #         for b in topk[a]:
        #             if b in unique_aids: continue
        #             if b not in cnt: cnt[b] = 0
        #             cnt[b] += 1
        #     result_candidates.extend(heap_topk(cnt, 0, 20 - len(result_candidates)))
        result[session] = np.array(result_candidates)
        
@nb.jit(nopython = True)
def inference(aids, ops, row, 
              result_clicks, result_buy,
              test_ops_weights):
    inference_(aids, ops, row, result_clicks, test_ops_weights, 0.1)
    inference_(aids, ops, row, result_buy, test_ops_weights, 0.5)

# %%
test_ops_weights = np.array([1.0, 10.0, 3.0])
# result place holder
result_clicks = nb.typed.Dict.empty(
    key_type = nb.types.int64,
    value_type = nb.types.int64[:])
result_buy = nb.typed.Dict.empty(
    key_type = nb.types.int64,
    value_type = nb.types.int64[:])
for idx in tqdm(range(len(df) - len(df_test), len(df), parallel)):
    row = df.iloc[idx:min(idx + parallel, len(df))][['session', 'idx', 'length']].values
    inference(aids, ops, row, result_clicks, result_buy, test_ops_weights)

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
        test_labels_sub['hits'] = test_labels_sub.apply(lambda df: len(set(df.ground_truth).intersection(set(df.labels[:50]))), axis=1)
        test_labels_sub['gt_count'] = test_labels_sub.ground_truth.str.len().clip(0,20)
        recall = test_labels_sub['hits'].sum() / test_labels_sub['gt_count'].sum()
        score += weights[t]*recall
        print(f'{t} recall =',recall)
    print('=============')
    print('Overall Recall =',score)
    print('=============')

# %%
if IS_TRAIN:
    pred_df.to_parquet('/root/autodl-tmp/ottodata/tmp/pe4train100.parquet')
else:
    pred_df.to_parquet('/root/autodl-tmp/ottodata/tmp/pe4test100.parquet')

# %%



