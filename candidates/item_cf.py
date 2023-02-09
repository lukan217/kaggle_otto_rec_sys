# %%
import cudf
import pandas as pd
from scipy import sparse
import implicit
import numpy as np
import argparse
parser = argparse.ArgumentParser(description='Test for argparse')
parser.add_argument('--is_train', '-v', type=bool, help='is_train',default=False)
args = parser.parse_args()
IS_TRAIN = args.is_train

# %%
if IS_TRAIN:
    test = cudf.read_parquet('/root/autodl-tmp/ottodata/valid/test.parquet')
    user_items_train = sparse.load_npz("/root/autodl-tmp/ottodata/tmp/ui_train_data.npz")
else:
    test = cudf.read_parquet('/root/autodl-tmp/ottodata/test.parquet')
    user_items_train = sparse.load_npz("/root/autodl-tmp/ottodata/tmp/ui_test_data.npz")

# %%
test = test.drop_duplicates(subset=['session'], keep='last')
test = test.to_pandas()

# %%
bm25_model = implicit.nearest_neighbours.BM25Recommender(K=50,num_threads=15)
bm25_model.fit(user_items_train)

# %%
from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True,nb_workers=15)

# %%
test['labels'] = test['aid'].parallel_apply(lambda x: bm25_model.similar_items(x, N=51)[0][1:])

# %%
test = test[['session','labels']]

# %%
pred_dfs = []
for t in  ['clicks', 'carts', 'orders']:
    pred_df = test.copy()
    pred_df['type'] = t
    pred_dfs.append(pred_df)
pred_df = pd.concat(pred_dfs)

# %%
if IS_TRAIN:
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
pred_df = pred_df.reset_index(drop=True)

# %%
if IS_TRAIN:
    candidates_2 = cudf.read_parquet('/root/autodl-tmp/ottodata/tmp/co4train_v2.parquet')
    candidates_1 = cudf.read_parquet('/root/autodl-tmp/ottodata/tmp/pe4train.parquet')
else:
    candidates_1 = cudf.read_parquet('/root/autodl-tmp/ottodata/tmp/pe4test.parquet')
    candidates_2 = cudf.read_parquet('/root/autodl-tmp/ottodata/tmp/co4test_v2.parquet')
candidates_1 = candidates_1.to_pandas()
candidates_2 = candidates_2.to_pandas()

# %%
pred_df['labels_1'] = candidates_1['labels']
pred_df['labels_2'] = candidates_2['labels']
del candidates_1, candidates_2

# %%
from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True,nb_workers=15)

# %%
pred_df['labels'] = pred_df.parallel_apply(lambda df: np.setdiff1d(df.labels,df.labels_1), axis=1)
pred_df['labels'] = pred_df.parallel_apply(lambda df: np.setdiff1d(df.labels,df.labels_2), axis=1)

# %%
pred_df = pred_df[['session','labels','type']]

# %%
if IS_TRAIN:
    pred_df.to_parquet('/root/autodl-tmp/ottodata/tmp/bm4train.parquet')
else:
    pred_df.to_parquet('/root/autodl-tmp/ottodata/tmp/bm4test.parquet')

# %%



