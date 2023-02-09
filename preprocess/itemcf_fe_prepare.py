# %%
import numpy as np
import pickle
import pandas as pd
import implicit
from scipy import sparse
import pickle
import cudf
from tqdm import tqdm
import argparse
parser = argparse.ArgumentParser(description='Test for argparse')
parser.add_argument('--is_train', '-v', type=bool, help='is_train',default=False)
parser.add_argument('--feat_name', '-f', type=str, help='feat_name, bm25, tdidf, cos',default='bm25')

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
if args.feat_name == 'bm25':
    bm25_model = implicit.nearest_neighbours.BM25Recommender(K=50,num_threads=15)
    bm25_model.fit(user_items_train)
elif args.feat_name == 'tfidf':
    bm25_model = implicit.nearest_neighbours.BM25Recommender(num_threads=15)
    bm25_model.fit(user_items_train)
else:
    bm25_model = implicit.nearest_neighbours.CosineRecommender(num_threads=15)
    bm25_model.fit(user_items_train)

# %%
unique_aid = list(test['aid'].unique().to_pandas().values)

# %%
bm25_sim = {}
for aid in tqdm(unique_aid):
    key, value = bm25_model.similar_items(aid)
    sim = dict(zip(key, value))
    bm25_sim[aid] = sim

# %%
if IS_TRAIN:
    with open(f"/root/autodl-tmp/ottodata/tmp/{args.feat_name}_sim_train.pkl", "wb") as f:
        pickle.dump(bm25_sim, f)
else:

    with open(f"/root/autodl-tmp/ottodata/tmp/{args.feat_name}_sim_test.pkl", "wb") as f:
        pickle.dump(bm25_sim, f)

# %%



