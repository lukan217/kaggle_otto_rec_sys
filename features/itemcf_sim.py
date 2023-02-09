import numpy as np
import cudf
from pandarallel import pandarallel
import pickle
import gc

pandarallel.initialize(nb_workers=15)
import pandas as pd

import argparse

parser = argparse.ArgumentParser(description='Test for argparse')
parser.add_argument('--is_train', '-v', type=bool, help='is_train', default=False)
parser.add_argument('--feat_name', '-f', type=str, help='feat_name, bm25, tfidf, cos, or itemcf', default='bm25')
args = parser.parse_args()

IS_TRAIN = args.is_train

if IS_TRAIN:
    with open(f"/root/autodl-tmp/ottodata/tmp/{args.feat_name}_sim_train.pkl", "rb") as f:
        item_sim = pickle.load(f)
    test = cudf.read_parquet('/root/autodl-tmp/ottodata/valid/test.parquet')
    item_list = test.groupby('session')['aid'].collect().reset_index()
    candidates_names = ['co4train_v2', 'pe4train', 'bm4train']
else:
    with open(f"/root/autodl-tmp/ottodata/tmp/{args.feat_name}_sim_test.pkl", "rb") as f:
        item_sim = pickle.load(f)
    test = cudf.read_parquet('/root/autodl-tmp/ottodata/test.parquet')
    item_list = test.groupby('session')['aid'].collect().reset_index()
    candidates_names = ['co4test_v2', 'pe4test', 'bm4test']


def get_bm25_sim(row):
    sim = [[item_sim.get(j, {}).get(i, 0) for j in row['aid']] for i in row['labels']]
    last_sim = [i[-1] for i in sim]
    mean_sim = [np.mean(i) for i in sim]
    max_sim = [np.max(i) for i in sim]
    sim = (last_sim, mean_sim, max_sim)
    return sim


for candidates_name in candidates_names:
    for target in ['clicks', 'carts', 'orders']:
        print(f'processing {candidates_name} {target}')
        candidates = cudf.read_parquet(f'/root/autodl-tmp/ottodata/tmp/{candidates_name}.parquet')
        candidates = candidates[candidates['type'] == target]
        candidates = candidates.drop(columns=['type'])

        candidates = candidates.merge(item_list, on='session', how='left')
        candidates = candidates.sort_values(by=['session'], ascending=True)
        candidates = candidates.reset_index(drop=True)
        candidates = candidates.to_pandas()

        chunk = 2
        chunk_size = int(np.ceil(len(candidates) / chunk))
        tmps = []
        for i in range(chunk):
            print(f'processing chunk{i}')
            df = candidates.iloc[i * chunk_size:(i + 1) * chunk_size].copy()
            tmp = df.parallel_apply(lambda row: get_bm25_sim(row), axis=1)
            tmps.append(tmp)
            del df, tmp
            gc.collect()
        tmp = pd.concat(tmps)
        candidates[[f'{args.feat_name}_last_sim', f'{args.feat_name}_mean_sim',
                    f'{args.feat_name}_max_sim']] = tmp.values.tolist()
        del tmp, tmps
        gc.collect()
        for col in [f'{args.feat_name}_last_sim', f'{args.feat_name}_mean_sim', f'{args.feat_name}_max_sim']:
            tmp = candidates[['session', col]]
            tmp = cudf.from_pandas(tmp)
            tmp = tmp.explode(col)
            tmp[col] = tmp[col].astype('float32')
            tmp = tmp.fillna(-1)
            tmp = tmp[col].values
            np.save(f'/root/autodl-tmp/ottodata/tmp/{candidates_name}_{col}_{target}.npy', tmp)
            del tmp
            gc.collect()

        del candidates
        gc.collect()
