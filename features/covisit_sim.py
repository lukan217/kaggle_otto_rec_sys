import numpy as np
import cudf
from pandarallel import pandarallel
import pickle
import gc

pandarallel.initialize(nb_workers=15)
import argparse
import pandas as pd

parser = argparse.ArgumentParser(description='Test for argparse')
parser.add_argument('--is_train', '-v', type=bool, help='is_train', default=False)
parser.add_argument('--feat_name', '-f', type=str, help='feat_name, time or op', default='time')

args = parser.parse_args()

IS_TRAIN = args.is_train

if IS_TRAIN:
    with open(f"/root/autodl-tmp/ottodata/tmp/topks_{args.feat_name}_weight_with_w_train.pkl", "rb") as f:
        topks_op_weight = pickle.load(f)
    test = cudf.read_parquet('/root/autodl-tmp/ottodata/valid/test.parquet')
    item_list = test.groupby('session')['aid'].collect().reset_index()
    candidates_names = ['bm4train']

else:
    with open(f"/root/autodl-tmp/ottodata/tmp/topks_{args.feat_name}_weight_with_w_test.pkl", "rb") as f:
        topks_op_weight = pickle.load(f)
    test = cudf.read_parquet('/root/autodl-tmp/ottodata/test.parquet')
    item_list = test.groupby('session')['aid'].collect().reset_index()
    candidates_names = ['bm4test']


def get_op_weight(row):
    sim = [[topks_op_weight.get(j, {}).get(i, 0) for j in row['aid']] for i in row['labels']]
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
        candidates = candidates.to_pandas()

        chunk = 1
        chunk_size = int(np.ceil(len(candidates) / chunk))
        tmps = []
        for i in range(chunk):
            print(f'processing chunk{i}')
            df = candidates.iloc[i * chunk_size:(i + 1) * chunk_size].copy()
            tmp = df.parallel_apply(lambda row: get_op_weight(row), axis=1)
            tmps.append(tmp)
            del df, tmp
            gc.collect()
        tmp = pd.concat(tmps)
        cols = [f'covisit_{args.feat_name}_weight', f'covisit_{args.feat_name}_weight_mean',
                f'covisit_{args.feat_name}_weight_max']
        candidates[cols] = tmp.values.tolist()
        del tmp, tmps
        gc.collect()

        for col in cols:
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
