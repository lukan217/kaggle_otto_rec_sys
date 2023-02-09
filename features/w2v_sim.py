from gensim.models import Word2Vec
from annoy import AnnoyIndex
import numpy as np
import pandas as pd
import cudf
import argparse
import gc
from pandarallel import pandarallel

pandarallel.initialize(nb_workers=15)

parser = argparse.ArgumentParser(description='Test for argparse')
parser.add_argument('--is_train', '-v', type=bool, help='is_train', default=False)

args = parser.parse_args()

IS_TRAIN = args.is_train

if IS_TRAIN:
    print('train')
    w2vec = Word2Vec.load('/root/autodl-tmp/ottodata/tmp/w2v4train.model')
    aid2idx = {aid: i for i, aid in enumerate(w2vec.wv.index_to_key)}
    index = AnnoyIndex(64, 'dot')
    index.load('/root/autodl-tmp/ottodata/tmp/index4train.ann')
    test = cudf.read_parquet('/root/autodl-tmp/ottodata/valid/test.parquet')
    item_list = test.groupby('session')['aid'].collect().reset_index()
    candidates_names = ['bm4train']
else:
    print('test')
    w2vec = Word2Vec.load('/root/autodl-tmp/ottodata/tmp/w2v4test.model')
    aid2idx = {aid: i for i, aid in enumerate(w2vec.wv.index_to_key)}
    index = AnnoyIndex(64, 'dot')
    index.load('/root/autodl-tmp/ottodata/tmp/index4test.ann')
    test = cudf.read_parquet('/root/autodl-tmp/ottodata/test.parquet')
    item_list = test.groupby('session')['aid'].collect().reset_index()
    candidates_names = ['bm4test']


def get_w2v_sim(row):
    sim = [[index.get_distance(aid2idx[aid1], aid2idx[aid2]) for aid2 in row['aid']] for aid1 in row['labels']]
    last_sim = [s[-1] for s in sim]
    mean_sim = [np.mean(s) for s in sim]
    max_sim = [np.max(s) for s in sim]
    return last_sim, mean_sim, max_sim


for candidates_name in candidates_names:
    for target in ['clicks', 'carts', 'orders']:
        print(f'processing {candidates_name} {target}')
        candidates = cudf.read_parquet(f'/root/autodl-tmp/ottodata/tmp/{candidates_name}.parquet')
        candidates = candidates[candidates['type'] == target]
        candidates = candidates.drop(columns=['type'])

        candidates = candidates.merge(item_list, on='session', how='left')
        candidates = candidates.sort_values(by=['session'], ascending=True)
        candidates = candidates.to_pandas()

        candidates[['w2v_last_sim', 'w2v_mean_sim', 'w2v_max_sim']] = candidates.parallel_apply(
            lambda row: get_w2v_sim(row), axis=1).values.tolist()

        w2v_sim = pd.DataFrame()
        for col in ['w2v_last_sim', 'w2v_mean_sim', 'w2v_max_sim']:
            tmp = candidates[['session', col]]
            tmp = cudf.from_pandas(tmp)
            tmp = tmp.explode(col)
            tmp[col] = tmp[col].astype('float32')
            tmp = tmp.fillna(-1)
            tmp = tmp[col].values
            # w2v_sim[col] = tmp
            np.save(f'/root/autodl-tmp/ottodata/tmp/{candidates_name}_{col}_{target}.npy', tmp)
            del tmp
            gc.collect()

        del candidates
        gc.collect()
