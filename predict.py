import pandas as pd
import lightgbm as lgb
from tqdm.auto import tqdm
import argparse
import numpy as np
import gc

parser = argparse.ArgumentParser(description='Test for argparse')
parser.add_argument('--version', '-v', type=int, help='version', default=1)
args = parser.parse_args()

submit = []
probs = []
for target in tqdm(['clicks', 'carts', 'orders']):
    data = pd.read_parquet(f'/root/autodl-tmp/ottodata/tmp/{target}_candidates_test_v7.parquet')
    del_cols = ['session', 'aid', target]
    features = [c for c in data.columns if c not in del_cols]
    model = lgb.Booster(model_file=F'/root/autodl-tmp/ottodata/tmp/ranker_{target}_v{args.version}.model')
    CHUNKS = 20
    chunk_size = int(np.ceil(len(data) / CHUNKS))
    for i in range(CHUNKS):
        preds = model.predict(data[features][i * chunk_size:(i + 1) * chunk_size])
        data.loc[:, target].iloc[i * chunk_size:(i + 1) * chunk_size] = preds
        del preds
        gc.collect()

    data = data[['session', 'aid', target]]
    data.sort_values(['session', target], ascending=[True, False], inplace=True)
    data['n'] = data.groupby('session')['aid'].cumcount()
    data = data.loc[data['n'] < 20]
    sub = data.groupby('session')['aid'].apply(list)
    sub = sub.reset_index()
    sub.rename(columns={'aid': 'labels'}, inplace=True)
    sub['type'] = target
    submit.append(sub)
    del data, sub
    gc.collect()

submit = pd.concat(submit)
submit_with_prob = submit.copy()
submit['labels'] = submit['labels'].apply(lambda x: ' '.join([str(l) for l in x]))
submit['session'] = submit['session'].astype('str') + '_' + submit['type'].astype('str')
submit = submit[['session', 'labels']]
submit.rename(columns={'session': 'session_type'}, inplace=True)
submit.to_csv(f'submission_v{args.version}.csv.gz', index=False)
