# %%
import cudf
import numpy as np
import polars as pl

import argparse
parser = argparse.ArgumentParser(description='Test for argparse')
parser.add_argument('--is_train', '-v', type=bool, help='is_train',default=False)
args = parser.parse_args()
IS_TRAIN = args.is_train

# %%
if IS_TRAIN:
    train = cudf.read_parquet('/root/autodl-tmp/ottodata/valid/train.parquet')
    test = cudf.read_parquet('/root/autodl-tmp/ottodata/valid/test.parquet')
else:
    train = cudf.read_parquet('/root/autodl-tmp/ottodata/train.parquet')
    train = train[train['ts']>=train['ts'].min() + 7 * 24 * 3600]
    test = cudf.read_parquet('/root/autodl-tmp/ottodata/test.parquet')

# %%
train_df = train.groupby('session').agg({'aid':'count','ts':'min'}).reset_index()
test_df = test.groupby('session').agg({'aid':'count','ts':'min'}).reset_index()
train_df = train_df.sort_values(['session','ts'])
test_df = test_df.sort_values(['session','ts'])

# %%
train_df = train_df.rename(columns={'aid':'length','ts':'start_time'})
test_df = test_df.rename(columns={'aid':'length','ts':'start_time'})

# %%
train = train.groupby("session").agg(["collect"])
test = test.groupby("session").agg(["collect"])

# %%
aids = train['aid']['collect'].to_pandas()
aids = np.concatenate(aids.values)
ts = train['ts']['collect'].to_pandas()
ts = np.concatenate(ts.values)
type = train['type']['collect'].to_pandas()
type = np.concatenate(type.values)

# %%
test_aids = test['aid']['collect'].to_pandas()
test_aids = np.concatenate(test_aids.values)
test_ts = test['ts']['collect'].to_pandas()
test_ts = np.concatenate(test_ts.values)
test_type = test['type']['collect'].to_pandas()
test_type = np.concatenate(test_type.values)

# %%
if IS_TRAIN:
    np.savez('/root/autodl-tmp/ottodata/tmpdata/test_2_v2.npz', aids=test_aids, ts=test_ts, type=test_type)
    np.savez('/root/autodl-tmp/ottodata/tmpdata/train_2_v2.npz', aids=aids, ts=ts, type=type)
    train_df.to_csv('/root/autodl-tmp/ottodata/tmpdata/train_2_v2.csv')
    test_df.to_csv('/root/autodl-tmp/ottodata/tmpdata/test_2_v2.csv')
else:
    np.savez('/root/autodl-tmp/ottodata/tmpdata/test.npz', aids=test_aids, ts=test_ts, type=test_type)
    np.savez('/root/autodl-tmp/ottodata/tmpdata/train.npz', aids=aids, ts=ts, type=type)
    train_df.to_csv('/root/autodl-tmp/ottodata/tmpdata/train.csv')
    test_df.to_csv('/root/autodl-tmp/ottodata/tmpdata/test.csv')

# %%



