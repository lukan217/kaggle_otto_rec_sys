# %%
import cudf
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Test for argparse')
parser.add_argument('--is_train', '-v', type=bool, help='is_train', default=False)
args = parser.parse_args()
IS_TRAIN = args.is_train
# %%
if IS_TRAIN:
    data = cudf.read_parquet('/root/autodl-tmp/ottodata/valid/test.parquet')
else:
    data = cudf.read_parquet('/root/autodl-tmp/ottodata/test.parquet')

# %%
data = data.sort_values(['session', 'ts'], ascending=[True, False])
data['ui_action_reverse'] = data.groupby('session')['aid'].cumcount()
action_reverse = data.groupby(['session', 'aid'])['ui_action_reverse'].min()
data['ui_action_reverse_by_type'] = data.groupby(['session', 'type'])['aid'].cumcount()
action_reverse_by_type = cudf.pivot_table(data, index=['session', 'aid'], columns=['type'],
                                          values=['ui_action_reverse_by_type'], aggfunc='min', fill_value=-1)
action_reverse_by_type.columns = ['ui_action_reverse_click', 'ui_action_reverse_cart', 'ui_action_reverse_order']

# %%

data = data.sort_values(['session', 'ts'], ascending=[True, True])

# %%
data['session_length'] = data.groupby('session')['aid'].transform('count')
data['ui_log_score'] = 2 ** (0.1 + ((1 - 0.1) / (data['session_length'] - 1)) * (
        data['session_length'] - data['ui_action_reverse'] - 1)) - 1
data['ui_log_score'] = data['ui_log_score'].fillna(1.0)
type_weights = {0: 1, 1: 6, 2: 3}
data['ui_type_weight_log_score'] = data['type'].map(type_weights) * data['ui_log_score']

# %%
type_weight_log_score = data.groupby(['session', 'aid'])['ui_type_weight_log_score'].sum()
log_score = data.groupby(['session', 'aid'])['ui_log_score'].sum()
type_weight_log_score = type_weight_log_score.round(4)
log_score = log_score.round(4)

# %%
session_aid_count = data.groupby(['session', 'aid'])['ts'].count().rename('ui_session_aid_count')

# %%
history_count = cudf.pivot_table(data, index=['session', 'aid'], columns=['type'], values=['ts'], aggfunc='count',
                                 fill_value=0)
history_count.columns = ['ui_history_click_count', 'ui_history_cart_count', 'ui_history_order_count']

# %%
data['ts_diff'] = data.groupby('session')['ts'].transform('max') - data['ts']
last_ts_diff = np.log(data.groupby(['session', 'aid'])['ts_diff'].min() + 1)
last_type = data.groupby(['session', 'aid'])['type'].last().astype('int8')
user_item_feature = cudf.merge(last_ts_diff, last_type, how='left', on=['session', 'aid'])
user_item_feature.columns = ['ui_last_ts_diff', 'ui_last_type']
user_item_feature = cudf.merge(user_item_feature, history_count, how='left', on=['session', 'aid'])
user_item_feature = cudf.merge(user_item_feature, action_reverse, how='left', on=['session', 'aid'])
user_item_feature = cudf.merge(user_item_feature, action_reverse_by_type, how='left', on=['session', 'aid'])
user_item_feature = cudf.merge(user_item_feature, type_weight_log_score, how='left', on=['session', 'aid'])
user_item_feature = cudf.merge(user_item_feature, log_score, how='left', on=['session', 'aid'])
user_item_feature = cudf.merge(user_item_feature, session_aid_count, how='left', on=['session', 'aid'])


# %%
def reduce_mem_usage(df):
    start_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')
    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df


user_item_feature = reduce_mem_usage(user_item_feature)

# %%
if IS_TRAIN:
    user_item_feature.to_parquet('/root/autodl-tmp/ottodata/tmp/user_item_feature_train_v3.parquet')
else:
    user_item_feature.to_parquet('/root/autodl-tmp/ottodata/tmp/user_item_feature_test_v3.parquet')

# %%
