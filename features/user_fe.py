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
session_length = data.groupby('session').size().rename('u_session_length')
session_nunique = data.groupby('session')['aid'].nunique().rename('u_session_nunique')
session_nunique_ratio = (session_nunique / session_length).rename('u_session_nunique_ratio')

# %%
aid_nunique = data.groupby('aid')['session'].nunique().rename('u_aid_nunique')
data = data.merge(aid_nunique.reset_index(), on='aid', how='left')
session_median_unique = data.groupby('session')['u_aid_nunique'].median().rename('u_session_median_unique')

# %%
last_time = data.groupby('session')['ts'].max().rename('u_last_time')
last_time = cudf.to_datetime(last_time, unit='s', utc=False)
session_last_hour = last_time.dt.hour.rename('u_session_last_action_hour')
session_last_weekday = last_time.dt.weekday.rename('u_session_last_action_weekday')

# %%
test_start_time = data['ts'].min()
session_end_time = ((data.groupby('session')['ts'].max() - test_start_time) // 3600).rename(
    'u_session_end_time').sort_index()

# %%
data['time'] = cudf.to_datetime(data['ts'], unit='s', utc=False)
data['hour'] = data['time'].dt.hour
data['day'] = data['time'].dt.day_of_year
data['dayhour'] = data['day'] * 100 + data['hour']

# %%
session_averge_hour = data.groupby('session')['hour'].mean().rename('u_session_averge_hour')
session_std_hour = data.groupby('session')['hour'].std().rename('u_session_std_hour')
session_nunique_hour = data.groupby('session')['dayhour'].nunique().rename('u_session_nunique_hour')
session_hour_length = data.groupby(['session', 'hour']).size().reset_index().groupby('session')[0].mean().rename(
    'u_session_hour_length').sort_index()

# %%
session_hour_average_gap = ((data.groupby(['session', 'dayhour'])['ts'].max() - data.groupby(['session', 'dayhour'])[
    'ts'].min()) / data.groupby(['session', 'dayhour']).size()).reset_index().groupby('session')[0].mean().rename(
    'u_session_hour_average_gap').sort_index()
# 每小时平均点击次数
session_real_count = data.groupby(['session', 'dayhour'])['aid'].count().reset_index().groupby('session')[
    'aid'].mean().rename('u_session_real_count').sort_index()
session_real_nunique = data.groupby(['session', 'dayhour'])['aid'].nunique().reset_index().groupby('session')[
    'aid'].mean().rename('u_session_real_nunique').sort_index()

# %%
session_average_gap = np.log((data.groupby('session')['ts'].max() - data.groupby('session')['ts'].min()) / data.groupby(
    'session').size() + 1).rename('u_log_session_average_gap')

# %%
order_data = data[data['type'] == 2]
click_data = data[data['type'] == 0]
cart_data = data[data['type'] == 1]

# %%
session_click_average_gap = np.log(
    (click_data.groupby('session')['ts'].max() - click_data.groupby('session')['ts'].min()) / click_data.groupby(
        'session').size()).rename('u_log_session_click_average_gap')
session_cart_average_gap = np.log(
    (cart_data.groupby('session')['ts'].max() - cart_data.groupby('session')['ts'].min()) / cart_data.groupby(
        'session').size()).rename('u_log_session_cart_average_gap')
session_order_average_gap = np.log(
    (order_data.groupby('session')['ts'].max() - order_data.groupby('session')['ts'].min()) / order_data.groupby(
        'session').size()).rename('u_log_session_order_average_gap')

# %%
order_count = order_data.groupby('session').size().rename('u_order_count')
click_count = click_data.groupby('session').size().rename('u_click_count')
cart_count = cart_data.groupby('session').size().rename('u_cart_count')

# %%
order_nunique = order_data.groupby('session')['aid'].nunique().rename('u_order_nunique')
click_nunique = click_data.groupby('session')['aid'].nunique().rename('u_click_nunique')
cart_nunique = cart_data.groupby('session')['aid'].nunique().rename('u_cart_nunique')

# %%
user_feature = cudf.concat(
    [session_length, session_nunique, order_count, click_count, cart_count, order_nunique, click_nunique, cart_nunique,
     session_average_gap, session_last_hour, session_last_weekday, session_averge_hour, session_std_hour,
     session_nunique_hour, session_hour_length, session_hour_average_gap, session_real_count, session_real_nunique,
     session_click_average_gap, session_cart_average_gap, session_order_average_gap, session_end_time,
     session_median_unique, session_nunique_ratio
     ], axis=1)

# %%
user_feature['u_order_ratio'] = user_feature['u_order_count'] / user_feature['u_session_length']
user_feature['u_click_ratio'] = user_feature['u_click_count'] / user_feature['u_session_length']
user_feature['u_cart_ratio'] = user_feature['u_cart_count'] / user_feature['u_session_length']

# %%
user_feature.fillna(-1, inplace=True)


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


# %%
user_feature = reduce_mem_usage(user_feature)

# %%
user_feature.columns

# %%
if IS_TRAIN:
    user_feature.to_parquet('/root/autodl-tmp/ottodata/tmp/user_feature_train_v3.parquet')
else:
    user_feature.to_parquet('/root/autodl-tmp/ottodata/tmp/user_feature_test_v3.parquet')

# %%
