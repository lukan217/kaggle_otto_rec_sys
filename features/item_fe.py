# %%
import cudf
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Test for argparse')
parser.add_argument('--is_train', '-v', type=bool, help='is_train', default=False)
args = parser.parse_args()
IS_TRAIN = args.is_train

# IS_TRAIN = False

# %%
if IS_TRAIN:
    train = cudf.read_parquet('/root/autodl-tmp/ottodata/valid/train.parquet')
    test = cudf.read_parquet('/root/autodl-tmp/ottodata/valid/test.parquet')
else:
    train = cudf.read_parquet('/root/autodl-tmp/ottodata/train.parquet')
    train = train[train['ts'] >= train['ts'].min() + 7 * 24 * 3600]
    test = cudf.read_parquet('/root/autodl-tmp/ottodata/test.parquet')

# %%
last_week_time = train['ts'].max() - 7 * 24 * 3600
test_start_time = test['ts'].min()
data = cudf.concat([train, test])

# %%
data['hour'] = cudf.to_datetime(data['ts'], unit='s', utc=False).dt.hour
item_averge_hour = data.groupby('aid')['hour'].mean().rename('i_item_averge_hour')
item_click_averge_hour = data[data['type'] == 0].groupby('aid')['hour'].mean().rename('i_item_click_averge_hour')
item_cart_averge_hour = data[data['type'] == 1].groupby('aid')['hour'].mean().rename('i_item_cart_averge_hour')
item_order_averge_hour = data[data['type'] == 2].groupby('aid')['hour'].mean().rename('i_item_order_averge_hour')

# %%
item_start_time_day = ((test_start_time - data.groupby('aid')['ts'].min()) // (24 * 3600)).rename('i_item_start_time')
# item_end_time_day = ((test_start_time - data.groupby('aid')['ts'].max())//(24*3600)).rename('item_end_time')
item_start_time_hour = ((test_start_time - data.groupby('aid')['ts'].min()) // (3600)).rename('i_item_start_time_hour')
# item_end_time_hour = ((test_start_time - data.groupby('aid')['ts'].max())//(3600)).rename('item_end_time_hour')

# %%
# last_week_data = data[data['ts'] > last_week_time]
# item_count_last_week = last_week_data.groupby('aid').size().rename('item_count_last_week')
# item_nunique_last_week = last_week_data.groupby('aid')['session'].nunique().rename('item_nunique_last_week')
# type_count_last_week = cudf.pivot_table(last_week_data, index=['aid'],values='session', columns=['type'], aggfunc='count', fill_value=0)
# type_count_last_week.columns = ['click_count_last_week', 'cart_count_last_week', 'order_count_last_week']
# type_nunique_last_week = cudf.pivot_table(last_week_data, index=['aid'],values='session', columns=['type'], aggfunc='nunique', fill_value=0)
# type_nunique_last_week.columns = ['click_nunique_last_week', 'cart_nunique_last_week', 'order_nunique_last_week']

# %%
# last_day_data = data[data['ts'] > last_day_time]
# item_count_last_day = last_day_data.groupby('aid').size().rename('item_count_last_day')
# item_nunique_last_day = last_day_data.groupby('aid')['session'].nunique().rename('item_nunique_last_day')
# type_count_last_day = cudf.pivot_table(last_day_data, index=['aid'],values='session', columns=['type'], aggfunc='count', fill_value=0)
# type_count_last_day.columns = ['click_count_last_day', 'cart_count_last_day', 'order_count_last_day']
# type_nunique_last_day = cudf.pivot_table(last_day_data, index=['aid'],values='session', columns=['type'], aggfunc='nunique', fill_value=0)
# type_nunique_last_day.columns = ['click_nunique_last_day', 'cart_nunique_last_day', 'order_nunique_last_day']

# %%
test_week_data = data[data['ts'] > test_start_time]
item_count_test_week = test_week_data.groupby('aid').size().rename('i_item_count_test_week')
item_nunique_test_week = test_week_data.groupby('aid')['session'].nunique().rename('i_item_nunique_test_week')
type_count_test_week = cudf.pivot_table(test_week_data, index=['aid'], values='session', columns=['type'],
                                        aggfunc='count', fill_value=0)
type_count_test_week.columns = ['i_item_click_count_test_week', 'i_item_cart_count_test_week',
                                'i_item_order_count_test_week']
type_nunique_test_week = cudf.pivot_table(test_week_data, index=['aid'], values='session', columns=['type'],
                                          aggfunc='nunique', fill_value=0)
type_nunique_test_week.columns = ['i_item_click_nunique_test_week', 'i_item_cart_nunique_test_week',
                                  'i_item_order_nunique_test_week']

# %%
item_count = data.groupby('aid').size().rename('i_item_count')
item_nunique = data.groupby('aid')['session'].nunique().rename('i_item_nunique')
type_count = cudf.pivot_table(data, index=['aid'], values='session', columns=['type'], aggfunc='count', fill_value=0)
type_count.columns = ['i_item_click_count', 'i_item_cart_count', 'i_item_order_count']
type_nunique = cudf.pivot_table(data, index=['aid'], values='session', columns=['type'], aggfunc='nunique',
                                fill_value=0)
type_nunique.columns = ['i_item_click_nunique', 'i_item_cart_nunique', 'i_item_order_nunique']

# %%
item_feature = cudf.concat(
    [item_count, item_nunique, type_count, type_nunique, item_count_test_week, item_nunique_test_week,
     type_count_test_week, type_nunique_test_week, item_start_time_day, item_start_time_hour, item_averge_hour,
     item_click_averge_hour, item_cart_averge_hour, item_order_averge_hour], axis=1)

# %%
item_feature['i_item_order_ratio'] = item_feature['i_item_order_count'] / item_feature['i_item_count']
item_feature['i_item_cart_ratio'] = item_feature['i_item_cart_count'] / item_feature['i_item_count']
item_feature['i_item_click_ratio'] = item_feature['i_item_click_count'] / item_feature['i_item_count']

# %%
# item_feature['item_order_ratio_last_week'] = item_feature['order_count_last_week'] / item_feature['item_count_last_week']
# item_feature['item_cart_ratio_last_week'] = item_feature['cart_count_last_week'] / item_feature['item_count_last_week']
# item_feature['item_click_ratio_last_week'] = item_feature['click_count_last_week'] / item_feature['item_count_last_week']

# %%
item_feature['i_item_order_ratio_test_week'] = item_feature['i_item_order_count_test_week'] / item_feature[
    'i_item_count_test_week']
item_feature['i_item_cart_ratio_test_week'] = item_feature['i_item_cart_count_test_week'] / item_feature[
    'i_item_count_test_week']
item_feature['i_item_click_ratio_test_week'] = item_feature['i_item_click_count_test_week'] / item_feature[
    'i_item_count_test_week']

# %%
item_feature['i_item_order_ratio_change'] = item_feature['i_item_order_ratio_test_week'] - item_feature[
    'i_item_order_ratio']
item_feature['i_item_cart_ratio_change'] = item_feature['i_item_cart_ratio_test_week'] - item_feature[
    'i_item_cart_ratio']
item_feature['i_item_click_ratio_change'] = item_feature['i_item_click_ratio_test_week'] - item_feature[
    'i_item_click_ratio']

# %%
item_feature = item_feature.fillna(-1)


# %%
# item_label = cudf.read_parquet('/root/autodl-tmp/ottodata/tmp/item_label.parquet')
# data = data.merge(item_label,on='aid',how='left')
# item_feature = item_feature.merge(item_label,on='aid',how='left')
# item_feature.sort_values(['aid'])

# item_label_count = data.groupby('item_label').size().rename('i_item_label_count')
# item_label_nunique = data.groupby('item_label')['session'].nunique().rename('i_item_label_nunique')
# item_type_count = cudf.pivot_table(data, index=['item_label'],values='session', columns=['type'], aggfunc='count', fill_value=0)
# item_type_count.columns = ['i_item_label_click_count', 'i_item_label_cart_count', 'i_item_label_order_count']
# item_label_feature = cudf.concat([item_label_count,item_label_nunique,item_type_count], axis=1)
# item_label_feature['i_item_label_order_ratio'] = item_label_feature['i_item_label_order_count'] / item_label_feature['i_item_label_count']
# item_label_feature['i_item_label_cart_ratio'] = item_label_feature['i_item_label_cart_count'] / item_label_feature['i_item_label_count']
# item_label_feature['i_item_label_click_ratio'] = item_label_feature['i_item_label_click_count'] / item_label_feature['i_item_label_count']
# item_label_feature = item_label_feature.drop(columns=['i_item_label_click_count', 'i_item_label_cart_count', 'i_item_label_order_count'])
# item_label_feature = item_label_feature.fillna(-1)
# item_feature = item_feature.merge(item_label_feature,on='item_label',how='left')
# item_feature = item_feature.set_index('aid')
# item_feature = item_feature.sort_index()


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


item_feature = reduce_mem_usage(item_feature)

# %%
if IS_TRAIN:
    item_feature.to_parquet('/root/autodl-tmp/ottodata/tmp/item_feature_train_v3.parquet')
else:
    item_feature.to_parquet('/root/autodl-tmp/ottodata/tmp/item_feature_test_v3.parquet')

# %%
