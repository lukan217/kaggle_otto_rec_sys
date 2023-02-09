import pandas as pd
import lightgbm as lgb
import argparse
import gc
import time
import logging
import random

random.seed(2023)

parser = argparse.ArgumentParser(description='Test for argparse')
parser.add_argument('--version', '-v', type=int, help='version', default=1)
parser.add_argument('--lr', '-l', type=float, help='objective', default=0.1)
parser.add_argument('--num_boost_round', '-n', type=int, help='num_boost_round', default=1000)
parser.add_argument('--sample_rate', '-s', type=float, help='sample_rate', default=0.2)

args = parser.parse_args()
logging.basicConfig(filename=f"{time.strftime('%m-%d %H:%M', time.localtime())}.log",
                    filemode="w",
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

session = [i for i in range(11098528, 12899779)]
valid_session = random.sample(session, int(len(session) * 0.2))
test_labels = pd.read_parquet('/root/autodl-tmp/ottodata/valid/test_labels.parquet')
test_labels = test_labels[test_labels['session'].isin(valid_session)]
score = 0
weights = {'clicks': 0.10, 'carts': 0.30, 'orders': 0.60}
recalls = []


def feval_recall(preds):
    validate = valid[['session', 'aid', target]].copy()
    validate[target] = preds
    validate.sort_values(['session', target], ascending=[True, False], inplace=True)
    validate['n'] = validate.groupby('session')['aid'].cumcount()
    sub = validate.loc[validate['n'] < 20]
    sub = sub.groupby('session')['aid'].apply(list).reset_index()
    sub.rename(columns={'aid': 'labels'}, inplace=True)
    sub['type'] = target
    sub = test_labels_sub.merge(sub, how='left', on=['session'])
    sub['hits'] = sub.apply(lambda df: len(set(df.ground_truth).intersection(set(df.labels))), axis=1)
    sub['gt_count'] = sub.ground_truth.str.len().clip(0, 20)
    recall = sub['hits'].sum() / sub['gt_count'].sum()
    return recall


for target in ['clicks', 'carts', 'orders']:
    logging.info(f"training for {target}")
    test_labels_sub = test_labels.loc[test_labels['type'] == target]
    data = pd.read_parquet(f'/root/autodl-tmp/ottodata/tmp/{target}_candidates_train_v7.parquet')
    validation = data[data['session'].isin(valid_session)]
    data = data[~data['session'].isin(valid_session)]
    data['positive_rate'] = data.groupby('session')[target].transform('mean')
    data = data[data['positive_rate'] > 0]
    data = data.drop('positive_rate', axis=1)
    positives = data.loc[data[target] == 1]
    negatives = data.loc[data[target] == 0].sample(frac=args.sample_rate, random_state=2023)
    data = pd.concat([positives, negatives], axis=0, ignore_index=True)
    del positives, negatives
    gc.collect()
    data = data.sample(frac=1, random_state=2023)
    data = data.sort_values(by=['session'], ascending=True)
    del_cols = ['session', 'aid', target]
    features = [c for c in data.columns if c not in del_cols]
    params = {
        'objective': 'lambdarank',
        'learning_rate': args.lr,
        # 'num_iterations':args.num_boost_round,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'metric': "ndcg",
        'ndcg_eval_at': [20],
        'random_state': 2023,
        'n_jobs': -1,
    }

    train = data
    valid = validation
    del data, validation
    gc.collect()
    train_group = train.groupby('session').size().values
    valid_group = valid.groupby('session').size().values
    train_data = lgb.Dataset(train[features], label=train[target], group=train_group)
    valid_data = lgb.Dataset(valid[features], label=valid[target], group=valid_group)
    del train
    # valid = valid[['session','aid', target]]
    gc.collect()
    model = lgb.train(params, train_data, valid_sets=[valid_data],
                      callbacks=[lgb.early_stopping(20), lgb.log_evaluation(10)], num_boost_round=args.num_boost_round)
    recall = feval_recall(model.predict(valid[features]))
    score += weights[target] * recall
    recalls.append(recall)
    logging.info(f'{target} recall = {recall}')
    model.save_model(f'/root/autodl-tmp/ottodata/tmp/ranker_{target}_v{args.version}.model')
    del model, train_data, valid_data, valid, train_group, valid_group
    gc.collect()

logging.info(f'overall recall = {score}')
