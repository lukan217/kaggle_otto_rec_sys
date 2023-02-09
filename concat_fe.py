import cudf
import gc
import torch
import numpy as np
import polars as pl

for IS_TRAIN in [True, False]:
    print("IS_TRAIN", IS_TRAIN)
    for target in ['clicks', 'carts', 'orders']:
        if IS_TRAIN and target == 'clicks':
            continue
        print("concat feature for ", target)
        if IS_TRAIN:
            candidates_1 = cudf.read_parquet('/root/autodl-tmp/ottodata/tmp/co4train_v2.parquet')
            candidates_2 = cudf.read_parquet('/root/autodl-tmp/ottodata/tmp/pe4train.parquet')
            candidates_3 = cudf.read_parquet('/root/autodl-tmp/ottodata/tmp/bm4train.parquet')
            w2v_last_sim_1 = np.load(f'/root/autodl-tmp/ottodata/tmp/co4train_v2_w2v_last_sim_{target}.npy')
            w2v_last_sim_2 = np.load(f'/root/autodl-tmp/ottodata/tmp/pe4train_w2v_last_sim_{target}.npy')
            w2v_last_sim_3 = np.load(f'/root/autodl-tmp/ottodata/tmp/bm4train_w2v_last_sim_{target}.npy')

            w2v_mean_sim_1 = np.load(f'/root/autodl-tmp/ottodata/tmp/co4train_v2_w2v_mean_sim_{target}.npy')
            w2v_mean_sim_2 = np.load(f'/root/autodl-tmp/ottodata/tmp/pe4train_w2v_mean_sim_{target}.npy')
            w2v_mean_sim_3 = np.load(f'/root/autodl-tmp/ottodata/tmp/bm4train_w2v_mean_sim_{target}.npy')

            w2v_max_sim_1 = np.load(f'/root/autodl-tmp/ottodata/tmp/co4train_v2_w2v_max_sim_{target}.npy')
            w2v_max_sim_2 = np.load(f'/root/autodl-tmp/ottodata/tmp/pe4train_w2v_max_sim_{target}.npy')
            w2v_max_sim_3 = np.load(f'/root/autodl-tmp/ottodata/tmp/bm4train_w2v_max_sim_{target}.npy')

            bm25_last_sim_1 = np.load(f'/root/autodl-tmp/ottodata/tmp/co4train_v2_bm25_last_sim_{target}.npy')
            bm25_last_sim_2 = np.load(f'/root/autodl-tmp/ottodata/tmp/pe4train_bm25_last_sim_{target}.npy')
            bm25_last_sim_3 = np.load(f'/root/autodl-tmp/ottodata/tmp/bm4train_bm25_last_sim_{target}.npy')

            bm25_mean_sim_1 = np.load(f'/root/autodl-tmp/ottodata/tmp/co4train_v2_bm25_mean_sim_{target}.npy')
            bm25_mean_sim_2 = np.load(f'/root/autodl-tmp/ottodata/tmp/pe4train_bm25_mean_sim_{target}.npy')
            bm25_mean_sim_3 = np.load(f'/root/autodl-tmp/ottodata/tmp/bm4train_bm25_mean_sim_{target}.npy')

            bm25_max_sim_1 = np.load(f'/root/autodl-tmp/ottodata/tmp/co4train_v2_bm25_max_sim_{target}.npy')
            bm25_max_sim_2 = np.load(f'/root/autodl-tmp/ottodata/tmp/pe4train_bm25_max_sim_{target}.npy')
            bm25_max_sim_3 = np.load(f'/root/autodl-tmp/ottodata/tmp/bm4train_bm25_max_sim_{target}.npy')

            cos_last_sim_1 = np.load(f'/root/autodl-tmp/ottodata/tmp/co4train_v2_cos_last_sim_{target}.npy')
            cos_last_sim_2 = np.load(f'/root/autodl-tmp/ottodata/tmp/pe4train_cos_last_sim_{target}.npy')
            cos_last_sim_3 = np.load(f'/root/autodl-tmp/ottodata/tmp/bm4train_cos_last_sim_{target}.npy')

            cos_mean_sim_1 = np.load(f'/root/autodl-tmp/ottodata/tmp/co4train_v2_cos_mean_sim_{target}.npy')
            cos_mean_sim_2 = np.load(f'/root/autodl-tmp/ottodata/tmp/pe4train_cos_mean_sim_{target}.npy')
            cos_mean_sim_3 = np.load(f'/root/autodl-tmp/ottodata/tmp/bm4train_cos_mean_sim_{target}.npy')

            cos_max_sim_1 = np.load(f'/root/autodl-tmp/ottodata/tmp/co4train_v2_cos_max_sim_{target}.npy')
            cos_max_sim_2 = np.load(f'/root/autodl-tmp/ottodata/tmp/pe4train_cos_max_sim_{target}.npy')
            cos_max_sim_3 = np.load(f'/root/autodl-tmp/ottodata/tmp/bm4train_cos_max_sim_{target}.npy')

            tfidf_last_sim_1 = np.load(
                f'/root/autodl-tmp/ottodata/tmp/co4train_v2_tfidf_last_sim_{target}.npy')
            tfidf_last_sim_2 = np.load(f'/root/autodl-tmp/ottodata/tmp/pe4train_tfidf_last_sim_{target}.npy')
            tfidf_last_sim_3 = np.load(f'/root/autodl-tmp/ottodata/tmp/bm4train_tfidf_last_sim_{target}.npy')

            tfidf_mean_sim_1 = np.load(
                f'/root/autodl-tmp/ottodata/tmp/co4train_v2_tfidf_mean_sim_{target}.npy')
            tfidf_mean_sim_2 = np.load(f'/root/autodl-tmp/ottodata/tmp/pe4train_tfidf_mean_sim_{target}.npy')
            tfidf_mean_sim_3 = np.load(f'/root/autodl-tmp/ottodata/tmp/bm4train_tfidf_mean_sim_{target}.npy')

            tfidf_max_sim_1 = np.load(f'/root/autodl-tmp/ottodata/tmp/co4train_v2_tfidf_max_sim_{target}.npy')
            tfidf_max_sim_2 = np.load(f'/root/autodl-tmp/ottodata/tmp/pe4train_tfidf_max_sim_{target}.npy')
            tfidf_max_sim_3 = np.load(f'/root/autodl-tmp/ottodata/tmp/bm4train_tfidf_max_sim_{target}.npy')

            covisit_op_weight_1 = np.load(
                f'/root/autodl-tmp/ottodata/tmp/co4train_v2_covisit_op_weight_{target}.npy')
            covisit_op_weight_2 = np.load(
                f'/root/autodl-tmp/ottodata/tmp/pe4train_covisit_op_weight_{target}.npy')
            covisit_op_weight_3 = np.load(
                f'/root/autodl-tmp/ottodata/tmp/bm4train_covisit_op_weight_{target}.npy')
            covisit_op_weight_mean_1 = np.load(
                f'/root/autodl-tmp/ottodata/tmp/co4train_v2_covisit_op_weight_mean_{target}.npy')
            covisit_op_weight_mean_2 = np.load(
                f'/root/autodl-tmp/ottodata/tmp/pe4train_covisit_op_weight_mean_{target}.npy')
            covisit_op_weight_mean_3 = np.load(
                f'/root/autodl-tmp/ottodata/tmp/bm4train_covisit_op_weight_mean_{target}.npy')
            covisit_op_weight_max_1 = np.load(
                f'/root/autodl-tmp/ottodata/tmp/co4train_v2_covisit_op_weight_max_{target}.npy')
            covisit_op_weight_max_2 = np.load(
                f'/root/autodl-tmp/ottodata/tmp/pe4train_covisit_op_weight_max_{target}.npy')
            covisit_op_weight_max_3 = np.load(
                f'/root/autodl-tmp/ottodata/tmp/bm4train_covisit_op_weight_max_{target}.npy')

            covisit_time_weight_1 = np.load(
                f'/root/autodl-tmp/ottodata/tmp/co4train_v2_covisit_time_weight_{target}.npy')
            covisit_time_weight_2 = np.load(
                f'/root/autodl-tmp/ottodata/tmp/pe4train_covisit_time_weight_{target}.npy')
            covisit_time_weight_3 = np.load(
                f'/root/autodl-tmp/ottodata/tmp/bm4train_covisit_time_weight_{target}.npy')
            covisit_time_weight_mean_1 = np.load(
                f'/root/autodl-tmp/ottodata/tmp/co4train_v2_covisit_time_weight_mean_{target}.npy')
            covisit_time_weight_mean_2 = np.load(
                f'/root/autodl-tmp/ottodata/tmp/pe4train_covisit_time_weight_mean_{target}.npy')
            covisit_time_weight_mean_3 = np.load(
                f'/root/autodl-tmp/ottodata/tmp/bm4train_covisit_time_weight_mean_{target}.npy')
            covisit_time_weight_max_1 = np.load(
                f'/root/autodl-tmp/ottodata/tmp/co4train_v2_covisit_time_weight_max_{target}.npy')
            covisit_time_weight_max_2 = np.load(
                f'/root/autodl-tmp/ottodata/tmp/pe4train_covisit_time_weight_max_{target}.npy')
            covisit_time_weight_max_3 = np.load(
                f'/root/autodl-tmp/ottodata/tmp/bm4train_covisit_time_weight_max_{target}.npy')

            itemcf_sim_1 = np.load(f'/root/autodl-tmp/ottodata/tmp/co4train_v2_itemcf_sim_{target}.npy')
            itemcf_sim_2 = np.load(f'/root/autodl-tmp/ottodata/tmp/pe4train_itemcf_sim_{target}.npy')
            itemcf_sim_3 = np.load(f'/root/autodl-tmp/ottodata/tmp/bm4train_itemcf_sim_{target}.npy')

            itemcf_mean_sim_1 = np.load(
                f'/root/autodl-tmp/ottodata/tmp/co4train_v2_itemcf_mean_sim_{target}.npy')
            itemcf_mean_sim_2 = np.load(
                f'/root/autodl-tmp/ottodata/tmp/pe4train_itemcf_mean_sim_{target}.npy')
            itemcf_mean_sim_3 = np.load(
                f'/root/autodl-tmp/ottodata/tmp/bm4train_itemcf_mean_sim_{target}.npy')

            itemcf_max_sim_1 = np.load(
                f'/root/autodl-tmp/ottodata/tmp/co4train_v2_itemcf_max_sim_{target}.npy')
            itemcf_max_sim_2 = np.load(f'/root/autodl-tmp/ottodata/tmp/pe4train_itemcf_max_sim_{target}.npy')
            itemcf_max_sim_3 = np.load(f'/root/autodl-tmp/ottodata/tmp/bm4train_itemcf_max_sim_{target}.npy')

            # bpr_sim_1 = np.load(f'/root/autodl-tmp/ottodata/tmp/co4train_v2_bpr_sim_{target}.npy')
            # bpr_sim_2 = np.load(f'/root/autodl-tmp/ottodata/tmp/pe4train_bpr_sim_{target}.npy')


        else:
            candidates_1 = cudf.read_parquet('/root/autodl-tmp/ottodata/tmp/co4test_v2.parquet')
            candidates_2 = cudf.read_parquet('/root/autodl-tmp/ottodata/tmp/pe4test.parquet')
            candidates_3 = cudf.read_parquet('/root/autodl-tmp/ottodata/tmp/bm4test.parquet')
            w2v_last_sim_1 = np.load(f'/root/autodl-tmp/ottodata/tmp/co4test_v2_w2v_last_sim_{target}.npy')
            w2v_last_sim_2 = np.load(f'/root/autodl-tmp/ottodata/tmp/pe4test_w2v_last_sim_{target}.npy')
            w2v_last_sim_3 = np.load(f'/root/autodl-tmp/ottodata/tmp/bm4test_w2v_last_sim_{target}.npy')
            w2v_mean_sim_1 = np.load(f'/root/autodl-tmp/ottodata/tmp/co4test_v2_w2v_mean_sim_{target}.npy')
            w2v_mean_sim_2 = np.load(f'/root/autodl-tmp/ottodata/tmp/pe4test_w2v_mean_sim_{target}.npy')
            w2v_mean_sim_3 = np.load(f'/root/autodl-tmp/ottodata/tmp/bm4test_w2v_mean_sim_{target}.npy')
            w2v_max_sim_1 = np.load(f'/root/autodl-tmp/ottodata/tmp/co4test_v2_w2v_max_sim_{target}.npy')
            w2v_max_sim_2 = np.load(f'/root/autodl-tmp/ottodata/tmp/pe4test_w2v_max_sim_{target}.npy')
            w2v_max_sim_3 = np.load(f'/root/autodl-tmp/ottodata/tmp/bm4test_w2v_max_sim_{target}.npy')

            bm25_last_sim_1 = np.load(f'/root/autodl-tmp/ottodata/tmp/co4test_v2_bm25_last_sim_{target}.npy')
            bm25_last_sim_2 = np.load(f'/root/autodl-tmp/ottodata/tmp/pe4test_bm25_last_sim_{target}.npy')
            bm25_last_sim_3 = np.load(f'/root/autodl-tmp/ottodata/tmp/bm4test_bm25_last_sim_{target}.npy')

            bm25_mean_sim_1 = np.load(f'/root/autodl-tmp/ottodata/tmp/co4test_v2_bm25_mean_sim_{target}.npy')
            bm25_mean_sim_2 = np.load(f'/root/autodl-tmp/ottodata/tmp/pe4test_bm25_mean_sim_{target}.npy')
            bm25_mean_sim_3 = np.load(f'/root/autodl-tmp/ottodata/tmp/bm4test_bm25_mean_sim_{target}.npy')

            bm25_max_sim_1 = np.load(f'/root/autodl-tmp/ottodata/tmp/co4test_v2_bm25_max_sim_{target}.npy')
            bm25_max_sim_2 = np.load(f'/root/autodl-tmp/ottodata/tmp/pe4test_bm25_max_sim_{target}.npy')
            bm25_max_sim_3 = np.load(f'/root/autodl-tmp/ottodata/tmp/bm4test_bm25_max_sim_{target}.npy')

            cos_last_sim_1 = np.load(f'/root/autodl-tmp/ottodata/tmp/co4test_v2_cos_last_sim_{target}.npy')
            cos_last_sim_2 = np.load(f'/root/autodl-tmp/ottodata/tmp/pe4test_cos_last_sim_{target}.npy')
            cos_last_sim_3 = np.load(f'/root/autodl-tmp/ottodata/tmp/bm4test_cos_last_sim_{target}.npy')

            cos_mean_sim_1 = np.load(f'/root/autodl-tmp/ottodata/tmp/co4test_v2_cos_mean_sim_{target}.npy')
            cos_mean_sim_2 = np.load(f'/root/autodl-tmp/ottodata/tmp/pe4test_cos_mean_sim_{target}.npy')
            cos_mean_sim_3 = np.load(f'/root/autodl-tmp/ottodata/tmp/bm4test_cos_mean_sim_{target}.npy')

            cos_max_sim_1 = np.load(f'/root/autodl-tmp/ottodata/tmp/co4test_v2_cos_max_sim_{target}.npy')
            cos_max_sim_2 = np.load(f'/root/autodl-tmp/ottodata/tmp/pe4test_cos_max_sim_{target}.npy')
            cos_max_sim_3 = np.load(f'/root/autodl-tmp/ottodata/tmp/bm4test_cos_max_sim_{target}.npy')

            tfidf_last_sim_1 = np.load(
                f'/root/autodl-tmp/ottodata/tmp/co4test_v2_tfidf_last_sim_{target}.npy')
            tfidf_last_sim_2 = np.load(f'/root/autodl-tmp/ottodata/tmp/pe4test_tfidf_last_sim_{target}.npy')
            tfidf_last_sim_3 = np.load(f'/root/autodl-tmp/ottodata/tmp/bm4test_tfidf_last_sim_{target}.npy')

            tfidf_mean_sim_1 = np.load(
                f'/root/autodl-tmp/ottodata/tmp/co4test_v2_tfidf_mean_sim_{target}.npy')
            tfidf_mean_sim_2 = np.load(f'/root/autodl-tmp/ottodata/tmp/pe4test_tfidf_mean_sim_{target}.npy')
            tfidf_mean_sim_3 = np.load(f'/root/autodl-tmp/ottodata/tmp/bm4test_tfidf_mean_sim_{target}.npy')

            tfidf_max_sim_1 = np.load(f'/root/autodl-tmp/ottodata/tmp/co4test_v2_tfidf_max_sim_{target}.npy')
            tfidf_max_sim_2 = np.load(f'/root/autodl-tmp/ottodata/tmp/pe4test_tfidf_max_sim_{target}.npy')
            tfidf_max_sim_3 = np.load(f'/root/autodl-tmp/ottodata/tmp/bm4test_tfidf_max_sim_{target}.npy')

            covisit_op_weight_1 = np.load(
                f'/root/autodl-tmp/ottodata/tmp/co4test_v2_covisit_op_weight_{target}.npy')
            covisit_op_weight_2 = np.load(
                f'/root/autodl-tmp/ottodata/tmp/pe4test_covisit_op_weight_{target}.npy')
            covisit_op_weight_3 = np.load(
                f'/root/autodl-tmp/ottodata/tmp/bm4test_covisit_op_weight_{target}.npy')
            covisit_op_weight_mean_1 = np.load(
                f'/root/autodl-tmp/ottodata/tmp/co4test_v2_covisit_op_weight_mean_{target}.npy')
            covisit_op_weight_mean_2 = np.load(
                f'/root/autodl-tmp/ottodata/tmp/pe4test_covisit_op_weight_mean_{target}.npy')
            covisit_op_weight_mean_3 = np.load(
                f'/root/autodl-tmp/ottodata/tmp/bm4test_covisit_op_weight_mean_{target}.npy')
            covisit_op_weight_max_1 = np.load(
                f'/root/autodl-tmp/ottodata/tmp/co4test_v2_covisit_op_weight_max_{target}.npy')
            covisit_op_weight_max_2 = np.load(
                f'/root/autodl-tmp/ottodata/tmp/pe4test_covisit_op_weight_max_{target}.npy')
            covisit_op_weight_max_3 = np.load(
                f'/root/autodl-tmp/ottodata/tmp/bm4test_covisit_op_weight_max_{target}.npy')

            covisit_time_weight_1 = np.load(
                f'/root/autodl-tmp/ottodata/tmp/co4test_v2_covisit_time_weight_{target}.npy')
            covisit_time_weight_2 = np.load(
                f'/root/autodl-tmp/ottodata/tmp/pe4test_covisit_time_weight_{target}.npy')
            covisit_time_weight_3 = np.load(
                f'/root/autodl-tmp/ottodata/tmp/bm4test_covisit_time_weight_{target}.npy')
            covisit_time_weight_mean_1 = np.load(
                f'/root/autodl-tmp/ottodata/tmp/co4test_v2_covisit_time_weight_mean_{target}.npy')
            covisit_time_weight_mean_2 = np.load(
                f'/root/autodl-tmp/ottodata/tmp/pe4test_covisit_time_weight_mean_{target}.npy')
            covisit_time_weight_mean_3 = np.load(
                f'/root/autodl-tmp/ottodata/tmp/bm4test_covisit_time_weight_mean_{target}.npy')
            covisit_time_weight_max_1 = np.load(
                f'/root/autodl-tmp/ottodata/tmp/co4test_v2_covisit_time_weight_max_{target}.npy')
            covisit_time_weight_max_2 = np.load(
                f'/root/autodl-tmp/ottodata/tmp/pe4test_covisit_time_weight_max_{target}.npy')
            covisit_time_weight_max_3 = np.load(
                f'/root/autodl-tmp/ottodata/tmp/bm4test_covisit_time_weight_max_{target}.npy')

            itemcf_sim_1 = np.load(f'/root/autodl-tmp/ottodata/tmp/co4test_v2_itemcf_sim_{target}.npy')
            itemcf_sim_2 = np.load(f'/root/autodl-tmp/ottodata/tmp/pe4test_itemcf_sim_{target}.npy')
            itemcf_sim_3 = np.load(f'/root/autodl-tmp/ottodata/tmp/bm4test_itemcf_sim_{target}.npy')
            itemcf_mean_sim_1 = np.load(
                f'/root/autodl-tmp/ottodata/tmp/co4test_v2_itemcf_mean_sim_{target}.npy')
            itemcf_mean_sim_2 = np.load(f'/root/autodl-tmp/ottodata/tmp/pe4test_itemcf_mean_sim_{target}.npy')
            itemcf_mean_sim_3 = np.load(f'/root/autodl-tmp/ottodata/tmp/bm4test_itemcf_mean_sim_{target}.npy')
            itemcf_max_sim_1 = np.load(
                f'/root/autodl-tmp/ottodata/tmp/co4test_v2_itemcf_max_sim_{target}.npy')
            itemcf_max_sim_2 = np.load(f'/root/autodl-tmp/ottodata/tmp/pe4test_itemcf_max_sim_{target}.npy')
            itemcf_max_sim_3 = np.load(f'/root/autodl-tmp/ottodata/tmp/bm4test_itemcf_max_sim_{target}.npy')

        candidates_1 = candidates_1[candidates_1['type'] == target]
        candidates_1 = candidates_1.drop(columns=['type'])
        candidates_1 = candidates_1.explode('labels').rename(columns={'labels': 'aid'}).reset_index(drop=True)
        candidates_1['ui_w2v_last_sim'] = w2v_last_sim_1
        candidates_1['ui_w2v_mean_sim'] = w2v_mean_sim_1
        candidates_1['ui_w2v_max_sim'] = w2v_max_sim_1
        candidates_1['ui_bm25_last_sim'] = bm25_last_sim_1
        candidates_1['ui_bm25_mean_sim'] = bm25_mean_sim_1
        candidates_1['ui_bm25_max_sim'] = bm25_max_sim_1
        candidates_1['ui_cos_last_sim'] = cos_last_sim_1
        candidates_1['ui_cos_mean_sim'] = cos_mean_sim_1
        candidates_1['ui_cos_max_sim'] = cos_max_sim_1
        candidates_1['ui_tfidf_last_sim'] = tfidf_last_sim_1
        candidates_1['ui_tfidf_mean_sim'] = tfidf_mean_sim_1
        candidates_1['ui_tfidf_max_sim'] = tfidf_max_sim_1
        candidates_1['ui_covisit_op_weight'] = covisit_op_weight_1
        candidates_1['ui_covisit_time_weight'] = covisit_time_weight_1
        candidates_1['ui_covisit_time_weight_mean'] = covisit_time_weight_mean_1
        candidates_1['ui_covisit_time_weight_max'] = covisit_time_weight_max_1
        candidates_1['ui_covisit_op_weight_max'] = covisit_op_weight_max_1
        candidates_1['ui_covisit_op_weight_mean'] = covisit_op_weight_mean_1
        candidates_1['ui_itemcf_sim'] = itemcf_sim_1
        candidates_1['ui_itemcf_mean_sim'] = itemcf_mean_sim_1
        candidates_1['ui_itemcf_max_sim'] = itemcf_max_sim_1
        # candidates_1['ui_bpr_sim'] = bpr_sim_1
        candidates_1['candidate_rank_1'] = candidates_1.groupby('session')['aid'].cumcount().astype('int8')
        del w2v_last_sim_1, w2v_mean_sim_1, w2v_max_sim_1, covisit_op_weight_1, covisit_time_weight_1, itemcf_sim_1, itemcf_mean_sim_1, itemcf_max_sim_1, covisit_time_weight_mean_1, covisit_time_weight_max_1, covisit_op_weight_max_1, covisit_op_weight_mean_1, bm25_last_sim_1, bm25_mean_sim_1, bm25_max_sim_1, cos_last_sim_1, cos_mean_sim_1, cos_max_sim_1, tfidf_last_sim_1, tfidf_mean_sim_1, tfidf_max_sim_1
        gc.collect()

        candidates_2 = candidates_2[candidates_2['type'] == target]
        candidates_2 = candidates_2.drop(columns=['type'])
        candidates_2 = candidates_2.explode('labels').rename(columns={'labels': 'aid'}).reset_index(drop=True)
        candidates_2['ui_w2v_last_sim'] = w2v_last_sim_2
        candidates_2['ui_w2v_mean_sim'] = w2v_mean_sim_2
        candidates_2['ui_w2v_max_sim'] = w2v_max_sim_2
        candidates_2['ui_bm25_last_sim'] = bm25_last_sim_2
        candidates_2['ui_bm25_mean_sim'] = bm25_mean_sim_2
        candidates_2['ui_bm25_max_sim'] = bm25_max_sim_2
        candidates_2['ui_cos_last_sim'] = cos_last_sim_2
        candidates_2['ui_cos_mean_sim'] = cos_mean_sim_2
        candidates_2['ui_cos_max_sim'] = cos_max_sim_2
        candidates_2['ui_tfidf_last_sim'] = tfidf_last_sim_2
        candidates_2['ui_tfidf_mean_sim'] = tfidf_mean_sim_2
        candidates_2['ui_tfidf_max_sim'] = tfidf_max_sim_2
        candidates_2['ui_covisit_op_weight'] = covisit_op_weight_2
        candidates_2['ui_covisit_time_weight'] = covisit_time_weight_2
        candidates_2['ui_covisit_time_weight_mean'] = covisit_time_weight_mean_2
        candidates_2['ui_covisit_time_weight_max'] = covisit_time_weight_max_2
        candidates_2['ui_covisit_op_weight_max'] = covisit_op_weight_max_2
        candidates_2['ui_covisit_op_weight_mean'] = covisit_op_weight_mean_2
        candidates_2['ui_itemcf_sim'] = itemcf_sim_2
        candidates_2['ui_itemcf_mean_sim'] = itemcf_mean_sim_2
        candidates_2['ui_itemcf_max_sim'] = itemcf_max_sim_2
        # candidates_2['ui_bpr_sim'] = bpr_sim_2
        candidates_2['candidate_rank_2'] = candidates_2.groupby('session')['aid'].cumcount().astype('int8')

        candidates_3 = candidates_3[candidates_3['type'] == target]
        candidates_3 = candidates_3.drop(columns=['type'])
        candidates_3 = candidates_3.explode('labels').rename(columns={'labels': 'aid'}).reset_index(drop=True)
        candidates_3['ui_w2v_last_sim'] = w2v_last_sim_3
        candidates_3['ui_w2v_mean_sim'] = w2v_mean_sim_3
        candidates_3['ui_w2v_max_sim'] = w2v_max_sim_3
        candidates_3['ui_bm25_last_sim'] = bm25_last_sim_3
        candidates_3['ui_bm25_mean_sim'] = bm25_mean_sim_3
        candidates_3['ui_bm25_max_sim'] = bm25_max_sim_3
        candidates_3['ui_cos_last_sim'] = cos_last_sim_3
        candidates_3['ui_cos_mean_sim'] = cos_mean_sim_3
        candidates_3['ui_cos_max_sim'] = cos_max_sim_3
        candidates_3['ui_tfidf_last_sim'] = tfidf_last_sim_3
        candidates_3['ui_tfidf_mean_sim'] = tfidf_mean_sim_3
        candidates_3['ui_tfidf_max_sim'] = tfidf_max_sim_3
        candidates_3['ui_covisit_op_weight'] = covisit_op_weight_3
        candidates_3['ui_covisit_time_weight'] = covisit_time_weight_3
        candidates_3['ui_covisit_time_weight_mean'] = covisit_time_weight_mean_3
        candidates_3['ui_covisit_time_weight_max'] = covisit_time_weight_max_3
        candidates_3['ui_covisit_op_weight_max'] = covisit_op_weight_max_3
        candidates_3['ui_covisit_op_weight_mean'] = covisit_op_weight_mean_3
        candidates_3['ui_itemcf_sim'] = itemcf_sim_3
        candidates_3['ui_itemcf_mean_sim'] = itemcf_mean_sim_3
        candidates_3['ui_itemcf_max_sim'] = itemcf_max_sim_3
        # candidates_3['ui_bpr_sim'] = bpr_sim_3
        candidates_3['candidate_rank_3'] = candidates_3.groupby('session')['aid'].cumcount().astype('int8')

        del w2v_last_sim_2, w2v_mean_sim_2, w2v_max_sim_2, covisit_op_weight_2, covisit_time_weight_2, itemcf_sim_2, itemcf_mean_sim_2, itemcf_max_sim_2, covisit_time_weight_mean_2, covisit_time_weight_max_2, covisit_op_weight_max_2, covisit_op_weight_mean_2, w2v_last_sim_3, w2v_mean_sim_3, w2v_max_sim_3, covisit_op_weight_3, covisit_time_weight_3, itemcf_sim_3, itemcf_mean_sim_3, itemcf_max_sim_3, covisit_time_weight_mean_3, covisit_time_weight_max_3, covisit_op_weight_max_3, covisit_op_weight_mean_3, bm25_last_sim_2, bm25_mean_sim_2, bm25_max_sim_2, bm25_last_sim_3, bm25_mean_sim_3, bm25_max_sim_3, cos_last_sim_2, cos_mean_sim_2, cos_max_sim_2, cos_last_sim_3, cos_mean_sim_3, cos_max_sim_3, tfidf_last_sim_2, tfidf_mean_sim_2, tfidf_max_sim_2, tfidf_last_sim_3, tfidf_mean_sim_3, tfidf_max_sim_3
        gc.collect()
        candidates_1 = pl.from_pandas(candidates_1.to_pandas())
        candidates_2 = pl.from_pandas(candidates_2.to_pandas())
        candidates_3 = pl.from_pandas(candidates_3.to_pandas())
        candidates_1 = candidates_1.with_columns([pl.col('aid').cast(pl.Int32), pl.col('session').cast(pl.Int32)])
        candidates_2 = candidates_2.with_columns([pl.col('aid').cast(pl.Int32), pl.col('session').cast(pl.Int32)])
        candidates_3 = candidates_3.with_columns([pl.col('aid').cast(pl.Int32), pl.col('session').cast(pl.Int32)])
        candidates = pl.concat([candidates_1, candidates_2, candidates_3], how='diagonal')
        del candidates_1, candidates_2, candidates_3
        gc.collect()
        candidates = candidates.with_columns(
            [(pl.col('ui_w2v_last_sim') - pl.col('ui_w2v_mean_sim')).alias('ui_sim_chage'),
             (pl.col('ui_itemcf_sim') - pl.col('ui_itemcf_mean_sim')).alias('ui_item_cf_sim_chage')])
        candidates = candidates.unique(subset=["session", "aid"])
        candidates = candidates.fill_null(-1)
        candidates = candidates.filter(pl.col("aid") != -1)
        candidates = candidates.sort([pl.col('session'), pl.col('aid')])
        if IS_TRAIN:
            test_labels = cudf.read_parquet('/root/autodl-tmp/ottodata/valid/test_labels.parquet')
            test_labels = test_labels.loc[test_labels['type'] == target]
            test_labels = test_labels.drop(columns=['type'])
            test_labels = test_labels.explode('ground_truth')
            test_labels.columns = ['session', 'aid']
            test_labels[target] = 1
            test_labels['aid'] = test_labels['aid'].astype('int32')
            test_labels['session'] = test_labels['session'].astype('int32')
            test_labels[target] = test_labels[target].astype('uint8')
            test_labels = pl.from_pandas(test_labels.to_pandas())
            candidates = candidates.join(test_labels, on=['session', 'aid'], how='left')
            candidates = candidates.with_column(pl.col(target).fill_null(0).alias(target))
            del test_labels;
            gc.collect()
        else:
            candidates = candidates.with_column(pl.lit(0).alias(target))

        if IS_TRAIN:
            user_features = pl.read_parquet('/root/autodl-tmp/ottodata/tmp/user_feature_train_v3.parquet')
            item_features = pl.read_parquet('/root/autodl-tmp/ottodata/tmp/item_feature_train_v3.parquet')
            user_item_features = pl.read_parquet(
                '/root/autodl-tmp/ottodata/tmp/user_item_feature_train_v3.parquet')
        else:
            user_features = pl.read_parquet('/root/autodl-tmp/ottodata/tmp/user_feature_test_v3.parquet')
            item_features = pl.read_parquet('/root/autodl-tmp/ottodata/tmp/item_feature_test_v3.parquet')
            user_item_features = pl.read_parquet(
                '/root/autodl-tmp/ottodata/tmp/user_item_feature_test_v3.parquet')
        candidates = candidates.join(item_features, on='aid', how='left').fill_null(-1)
        candidates = candidates.with_columns([
            pl.col(pl.Float64).cast(pl.Float32).keep_name()])
        candidates = candidates.join(user_features, on='session', how='left').fill_null(-1)
        candidates = candidates.with_columns([
            pl.col(pl.Float64).cast(pl.Float32).keep_name()])
        candidates = candidates.join(user_item_features, on=['session', 'aid'], how='left').fill_null(-1)
        candidates = candidates.with_columns([
            pl.col(pl.Float64).cast(pl.Float32).keep_name()])
        del user_features, item_features, user_item_features
        label = 'train' if IS_TRAIN else 'test'
        # candidates = candidates.to_pandas()
        candidates.write_parquet(f'/root/autodl-tmp/ottodata/tmp/{target}_candidates_{label}_v7.parquet')
        del candidates
        torch.cuda.empty_cache()
        gc.collect()
