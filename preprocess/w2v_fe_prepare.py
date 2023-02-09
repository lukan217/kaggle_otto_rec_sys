# %%
import polars as pl
from gensim.test.utils import common_texts
from gensim.models import Word2Vec
import pandas as pd
import numpy as np
from collections import defaultdict
from annoy import AnnoyIndex
import argparse
parser = argparse.ArgumentParser(description='Test for argparse')
parser.add_argument('--is_train', '-v', type=bool, help='is_train',default=False)

IS_TRAIN = args.is_train

if IS_TRAIN:
    train = pl.read_parquet('/root/autodl-tmp/ottodata/valid/train.parquet')
    test = pl.read_parquet('/root/autodl-tmp/ottodata/valid/test.parquet')
else:
    train = pl.read_parquet('/root/autodl-tmp/ottodata/train.parquet')
    test = pl.read_parquet('/root/autodl-tmp/ottodata/test.parquet')

# %%
sentences_df = pl.concat([train, test]).groupby('session').agg(
    pl.col('aid').alias('sentence')
)
sentences = sentences_df['sentence'].to_list()
w2vec = Word2Vec(sentences=sentences, vector_size= 64, window = 3, negative = 8, ns_exponent = 0.2, sg = 1, min_count=1, workers=15)
aid2idx = {aid: i for i, aid in enumerate(w2vec.wv.index_to_key)}

index = AnnoyIndex(64, 'dot')
for aid, idx in aid2idx.items():
    index.add_item(idx, w2vec.wv.vectors[idx]/np.linalg.norm(w2vec.wv.vectors[idx]))
index.build(50)
if IS_TRAIN:
    w2vec.save('/root/autodl-tmp/ottodata/tmp/w2v4train.model')
    index.save('/root/autodl-tmp/ottodata/tmp/index4train.ann')
else:
    w2vec.save('/root/autodl-tmp/ottodata/tmp/w2v4test.model')
    index.save('/root/autodl-tmp/ottodata/tmp/index4test.ann')

# %%



