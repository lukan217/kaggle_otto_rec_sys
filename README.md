# 运行环境
大概需要160G+内存，250G硬盘

# 候选召回

1. covisit矩阵
2. 用户过去发生行为的商品序列
3. itemcf


# 特征工程

特征分为四类：

1. 用户特征
2. 商品特征
3. 用户商品交互特征
4. 相似度特征：itemcf,word2vec,covist生成的相似度特征

# 模型
采用lgbranker，单模，线上0.596