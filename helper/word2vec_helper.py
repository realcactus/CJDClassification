# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     word2vec_helper
   Author :        Xiaosong Zhou
   date：          2019/8/8
-------------------------------------------------
"""
__author__ = 'Xiaosong Zhou'
from gensim.models.word2vec import Word2Vec
from gensim.models import KeyedVectors
import numpy as np


# 训练，测试以及生成word2vec权重矩阵

def test_word2vec(model_dir):
    # w2v_model = Word2Vec.load(model_dir)
    w2v_model = KeyedVectors.load_word2vec_format(model_dir, binary=True)
    # print(w2v_model.most_similar('受伤'))
    # print(w2v_model.most_similar('刀'))
    # print(w2v_model.most_similar('故意'))
    # print(w2v_model.most_similar('偷窃'))
    # print(w2v_model.most_similar('<UNK>'))
    vocabs = w2v_model.wv.vocab.keys()
    with open('../helper/word2vec/baike_vocab.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(vocabs))


def get_word_embedding(model_dir, vocab_dir, save_dir, embed_dim):
    # 这里的逻辑是，unk用一个随机emb维向量，pad用一个全0向量
    w2v_model = KeyedVectors.load_word2vec_format(model_dir, binary=True)
    # vocab = [line for line in open(vocab_dir, 'r', encoding='utf-8').read().splitlines()]
    vocab = open(vocab_dir, 'r', encoding='utf-8').read().splitlines()
    weight_matrix = []
    unk_emb = np.asarray([-1 + 2 * np.random.rand() for _ in range(embed_dim)])
    unk_num = 0
    for i in range(len(vocab)):
        if i == 0:
            # pad
            weight_matrix.append(np.zeros(embed_dim, dtype='float32'))
        elif i == 1:
            # unk
            # weight_matrix.append(unk_emb)
            # 用pad代替试试看
            weight_matrix.append(np.zeros(embed_dim, dtype='float32'))
        else:
            word = vocab[i]
            if word in w2v_model:
                weight_matrix.append(np.asarray(w2v_model[word]))
            else:
                # weight_matrix.append(unk_emb)
                unk_num += 1
                # 用pad代替试试看
                weight_matrix.append(np.zeros(embed_dim, dtype='float32'))
    # 写入
    with open(save_dir, 'w', encoding='utf-8') as f:
        for line in weight_matrix:
            str_line = [str(x) for x in line]
            f.write(' '.join(str_line) + '\n')

    print('未查到词语数：' + str(unk_num))


def train_w2v(file_dir):
    content = []
    with open(file_dir, 'r', encoding='utf-8') as f:
        line = f.readline()
        while line:
            line_list = line.strip().split()
            content.append(line_list)
            line = f.readline()
    model = Word2Vec(content, size=100, window=5, min_count=5, workers=1)
    model.save('../helper/word2vec/test_word2vec')



if __name__ == '__main__':
    # test_word2vec('../helper/word2vec/baike_26g_news_13g_novel_229g.bin')

    get_word_embedding(model_dir='../helper/word2vec/baike_26g_news_13g_novel_229g.bin',
                       vocab_dir='../data/small/prepro/train_vocab.txt',
                       save_dir='../data/small/prepro/vocab_emb.txt',
                       embed_dim=128)




