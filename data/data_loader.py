# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     data_loader
   Author :        Xiaosong Zhou
   date：          2019/8/2
-------------------------------------------------
"""
__author__ = 'Xiaosong Zhou'

# 加载数据文件

import pandas as pd
import os
import re
import numpy as np
import tensorflow as tf


def load_vocab(vocab_dir):
    vocab = [line for line in open(vocab_dir, 'r', encoding='utf-8').read().splitlines()]
    token2idx = {token: idx for idx, token in enumerate(vocab)}
    idx2token = {idx: token for idx, token in enumerate(vocab)}
    return token2idx, idx2token


# 这里默认数据已经分词好
def write_x_to_ids(seg_file, vocab_file, save_file):
    contents = []
    token2id, _ = load_vocab(vocab_file)
    with open(seg_file, 'r', encoding='utf-8') as f:
        content = f.readline()
        while content:
            content_list = content.strip().split()
            content_list_ids = [token2id.get(t, token2id["<UNK>"]) for t in content_list]
            contents.append(content_list_ids)
            content = f.readline()
    with open(save_file, 'w', encoding='utf-8') as f:
        for line in contents:
            str_line = [str(x) for x in line]
            f.write(' '.join(str_line) + '\n')


# 因为目前看作单分类任务，所以将y_id中的第一个保留，其他删去
def write_y_to_ids(y_file, vocab_file, save_file):
    contents = []
    token2id, _ = load_vocab(vocab_file)
    with open(y_file, 'r', encoding='utf-8') as f:
        content = f.readline()
        while content:
            id_1 = content.strip().split()[0]
            res = token2id.get(id_1)
            contents.append(res)
            content = f.readline()
    with open(save_file, 'w', encoding='utf-8') as f:
        for line in contents:
            f.write(str(line).strip() + '\n')


# 这里是输入原始文本数据
def get_word_to_id():
    pass


def batch_iter(x, y, batch_size=64, shuffle=True):
    """生成批次数据"""
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1

    if shuffle:
        indices = np.random.permutation(np.arange(data_len))
        x_shuffle = np.array(x)[indices]
        y_shuffle = np.array(y)[indices]
    else:
        x_shuffle = x[:]
        y_shuffle = y[:]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]
        # 最后一组不到batch_size的组丢掉，因为rnn好像不允许这样
        # if (i + 1) * batch_size <= data_len:
        #     end_id = (i + 1) * batch_size
        #     yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]


def read_data_x(file_dir, max_length):
    # 加载数据进行训练/测试
    contents = []
    with open(file_dir, 'r', encoding='utf-8') as f:
        content = f.readline()
        while content:
            content_list = [int(x) for x in content.strip().split()]
            contents.append(content_list)
            content = f.readline()
    x_pad = tf.keras.preprocessing.sequence.pad_sequences(contents, max_length,
                                                          padding='post', truncating='post')
    return x_pad


def read_data_y(file_dir):
    contents = []
    with open(file_dir, 'r', encoding='utf-8') as f:
        content = f.readline()
        while content:
            y_id = int(content.strip())
            contents.append(y_id)
            content = f.readline()
    return contents


if __name__ == '__main__':
    write_x_to_ids(seg_file='../data/small/seg/val_x.txt',
                   vocab_file='../data/small/prepro/train_vocab.txt',
                   save_file='../data/small/prepro/val.txt')
    write_x_to_ids(seg_file='../data/small/seg/train_x.txt',
                   vocab_file='../data/small/prepro/train_vocab.txt',
                   save_file='../data/small/prepro/train.txt')
    write_x_to_ids(seg_file='../data/small/seg/test_x.txt',
                   vocab_file='../data/small/prepro/train_vocab.txt',
                   save_file='../data/small/prepro/test.txt')
    write_y_to_ids(y_file='../data/small/val_y.txt',
                   vocab_file='../data/small/prepro/vocab_y.txt',
                   save_file='../data/small/prepro/val_y.txt')
    write_y_to_ids(y_file='../data/small/train_y.txt',
                   vocab_file='../data/small/prepro/vocab_y.txt',
                   save_file='../data/small/prepro/train_y.txt')
    write_y_to_ids(y_file='../data/small/test_y.txt',
                   vocab_file='../data/small/prepro/vocab_y.txt',
                   save_file='../data/small/prepro/test_y.txt')