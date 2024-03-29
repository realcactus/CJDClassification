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
import h5py


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


def batch_iter_c(x, y, batch_size=64, shuffle=True):
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1

    if shuffle:
        indices = np.random.permutation(np.arange(data_len))
        x_shuffle = np.asarray(x)[indices]
        y_shuffle = np.asarray(y)[indices]
    else:
        x_shuffle = x[:]
        y_shuffle = y[:]
    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]


def batch_iter(x, x_w, x_sentence, y, batch_size=64, shuffle=True):
    """生成批次数据"""
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1

    if shuffle:
        indices = np.random.permutation(np.arange(data_len))
        x_shuffle = np.asarray(x)[indices]
        x_shuffle_w = np.asarray(x_w)[indices]
        x_shuffle_sentence = np.asarray(x_sentence)[indices]
        y_shuffle = np.asarray(y)[indices]
    else:
        x_shuffle = x[:]
        x_shuffle_w = x_w[:]
        x_shuffle_sentence = x_sentence[:]
        y_shuffle = y[:]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], x_shuffle_w[start_id:end_id], x_shuffle_sentence[start_id:end_id], y_shuffle[start_id:end_id]
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


def read_data_x_from_h5(file_dir, mode):
    # 该方法直接从h5文件中读到已经转换好的数据，不需要嵌入层
    if mode == 'train':
        open_file = h5py.File(file_dir, 'r')
        data_train = open_file['train'][:]
        data_val = open_file['val'][:]
        open_file.close()
        return data_train, data_val
    elif mode == 'test':
        open_file = h5py.File(file_dir, 'r')
        data_test = open_file['test'][:]
        open_file.close()
        return data_test
    return None


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
    # val_c
    # write_x_to_ids(seg_file='../data/legal_domain/seg/val_x_c.txt',
    #                vocab_file='../data/legal_domain/prepro/train_vocab_c.txt',
    #                save_file='../data/legal_domain/prepro/val_c.txt')

    # val_w
    write_x_to_ids(seg_file='../data/imdb/seg/val_x_w.txt',
                   vocab_file='../data/imdb/prepro/train_vocab_w.txt',
                   save_file='../data/imdb/prepro/val_w.txt')

    # train_c
    # write_x_to_ids(seg_file='../data/legal_domain/seg/train_x_c.txt',
    #                vocab_file='../data/legal_domain/prepro/train_vocab_c.txt',
    #                save_file='../data/legal_domain/prepro/train_c.txt')

    # train_w
    write_x_to_ids(seg_file='../data/imdb/seg/train_x_w.txt',
                   vocab_file='../data/imdb/prepro/train_vocab_w.txt',
                   save_file='../data/imdb/prepro/train_w.txt')

    # test_c
    # write_x_to_ids(seg_file='../data/legal_domain/seg/test_x_c.txt',
    #                vocab_file='../data/legal_domain/prepro/train_vocab_c.txt',
    #                save_file='../data/legal_domain/prepro/test_c.txt')

    # test_w
    write_x_to_ids(seg_file='../data/imdb/seg/test_x_w.txt',
                   vocab_file='../data/imdb/prepro/train_vocab_w.txt',
                   save_file='../data/imdb/prepro/test_w.txt')

    # y
    write_y_to_ids(y_file='../data/imdb/val_y.txt',
                   vocab_file='../data/imdb/prepro/vocab_y.txt',
                   save_file='../data/imdb/prepro/val_y.txt')
    write_y_to_ids(y_file='../data/imdb/train_y.txt',
                   vocab_file='../data/imdb/prepro/vocab_y.txt',
                   save_file='../data/imdb/prepro/train_y.txt')
    write_y_to_ids(y_file='../data/imdb/test_y.txt',
                   vocab_file='../data/imdb/prepro/vocab_y.txt',
                   save_file='../data/imdb/prepro/test_y.txt')