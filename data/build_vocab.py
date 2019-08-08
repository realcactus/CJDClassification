# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     data_process
   Author :        Xiaosong Zhou
   date：          2019/8/1
-------------------------------------------------
"""
__author__ = 'Xiaosong Zhou'

# 数据处理，可用于预处理，也可用于预测时处理原始数据

import re
import numpy as np
import jieba
from collections import Counter


def build_vocab(file_path, vocab_dir, vocab_size=5000):
    contents = []
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.readline()
        while content:
            content_list = content.strip().split()
            contents.extend(content_list)
            content = f.readline()
    counter = Counter(contents)
    count_pairs = counter.most_common(vocab_size - 1)
    words, _ = list(zip(*count_pairs))
    # words = [str(x) for x in count_pairs]
    words = ['<UNK>'] + list(words)
    words = ['<PAD>'] + list(words)
    with open(vocab_dir, 'w', encoding='utf-8') as f:
        f.write('\n'.join(words))


def build_vocab_y(file_path, vocab_dir, vocab_size=5000):
    contents = []
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.readline()
        while content:
            content_list = content.strip().split()
            contents.extend(content_list)
            content = f.readline()
    counter = Counter(contents)
    count_pairs = counter.most_common(vocab_size - 1)
    words, _ = list(zip(*count_pairs))
    # words = [str(x) for x in count_pairs]
    with open(vocab_dir, 'w', encoding='utf-8') as f:
        f.write('\n'.join(words))


if __name__ == '__main__':
    # build_vocab(file_path='../data/small/seg/train_x.txt',
    #             vocab_dir='../data/small/prepro/train_vocab.txt',
    #             vocab_size=30000)
    # build_vocab_y(file_path='../data/small/train_y.txt',
    #               vocab_dir='../data/small/prepro/vocab_y.txt',
    #               vocab_size=30000)
    pass


