# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     cal_distribution
   Author :        Xiaosong Zhou
   date：          2019/8/3
-------------------------------------------------
"""
__author__ = 'Xiaosong Zhou'
import re

# 统计语料库中的序列长度分布
# TRAIN_DATA_DIR = '../data/legal_domain/train_x.txt'
TRAIN_DATA_DIR = '../data/cnews/train_x.txt'
# TRAIN_DATA_DIR = '../data/legal_domain/prepro/train_c.txt'
# 训练集中最大长度8647个单词
MAX_LENGTH = 8647
distribution = [0 for i in range(87)]

# 训练集中最大长度103个句子
# sentence_distribution = [0 for i in range(104)]
sentence_distribution = [0 for i in range(648)]


def get_max_length(file_path):
    max_len = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.readline()
        while content:
            max_len = max(max_len, len(content.strip().split()))
            content = f.readline()
    return max_len


def cal_distribution(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.readline()
        while content:
            tmp_length = len(content.strip().split())
            distribution[int(tmp_length / 100)] += 1
            content = f.readline()


def get_max_sentence_length(file_path):
    max_len = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.readline()
        while content:
            sentences = re.split('。', content)
            max_len = max(max_len, len(sentences))
            content = f.readline()
    return max_len


def cal_sentence_distribution(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.readline()
        while content:
            sentences = re.split('。', content)
            tmp_length = len(sentences)
            sentence_distribution[tmp_length] += 1
            content = f.readline()


if __name__ == '__main__':
    # res = get_max_length(TRAIN_DATA_DIR)
    # print(res)
    # cal_distribution(TRAIN_DATA_DIR)
    # print(distribution)
    res = get_max_sentence_length(TRAIN_DATA_DIR)
    print(res)
    cal_sentence_distribution(TRAIN_DATA_DIR)
    for i in range(80):
        print(sentence_distribution[i])
    print(sentence_distribution)
