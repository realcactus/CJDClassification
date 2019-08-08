# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     cal_distribution
   Author :        Xiaosong Zhou
   date：          2019/8/3
-------------------------------------------------
"""
__author__ = 'Xiaosong Zhou'

# 统计语料库中的序列长度分布

TRAIN_DATA_DIR = '../data/small/prepro/train.txt'
# 训练集中最大长度8647个单词
MAX_LENGTH = 8647
distribution = [0 for i in range(87)]


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


if __name__ == '__main__':
    # res = get_max_length(TRAIN_DATA_DIR)
    # print(res)
    cal_distribution(TRAIN_DATA_DIR)
    print(distribution)

