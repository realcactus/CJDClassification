# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     extract_data
   Author :        Xiaosong Zhou
   date：          2019/8/2
-------------------------------------------------
"""
__author__ = 'Xiaosong Zhou'

# 从原始文件中提取数据，原始文件有两种形式，json以及csv

import pandas as pd
import xlrd
import os
import re


def write_content(file_path, content):
    with open(file_path, 'w', encoding='utf-8') as f:
        for line in content:
            x = str(line)
            x = re.sub(r'\r', '', x)
            x = re.sub(r'\n', '', x)
            f.write(x.strip() + '\n')


def read_data_from_csv(file_path, save_path):
    train_path = os.path.join(file_path, 'train.xlsx')
    val_path = os.path.join(file_path, 'valid.xlsx')
    test_path = os.path.join(file_path, 'test.xlsx')
    df_val = pd.read_excel(val_path, header=None, names=['name', 'non', 'time', 'non2', 'money', 'charge',
                                                         'statute_id', 'fact', 'statute_content'], encoding='utf-8')
    df_train = pd.read_excel(train_path, header=None, names=['name', 'non', 'time', 'non2', 'money', 'charge',
                                                             'statute_id', 'fact', 'statute_content'], encoding='utf-8')

    df_test = pd.read_excel(test_path, header=None, names=['name', 'non', 'time', 'non2', 'money', 'charge',
                                                           'statute_id', 'fact', 'statute_content'], encoding='utf-8')
    # train
    train_charge_dir = os.path.join(save_path, 'train_charge.txt')
    write_content(train_charge_dir, df_train['charge'].values)
    train_fact_dir = os.path.join(save_path, 'train_x.txt')
    write_content(train_fact_dir, df_train['fact'].values)
    train_y_id_dir = os.path.join(save_path, 'train_y.txt')
    write_content(train_y_id_dir, df_train['statute_id'].values)
    train_y_content_dir = os.path.join(save_path, 'train_y_content.txt')
    write_content(train_y_content_dir, df_train['statute_content'].values)

    # val
    val_charge_dir = os.path.join(save_path, 'val_charge.txt')
    write_content(val_charge_dir, df_val['charge'].values)
    val_fact_dir = os.path.join(save_path, 'val_x.txt')
    write_content(val_fact_dir, df_val['fact'].values)
    val_y_id_dir = os.path.join(save_path, 'val_y.txt')
    write_content(val_y_id_dir, df_val['statute_id'].values)
    val_y_content_dir = os.path.join(save_path, 'val_y_content.txt')
    write_content(val_y_content_dir, df_val['statute_content'].values)

    # test
    test_charge_dir = os.path.join(save_path, 'test_charge.txt')
    write_content(test_charge_dir, df_test['charge'].values)
    test_fact_dir = os.path.join(save_path, 'test_x.txt')
    write_content(test_fact_dir, df_test['fact'].values)
    test_y_id_dir = os.path.join(save_path, 'test_y.txt')
    write_content(test_y_id_dir, df_test['statute_id'].values)
    test_y_content_dir = os.path.join(save_path, 'test_y_content.txt')
    write_content(test_y_content_dir, df_test['statute_content'].values)


if __name__ == '__main__':
    read_data_from_csv('../data/small', '../data/small/')