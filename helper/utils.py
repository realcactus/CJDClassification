# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     utils
   Author :        Xiaosong Zhou
   date：          2019/8/5
-------------------------------------------------
"""
__author__ = 'Xiaosong Zhou'

# 一些工具方法

import tensorflow as tf


def postprocess(hypotheses, idx2token):
    # 将预测数字序列还原成token序列
    _hypotheses = []
    for h in hypotheses:
        sent = "".join(idx2token[h])
        _hypotheses.append(sent.strip())
    return _hypotheses


def save_variable_specs(fpath):
    '''Saves information about variables such as
    their name, shape, and total parameter number
    fpath: string. output file path

    Writes
    a text file named fpath.
    '''
    def _get_size(shp):
        '''Gets size of tensor shape
        shp: TensorShape

        Returns
        size
        '''
        size = 1
        for d in range(len(shp)):
            size *=shp[d]
        return size

    params, num_params = [], 0
    # for v in tf.global_variables():
    for v in tf.trainable_variables():
        params.append("{}==={}".format(v.name, v.shape))
        num_params += _get_size(v.shape)
    print("num_params: ", num_params)
    with open(fpath, 'w') as fout:
        fout.write("num_params: {}\n".format(num_params))
        fout.write("\n".join(params))
    print("Variables info has been saved.")
