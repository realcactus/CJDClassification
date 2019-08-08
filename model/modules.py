# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     modules
   Author :        Xiaosong Zhou
   date：          2019/8/3
-------------------------------------------------
"""
__author__ = 'Xiaosong Zhou'

# 一些模块，包括层归一化，卷积等等

import numpy as np
import tensorflow as tf


def ln(inputs, epsilon=1e-8, scope="ln"):
    '''
        层归一layer normalization
        tensorflow 在实现 Batch Normalization（各个网络层输出的归一化）时，主要用到nn.moments和batch_normalization
        其中moments作用是统计矩，mean 是一阶矩，variance 则是二阶中心矩
        tf.nn.moments 计算返回的 mean 和 variance 作为 tf.nn.batch_normalization 参数进一步调用
        :param inputs: 一个有2个或更多维度的张量，第一个维度是batch_size
        :param epsilon: 很小的数值，防止区域划分错误
        :param scope:
        :return: 返回一个与inputs相同shape和数据的dtype
    '''
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]
        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.get_variable("beta", params_shape, initializer=tf.zeros_initializer())
        gamma = tf.get_variable("gamma", params_shape, initializer=tf.ones_initializer())
        normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
        outputs = gamma * normalized + beta
    return outputs


def get_token_embeddings(vocab_size, num_units, zero_pad=True):
    with tf.variable_scope("embed_weight_matrix", reuse=tf.AUTO_REUSE):
        embeddings = tf.get_variable('weight_mat',
                                     dtype=tf.float32,
                                     shape=(vocab_size, num_units))
        if zero_pad:
            embeddings = tf.concat((tf.zeros(shape=[1, num_units]),
                                    embeddings[1:, :]), 0)
    return embeddings


def mlp(inputs, units, drop_rate, training=True):
    # mlp方法，也就是单纯用词向量结合全连接的方法来做分类
    with tf.variable_scope("fnn_mean_pooling", reuse=tf.AUTO_REUSE):
        mean_pool = tf.reduce_mean(inputs, axis=1)  # (N, d_model)
        mean_pool = tf.layers.dense(mean_pool, units * 2, activation=tf.nn.relu)
        mean_pool = tf.layers.dense(mean_pool, units, activation=tf.nn.relu)
        mean_pool = tf.layers.dropout(mean_pool, rate=drop_rate, training=training)
    return mean_pool


def score_layer(inputs, y_dim):
    # 最后一层score层
    with tf.variable_scope("score", reuse=tf.AUTO_REUSE):
        logits = tf.layers.dense(inputs, y_dim)
    y_pred_cls = tf.argmax(tf.nn.softmax(logits), 1)  # 预测类别
    return logits, y_pred_cls
