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
from helper.word2vec_helper import get_word_embedding


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


def residual_ff(inputs, num_units, config, scope="residual_feedforward"):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # Inner layer

        outputs = tf.layers.dense(inputs, num_units[0], activation=tf.nn.relu)

        # Outer layer
        outputs = tf.layers.dense(outputs, num_units[1])

        # Residual connection
        outputs += inputs

        # Normalize
        outputs = ln(outputs)

        # outputs = tf.contrib.layers.dropout(outputs, config.keep_prob)

    return outputs


def get_token_from_embeddings(inputs, vocab_size, embed_matrix, num_units):
    with tf.variable_scope("embed_weight_matrix", reuse=tf.AUTO_REUSE):
        embed_matrix_array = np.asarray(embed_matrix)

        # embeddings = tf.get_variable('weight_mat',
        #                              dtype=tf.float32,
        #                              shape=(vocab_size, num_units),
        #                              initializer=tf.constant_initializer(embed_matrix_array),
        #                              trainable=True)

        embeddings = tf.get_variable('weight_mat',
                                     dtype=tf.float32,
                                     shape=(vocab_size, num_units))

        # if zero_pad:
        #     embeddings = tf.concat((tf.zeros(shape=[1, num_units]),
        #                             embeddings[1:, :]), 0)

        enc = tf.nn.embedding_lookup(embeddings, inputs, name='emb_look_up')  # (N, T1, d_model)
    return enc


def sentence_attention(inputs, h_dim, attention_size, seq_len):
    trans = tf.transpose(inputs, [1, 0, 2])
    att_input = tf.reshape(trans, [-1, h_dim])
    att_input_list = tf.split(att_input, seq_len, 0)

    with tf.variable_scope("sentence_attention", reuse=tf.AUTO_REUSE):
        attention_w = tf.get_variable('attention_w',
                                      dtype=tf.float32,
                                      shape=(h_dim, attention_size))
        attention_b = tf.get_variable('attention_b',
                                      dtype=tf.float32,
                                      shape=attention_size)
        u_list = []
        for t in range(seq_len):
            u_t = tf.tanh(tf.matmul(att_input_list[t], attention_w) + attention_b)
            u_list.append(u_t)
        u_w = tf.get_variable('attention_uw',
                              dtype=tf.float32,
                              shape=(attention_size, 1))
        attn_z = []
        for t in range(seq_len):
            z_t = tf.matmul(u_list[t], u_w)
            attn_z.append(z_t)
        attn_zconcat = tf.concat(attn_z, axis=1)
        alpha = tf.nn.softmax(attn_zconcat)
        alpha_trans = tf.reshape(tf.transpose(alpha, [1, 0]), [seq_len, -1, 1])
        final_output = tf.reduce_sum(att_input_list * alpha_trans, 0)
    return tf.reduce_sum(tf.transpose(alpha_trans, [1, 0, 2]), axis=2), final_output


def attentive_pooling(inputs_conv, h_dim, attention_size, seq_len):
    # 输入inputs_conv应该是卷积后的张量，shape为(N, seq_len, filter_num)
    # 注意这个seq_len应该等于输入序列长度 - filter_size + 1
    # 128
    filter_seq_len = inputs_conv.shape[2]
    conv = tf.reshape(inputs_conv, [-1, filter_seq_len])
    conv = tf.expand_dims(conv, axis=-1)


    trans = tf.transpose(inputs, [1, 0, 2])
    att_input = tf.reshape(trans, [-1, h_dim])
    att_input_list = tf.split(att_input, seq_len, 0)

    with tf.variable_scope("sentence_attention", reuse=tf.AUTO_REUSE):
        attention_w = tf.get_variable('attention_w',
                                      dtype=tf.float32,
                                      shape=(h_dim, attention_size))
        attention_b = tf.get_variable('attention_b',
                                      dtype=tf.float32,
                                      shape=attention_size)
        u_list = []
        for t in range(seq_len):
            u_t = tf.tanh(tf.matmul(att_input_list[t], attention_w) + attention_b)
            u_list.append(u_t)
        u_w = tf.get_variable('attention_uw',
                              dtype=tf.float32,
                              shape=(attention_size, 1))
        attn_z = []
        for t in range(seq_len):
            z_t = tf.matmul(u_list[t], u_w)
            attn_z.append(z_t)
        attn_zconcat = tf.concat(attn_z, axis=1)
        alpha = tf.nn.softmax(attn_zconcat)
        alpha_trans = tf.reshape(tf.transpose(alpha, [1, 0]), [seq_len, -1, 1])
        final_output = tf.reduce_sum(att_input_list * alpha_trans, 0)
    return tf.reduce_sum(tf.transpose(alpha_trans, [1, 0, 2]), axis=2), final_output


def text_cnn_c(inputs, kernel_sizes, num_filters):
    filters = str(kernel_sizes).split(',')
    pooled_outputs = []
    for filter_size in filters:
        scope_name = str('c-conv-maxpool-%s' % filter_size)
        with tf.device('/gpu:0'), tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
            conv = tf.layers.conv1d(inputs=inputs, filters=num_filters,
                                    kernel_size=int(filter_size), name='c-conv%s' % filter_size)
            conv = tf.nn.relu(conv)

            k = 10
            conv_trans = tf.transpose(conv, [0,2,1])
            gmp_trans, index_gmp = tf.nn.top_k(conv_trans, k)
            gmp = tf.transpose(gmp_trans, perm=[0, 2, 1])

            # top-k-mean池化
            gmp = tf.reduce_mean(gmp, reduction_indices=[1])

            # gmp = tf.reshape(gmp_trans, [-1, conv.shape[2] * k])

            # gmp = tf.reduce_max(conv, reduction_indices=[1], name='c-gmp%s' % filter_size)
        pooled_outputs.append(gmp)
    pool = tf.concat(pooled_outputs, 1)
    return pool


def text_cnn_w(inputs, kernel_sizes, num_filters):
    filters = str(kernel_sizes).split(',')
    pooled_outputs = []
    for filter_size in filters:
        scope_name = str('w-conv-maxpool-%s' % filter_size)
        with tf.device('/gpu:0'), tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
            conv = tf.layers.conv1d(inputs=inputs, filters=num_filters,
                                    kernel_size=int(filter_size), name='w-conv%s' % filter_size)
            conv = tf.nn.relu(conv)
            # gmp = tf.reduce_max(conv, reduction_indices=[1], name='w-gmp%s' % filter_size)

            k = 10
            conv_trans = tf.transpose(conv, [0,2,1])
            gmp_trans, index_gmp = tf.nn.top_k(conv_trans, k)
            gmp = tf.transpose(gmp_trans, perm=[0, 2, 1])

            # top-k-mean池化
            gmp = tf.reduce_mean(gmp, reduction_indices=[1])

            # gmp = tf.reshape(gmp_trans, [-1, conv.shape[2] * k])

            # gmp = tf.reduce_max(conv, reduction_indices=[1], name='w-gmp%s' % filter_size)

        pooled_outputs.append(gmp)
    pool = tf.concat(pooled_outputs, 1)
    return pool


def text_cnn_s(inputs, kernel_sizes, num_filters):
    filters = str(kernel_sizes).split(',')
    pooled_outputs = []
    for filter_size in filters:
        scope_name = str('s-conv-maxpool-%s' % filter_size)
        with tf.device('/gpu:0'), tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
            conv = tf.layers.conv1d(inputs=inputs, filters=num_filters,
                                    kernel_size=int(filter_size), name='s-conv%s' % filter_size)
            conv = tf.nn.relu(conv)
            gmp = tf.reduce_max(conv, reduction_indices=[1], name='s-gmp%s' % filter_size)
        pooled_outputs.append(gmp)
    pool = tf.concat(pooled_outputs, 1)
    return pool


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
