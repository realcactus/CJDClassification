# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     TextCNN
   Author :        Xiaosong Zhou
   date：          2019/8/4
-------------------------------------------------
"""
__author__ = 'Xiaosong Zhou'

import tensorflow as tf
import numpy as np
from model.modules import score_layer, get_token_from_embeddings, text_cnn_c


class TCNNConfig(object):
    """CNN配置参数"""

    embedding_dim = 128  # 词向量维度
    # seq_length_c = 600  # 字符级序列长度
    seq_length_w = 900  # 词语级序列长度
    seq_length_c = 1500  # 字符级序列长度
    num_classes = 183  # 类别数
    num_filters = 128  # 卷积核数目
    # kernel_sizes = "3,4,5"  # 卷积核尺寸
    kernel_sizes_c = "3,4,5"  # 词语级卷积核尺寸
    # vocab_size_c = 100000  # 字表大小
    vocab_size_w = 30000  # 词表大小
    vocab_size_c = 5000  # 词表大小
    hidden_dim = 128  # 全连接层神经元
    dropout_keep_prob = 0.8  # dropout保留比例
    learning_rate = 1e-3  # 学习率
    batch_size = 64  # 每批训练大小
    num_epochs = 50  # 总迭代轮次

    print_per_batch = 20  # 每多少个batch输出一次结果
    save_per_batch = 10  # 每多少轮存入tensorboard

    embed_matrix = []  # 词向量embed权重矩阵


class TextCNN(object):
    """文本分类，CNN模型"""

    def __init__(self, config):
        self.config = config
        # 待输入的数据
        self.input_x_c = tf.placeholder(tf.int32, [None, self.config.seq_length_c], name='input_x_c')
        # self.input_x_w = tf.placeholder(tf.int32, [None, self.config.seq_length_w], name='input_x_w')
        self.input_y = tf.placeholder(tf.int32, [None], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.enc = get_token_from_embeddings(self.input_x_c,
                                             self.config.vocab_size_c,
                                             None,
                                             self.config.embedding_dim)

        # self.enc = tf.nn.embedding_lookup(self.embeddings, self.input_x_w)  # (N, T1, d_model)
        # NN网
        # self.mean_pool = mlp(inputs=self.enc, drop_rate=1 - self.config.dropout_keep_prob,
        #                      units=self.config.hidden_dim, training=True)
        # CNN
        self.max_pool = text_cnn_c(inputs=self.enc,
                                   kernel_sizes=self.config.kernel_sizes_c,
                                   num_filters=self.config.num_filters)

        # self.fnn_pool = tf.layers.dense(self.max_pool, self.config.hidden_dim * 2, activation=tf.nn.relu)
        self.fnn_pool = tf.layers.dense(self.max_pool, self.config.hidden_dim, activation=tf.nn.relu)
        self.fnn_pool = tf.contrib.layers.dropout(self.fnn_pool, self.keep_prob)

        self.logits, self.y_pred_cls = score_layer(inputs=self.fnn_pool, y_dim=self.config.num_classes)
        y_ = tf.one_hot(self.input_y, depth=self.config.num_classes)
        ce = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=y_)
        self.loss = tf.reduce_mean(ce)
        self.optimizer = tf.train.AdamOptimizer(self.config.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss)
        self.correct_pred = tf.equal(tf.argmax(y_, 1), self.y_pred_cls)
        self.acc = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))









