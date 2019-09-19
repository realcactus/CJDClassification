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
from model.modules import get_token_from_embeddings, mlp, score_layer, text_cnn_c, text_cnn_w, text_cnn_s, sentence_attention, residual_ff


class TCNNBertConfig(object):
    """CNN配置参数"""
    embedding_dim = 768  # BERT词向量维度
    embedding_dim_w = 100  # 随机词向量维度
    seq_length_w = 800  # 词语级序列长度
    seq_length_c = 512  # 字符级序列长度
    sentence_length = 80  # 句子长度，代表一个样本最多有多少个句子
    # sentence_length = 20  # 句子长度，代表一个样本最多有多少个句子
    num_classes = 183  # 类别数
    num_filters = 128  # 卷积核数目
    attention_size = 128  # 句子序列的attention size
    kernel_sizes_c = "3,4,5"  # 词语级卷积核尺寸
    kernel_sizes_w = "3,4,5"  # 词语级卷积核尺寸
    kernel_sizes_s = "2"  # 句子级卷积核尺寸
    vocab_size_w = 30000  # 词表大小
    vocab_size_c = 5000  # 词表大小
    hidden_dim = 128  # 全连接层神经元
    dropout_keep_prob = 0.8  # dropout保留比例
    learning_rate = 1e-3  # 学习率
    batch_size = 64  # 每批训练大小
    num_epochs = 30  # 总迭代轮次

    print_per_batch = 20  # 每多少个batch输出一次结果
    save_per_batch = 10  # 每多少轮存入tensorboard

    embed_matrix = []  # 词向量embed权重矩阵


class TextCNNBert(object):
    """文本分类，CNN模型"""

    def __init__(self, config):
        self.config = config
        # 待输入的数据
        # bert版输入
        # (N, T1, d_model)
        self.enc = tf.placeholder(tf.float32, [None, self.config.seq_length_c,
                                               self.config.embedding_dim], name='input_x_c')
        # (N, S, d_model)
        # 句子级输入
        self.enc_sentence = tf.placeholder(tf.float32, [None, self.config.sentence_length,
                                                        self.config.embedding_dim], name='input_x_s')

        self.input_x_w = tf.placeholder(tf.int32, [None, self.config.seq_length_w], name='input_x_w')
        self.enc_w = get_token_from_embeddings(self.input_x_w,
                                               self.config.vocab_size_w,
                                               self.config.embed_matrix,
                                               self.config.embedding_dim_w)

        self.input_y = tf.placeholder(tf.int32, [None], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        # CNN
        # 字符级
        self.max_pool_c = text_cnn_c(inputs=self.enc,
                                     kernel_sizes=self.config.kernel_sizes_c,
                                     num_filters=self.config.num_filters)

        # 词语级
        self.max_pool_w = text_cnn_w(inputs=self.enc_w,
                                     kernel_sizes=self.config.kernel_sizes_w,
                                     num_filters=self.config.num_filters)

        # 句子级
        # self.max_pool_s = text_cnn_s(inputs=self.enc_sentence,
        #                              kernel_sizes=self.config.kernel_sizes_s,
        #                              num_filters=self.config.num_filters)
        self.atten, self.att_pool = sentence_attention(self.enc_sentence,
                                                       self.config.embedding_dim,
                                                       self.config.attention_size,
                                                       self.config.sentence_length)
        self.max_pool_s = tf.layers.dense(self.att_pool, self.config.hidden_dim * 3, activation=tf.nn.relu)

        pooled_outputs_cws = []
        pooled_outputs_cws.append(self.max_pool_c)
        pooled_outputs_cws.append(self.max_pool_w)
        pooled_outputs_cws.append(self.max_pool_s)
        # 字词句结合
        self.pooled_outputs_cws = tf.concat(pooled_outputs_cws, 1)
        print('###################################################################')
        print(self.pooled_outputs_cws.shape)

        with tf.variable_scope("fnn", reuse=tf.AUTO_REUSE):
            # self.fnn_pool = tf.layers.dense(self.pooled_outputs_cws, self.config.hidden_dim * 8, activation=tf.nn.relu)
            # self.fnn_pool = tf.layers.dense(self.fnn_pool, self.config.hidden_dim * 4, activation=tf.nn.relu)
            # self.fnn_pool = tf.layers.dense(self.fnn_pool, self.config.hidden_dim * 2, activation=tf.nn.relu)

            self.fnn_pool = tf.layers.dense(self.pooled_outputs_cws, self.config.hidden_dim, activation=tf.nn.relu)
            self.fnn_pool = tf.contrib.layers.dropout(self.fnn_pool, self.keep_prob)

        self.logits, self.y_pred_cls = score_layer(inputs=self.fnn_pool, y_dim=self.config.num_classes)
        y_ = tf.one_hot(self.input_y, depth=self.config.num_classes)
        ce = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=y_)
        self.loss = tf.reduce_mean(ce)
        self.optimizer = tf.train.AdamOptimizer(self.config.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss)
        self.correct_pred = tf.equal(tf.argmax(y_, 1), self.y_pred_cls)
        self.acc = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))









