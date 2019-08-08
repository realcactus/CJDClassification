# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     train
   Author :        Xiaosong Zhou
   date：          2019/8/4
-------------------------------------------------
"""
__author__ = 'Xiaosong Zhou'

# 训练代码

import os
import sys
import time
from datetime import timedelta

import numpy as np
import tensorflow as tf
from data.data_loader import read_data_x, read_data_y
from sklearn import metrics
from model.TextCNN import TextCNN, TCNNConfig
from data.data_loader import batch_iter, load_vocab
from helper.utils import postprocess, save_variable_specs


base_dir = 'data/small/prepro'
train_dir = os.path.join(base_dir, 'train.txt')
train_y_dir = os.path.join(base_dir, 'train_y.txt')

val_dir = os.path.join(base_dir, 'val.txt')
val_y_dir = os.path.join(base_dir, 'val_y.txt')

test_dir = os.path.join(base_dir, 'test.txt')
test_y_dir = os.path.join(base_dir, 'test_y.txt')

vocab_dir = os.path.join(base_dir, 'train_vocab.txt')
vocab_y_dir = os.path.join(base_dir, 'vocab_y.txt')

save_dir = 'model/checkpoints/textcnn'
save_path = os.path.join(save_dir, 'best_validation')  # 最佳验证结果保存路径
log_dir = 'model/log'
val_hypothesis_dir = os.path.join(log_dir, 'val_hypothesis.txt')
test_hypothesis_dir = os.path.join(log_dir, 'test_hypothesis.txt')


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def feed_data(x_batch, y_batch, keep_prob):
    feed_dict = {
        model.input_x_w: x_batch,
        model.input_y: y_batch,
        model.keep_prob: keep_prob
    }
    return feed_dict


def evaluate(sess, x_, y_):
    """评估在某一数据上的准确率和损失"""
    data_len = len(x_)
    batch_eval = batch_iter(x_, y_, 128, shuffle=False)
    total_loss = 0.0
    total_acc = 0.0
    total_pred_cls = []
    for x_batch, y_batch in batch_eval:
        batch_len = len(x_batch)
        feed_dict = feed_data(x_batch, y_batch, 1.0)
        loss, acc, pred_cls = sess.run([model.loss, model.acc, model.y_pred_cls], feed_dict=feed_dict)
        total_loss += loss * batch_len
        total_acc += acc * batch_len
        total_pred_cls.extend(pred_cls.tolist())

    return total_loss / data_len, total_acc / data_len, total_pred_cls


def train():
    print("Configuring TensorBoard and Saver...")
    # 配置 Tensorboard，重新训练时，请将tensorboard文件夹删除，不然图会覆盖
    tensorboard_dir = 'model/tensorboard/textcnn'
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)
    tf.summary.scalar("loss", model.loss)
    tf.summary.scalar("accuracy", model.acc)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir)

    # 配置 Saver
    saver = tf.train.Saver()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print("Loading training and validation data...")
    # 载入训练集与验证集
    start_time = time.time()
    x_train = read_data_x(train_dir, config.seq_length_w)
    y_train = read_data_y(train_y_dir)
    x_val = read_data_x(val_dir, config.seq_length_w)
    y_val = read_data_y(val_y_dir)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # 创建session
    session = tf.Session()
    checkpoint = tf.train.latest_checkpoint(save_dir)
    if checkpoint is None:
        print("Initializing from scratch")
        session.run(tf.global_variables_initializer())
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        save_variable_specs(os.path.join(log_dir, "specs"))
    else:
        print("Initializing from checkpoint")
        saver.restore(session, checkpoint)
    # session.run(tf.global_variables_initializer())
    writer.add_graph(session.graph)

    print('Training and evaluating...')
    start_time = time.time()
    total_batch = 0  # 总批次
    best_acc_val = 0.0  # 最佳验证集准确率
    last_improved = 0  # 记录上一次提升批次
    require_improvement = 5000  # 如果超过5000轮未提升，提前结束训练

    flag = False
    for epoch in range(config.num_epochs):
        print('Epoch:', epoch + 1)
        batch_train = batch_iter(x_train, y_train, config.batch_size)
        for x_batch, y_batch in batch_train:
            feed_dict = feed_data(x_batch, y_batch, config.dropout_keep_prob)

            if total_batch % config.save_per_batch == 0:
                # 每多少轮次将训练结果写入tensorboard scalar
                s = session.run(merged_summary, feed_dict=feed_dict)
                writer.add_summary(s, total_batch)

            if total_batch % config.print_per_batch == 0:
                # 每多少轮次输出在训练集和验证集上的性能
                feed_dict[model.keep_prob] = 1.0
                loss_train, acc_train = session.run([model.loss, model.acc], feed_dict=feed_dict)
                loss_val, acc_val, pred_cls = evaluate(session, x_val, y_val)

                if acc_val > best_acc_val:
                    # 保存最好结果
                    best_acc_val = acc_val
                    last_improved = total_batch
                    saver.save(sess=session, save_path=save_path)
                    improved_str = '*'
                    # 记录验证集上的预测结果
                    hypotheses = postprocess(pred_cls, id2token_y)
                    with open(val_hypothesis_dir, 'w', encoding='utf-8') as fout:
                        fout.write("\n".join(hypotheses))
                else:
                    improved_str = ''

                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6}, Train Loss: {1:>8.4}, Train Acc: {2:>9.4%},' \
                      + ' Val Loss: {3:>8.4}, Val Acc: {4:>9.4%}, Time: {5} {6}'
                print(msg.format(total_batch, loss_train, acc_train, loss_val, acc_val, time_dif, improved_str))

            session.run(model.train_op, feed_dict=feed_dict)  # 运行优化
            total_batch += 1

            if total_batch - last_improved > require_improvement:
                # 验证集正确率长期不提升，提前结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break  # 跳出循环
        if flag:  # 同上
            break


def test():
    print("Loading test data...")
    start_time = time.time()
    x_test = read_data_x(test_dir, config.seq_length_w)
    y_test = read_data_y(test_y_dir)
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess=session, save_path=save_path)  # 读取保存的模型
    print('Testing...')
    loss_test, acc_test, pred_cls = evaluate(session, x_test, y_test)
    msg = 'Test Loss: {0:>6.2}, Test Acc: {1:>7.2%}'
    print(msg.format(loss_test, acc_test))
    hypotheses = postprocess(pred_cls, id2token_y)
    with open(test_hypothesis_dir, 'w', encoding='utf-8') as fout:
        fout.write("\n".join(hypotheses))
    # 评估
    print("Precision, Recall and F1-Score...")
    print(metrics.classification_report(y_test, pred_cls, digits=4))

    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] not in ['train', 'test']:
        raise ValueError("""usage: python run_cnn_c.py [train / test]""")
    print('Configuring CNN model...')
    config = TCNNConfig()
    model = TextCNN(config)
    _, id2token_y = load_vocab(vocab_y_dir)
    if sys.argv[1] == 'train':
        train()
    else:
        test()

