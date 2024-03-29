# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     sentences2words
   Author :        Xiaosong Zhou
   date：          2019/8/1
-------------------------------------------------
"""
__author__ = 'Xiaosong Zhou'

# 完成句子到分词句子的转换

import re
import numpy as np
import jieba

jieba.load_userdict('name_dict.txt')

SAVE_PATH = '../data/'


def get_content(file_dict, mode='r', encoding='utf-8'):
    contents = []
    with open(file_dict, mode=mode, encoding=encoding) as f:
        content = f.readline()
        while content:
            contents.append(content.strip())
            content = f.readline()
    return contents


def cut_word(sentence_file_path, save_file_path_w):
    contents_list = []
    with open(sentence_file_path, 'r', encoding='utf-8') as f:
        content = f.readline()
        while content:
            content = re.sub(r'\r', '', content)
            content = re.sub(r'\n', '', content)
            # 去除时间格式
            content = re.sub(r'([0-9]{4}年)?[0-9]{1,2}月([0-9]{1,2}日)?', '', content)
            content = re.sub(r'[0-9]{1,2}时([0-9]{1,2}分)?许?', '', content)

            # 去除法条中的形式
            content = re.sub(r'第.+条 ', '', content)

            # 去除事实部分的 审理查明，***指控这种
            content = re.sub(r'^.+指控，', '', content)
            content = re.sub(r'^.+指控：', '', content)
            content = re.sub(r'^.+诉称，', '', content)
            content = re.sub(r'^.+诉称：', '', content)
            content = re.sub(r'经.*审理查明，', '', content)
            content = re.sub(r'经.*审理查明：', '', content)
            content = re.sub(r'经.*审理查明', '', content)

            cut_list = jieba.cut(content)
            res_list = []
            for w in cut_list:
                if w in stopwords:
                    continue
                elif '省' in w:
                    continue
                elif '市' in w:
                    continue
                elif '镇' in w:
                    continue
                elif '村' in w:
                    continue
                elif '路' in w:
                    continue
                elif '县' in w:
                    continue
                elif '区' in w:
                    continue
                elif '城' in w:
                    continue
                elif '府' in w:
                    continue
                elif '庄' in w:
                    continue
                elif '道' in w:
                    continue
                elif '车' in w:
                    continue
                elif '店' in w:
                    continue
                elif '某' in w:
                    continue
                elif '辆' in w:
                    continue
                elif '房' in w:
                    continue
                elif '馆' in w:
                    continue
                elif '场' in w:
                    continue
                elif '街' in w:
                    continue
                elif '墙' in w:
                    continue
                elif '牌' in w:
                    continue
                else:
                    res_list.append(w)
            contents_list.append(res_list)
            content = f.readline()

    with open(save_file_path_w, 'w', encoding='utf-8') as f:
        for line in contents_list:
            f.write(' '.join(line) + '\n')


def cut_word_en(sentence_file_path, save_file_path_w):
    contents_list = []
    # with open(sentence_file_path, 'r', encoding='utf-8') as f:
    #     contents_list_row = f.readlines()
    with open(sentence_file_path, 'r', encoding='utf-8') as f:
        content = f.readline()
        while content:
            # cut_list = content.split()
            content_new = content.rstrip('\n')
            content_new = content_new.lower()
            cut_list = re.split('\.| |,', content_new)
            while '' in cut_list:
                cut_list.remove('')
            contents_list.append(cut_list)
            content = f.readline()
    line_sum = 0
    with open(save_file_path_w, 'w', encoding='utf-8') as f:
        for line in contents_list:
            line_sum += 1
            f.write(' '.join(line) + '\n')
    print('line: ' + str(line_sum))


def cut_character(sentence_file_path, save_file_path_c):
    c_list = []
    with open(sentence_file_path, 'r', encoding='utf-8') as f:
        content = f.readline()
        while content:
            content = re.sub(r'\r', '', content)
            content = re.sub(r'\n', '', content)

            # 去除法条中的形式
            content = re.sub(r'第.+条 ', '', content)

            # 去除事实部分的 审理查明，***指控这种
            content = re.sub(r'^.+指控，', '', content)
            content = re.sub(r'^.+指控：', '', content)
            content = re.sub(r'^.+诉称，', '', content)
            content = re.sub(r'^.+诉称：', '', content)
            content = re.sub(r'经.*审理查明，', '', content)
            content = re.sub(r'经.*审理查明：', '', content)
            content = re.sub(r'经.*审理查明', '', content)
            c_list.append(list(content))
            content = f.readline()
    with open(save_file_path_c, 'w', encoding='utf-8') as f:
        for line in c_list:
            f.write(' '.join(line) + '\n')


if __name__ == '__main__':
    stopwords = get_content('stop_word.txt', 'r', 'utf-8')
    stopnames = get_content('name_dict.txt', 'r', 'utf-8')
    cut_word('../data/cnews/val_x.txt',
             '../data/cnews/seg/val_x_w.txt')
    # cut_word_en('../data/imdb/val_x.txt',
    #             '../data/imdb/seg/val_x_w.txt')
    # cut_character('../data/legal_domain/val_x.txt',
    #               '../data/legal_domain/seg/val_x_c.txt')

    cut_word('../data/cnews/train_x.txt',
             '../data/cnews/seg/train_x_w.txt')
    # cut_word_en('../data/imdb/train_x.txt',
    #             '../data/imdb/seg/train_x_w.txt')
    # cut_character('../data/legal_domain/train_x.txt',
    #               '../data/legal_domain/seg/train_x_c.txt')

    cut_word('../data/cnews/test_x.txt',
             '../data/cnews/seg/test_x_w.txt')
    # cut_word_en('../data/imdb/test_x.txt',
    #             '../data/imdb/seg/test_x_w.txt')
    # cut_character('../data/legal_domain/test_x.txt',
    #               '../data/legal_domain/seg/test_x_c.txt')

