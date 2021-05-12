#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
FilePath: /dialog_system/config.py
Desciption: 配置文件。
'''

import torch
import os
root_path = os.path.abspath(os.path.dirname(__file__))

train_raw = os.path.join(root_path, 'data/chat.txt')
dev_raw = os.path.join(root_path, 'data/开发集.txt')
test_raw = os.path.join(root_path, 'data/测试集.txt')
ware_path = os.path.join(root_path, 'data/ware.txt')

base_chinese_bert_vocab = os.path.join(root_path, 'lib/bert/vocab.txt')
sep = '[SEP]'

''' Data '''
# main
train_path = os.path.join(root_path, 'data/train_no_blank.csv')
dev_path = os.path.join(root_path, 'data/dev.csv')
test_path = os.path.join(root_path, 'data/test.csv')
# intention
business_train = os.path.join(root_path, 'data/intention/business.train')
business_test = os.path.join(root_path, 'data/intention/business.test')
keyword_path = os.path.join(root_path, 'data/intention/key_word.txt')


''' Intention '''
# fasttext
ft_path = os.path.join(root_path, "model/intention/fastext")

''' Retrival '''
# Embedding
w2v_path = os.path.join(root_path, "model/retrieval/word2vec")

# HNSW parameters
ef_construction = 3000  # ef_construction defines a construction time/accuracy trade-off
M = 64  # M defines tha maximum number of outgoing connections in the graph
hnsw_path = os.path.join(root_path, 'model/retrieval/hnsw_index')

# 通用配置
is_cuda = True
if is_cuda:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

max_sequence_length = 103
max_length = 416
batch_size = 32
lr = 2e-05
max_grad_norm = 10.0
log_path = os.path.join(root_path, "log/seq2seq.log")
bert_chinese_model_path = os.path.join(root_path, "lib/bert/pytorch_model.bin")
gradient_accumulation = 1


