#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
FilePath: /dialog_system/retrieval/word2vec.py
Desciption: 训练word2vec model。
'''

import logging
import multiprocessing
import sys
from time import time
import os

import pandas as pd
from gensim.models import Word2Vec
from gensim.models.phrases import Phraser, Phrases
from gensim.models.callbacks import CallbackAny2Vec

sys.path.append('..')
from config import root_path, train_raw
from preprocessor import clean, read_file
import config


logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s",
                    datefmt="%H:%M:%S",
                    level=logging.INFO)


class callback(CallbackAny2Vec):
    '''Callback to print loss after each epoch.'''

    def __init__(self):
        self.epoch = 0
        self.loss_to_be_subed = 0

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        loss_now = loss - self.loss_to_be_subed
        self.loss_to_be_subed = loss
        print('Loss after epoch {}: {}'.format(self.epoch, loss_now))
        logging.info('Loss after epoch {}: {}'.format(self.epoch, loss_now))
        self.epoch += 1


def read_data(file_path):
    '''
    @description: 读取数据，清洗
    @param {type}
    file_path: 文件所在路径
    @return: Training samples.
    '''
    train = pd.DataFrame(read_file(file_path, True),
                         columns=['session_id', 'role', 'content'])
    train['clean_content'] = train['content'].apply(clean)
    return train


def train_w2v(train, to_file):
    '''
    @description: 训练word2vec model， 并保存
    @param {type}
    train: 数据集 DataFrame
    to_file: 模型保存路径
    @return: None
    '''
    sent = [row.split() for row in train['clean_content']]
    phrases = Phrases(sent, min_count=5, progress_per=10000)
    bigram = Phraser(phrases)
    sentences = bigram[sent]

    cores = multiprocessing.cpu_count()
    w2v_model = Word2Vec(min_count=2,
                         window=2,
                         size=300,
                         sample=6e-5,
                         alpha=0.03,
                         min_alpha=0.0007,
                         negative=15,
                         workers=cores - 1,
                         iter=15)

    t = time()
    w2v_model.build_vocab(sentences)
    print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))

    t = time()
    w2v_model.train(sentences,
                    total_examples=w2v_model.corpus_count,
                    epochs=w2v_model.iter,
                    report_delay=1,
                    compute_loss=True,
                    callbacks=[callback()])
    print('Time to train vocab: {} mins'.format(round((time() - t) / 60, 2)))

    w2v_model.save(to_file)


if __name__ == "__main__":
    train = read_data(config.train_raw)
    train_w2v(train, config.w2v_path)
